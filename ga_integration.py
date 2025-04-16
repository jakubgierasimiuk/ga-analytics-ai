"""
Google Analytics Integration Module

This module provides functionality to connect to Google Analytics Data API (GA4)
and retrieve analytics data for further processing and analysis.

Features:
- OAuth2 authentication with Google
- Data retrieval from GA4 properties
- Support for various metrics and dimensions
- Data preprocessing and formatting
- Alternative data import from CSV/JSON files
"""

import os
import json
import datetime
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    RunReportRequest,
    RunReportResponse,
    BatchRunReportsRequest,
    BatchRunReportsResponse,
)
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

class GoogleAnalyticsConnector:
    """
    Class for connecting to Google Analytics Data API and retrieving data.
    Supports both OAuth2 authentication and service account authentication.
    """
    
    # GA4 API scopes required for data access
    SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
    
    def __init__(self, 
                 credentials_path: Optional[str] = None,
                 use_service_account: bool = False,
                 token_path: Optional[str] = None):
        """
        Initialize the Google Analytics connector.
        
        Args:
            credentials_path: Path to credentials JSON file (OAuth client ID or service account)
            use_service_account: Whether to use service account authentication
            token_path: Path to save/load OAuth tokens
        """
        self.credentials_path = credentials_path
        self.use_service_account = use_service_account
        self.token_path = token_path or os.path.join(os.path.expanduser('~'), '.ga_token.json')
        self.credentials = None
        self.client = None
    
    def authenticate(self) -> bool:
        """
        Authenticate with Google Analytics API.
        
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        try:
            if self.use_service_account and self.credentials_path:
                # Service account authentication
                self.credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path, scopes=self.SCOPES)
                
            else:
                # OAuth2 authentication
                self.credentials = self._get_oauth_credentials()
                
            # Create the Analytics Data API client
            self.client = BetaAnalyticsDataClient(credentials=self.credentials)
            return True
            
        except Exception as e:
            print(f"Authentication failed: {str(e)}")
            return False
    
    def _get_oauth_credentials(self) -> Credentials:
        """
        Get OAuth2 credentials, either from saved token or through OAuth flow.
        
        Returns:
            Credentials: Google OAuth credentials
        """
        creds = None
        
        # Try to load saved token
        if os.path.exists(self.token_path):
            with open(self.token_path, 'r') as token:
                creds = Credentials.from_authorized_user_info(
                    json.load(token), self.SCOPES)
        
        # If no valid credentials available, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_path:
                    raise ValueError("Credentials path must be provided for OAuth authentication")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())
                
        return creds
    
    def get_available_properties(self) -> List[Dict[str, str]]:
        """
        Get a list of available GA4 properties for the authenticated user.
        
        Note: This requires the Analytics Admin API, which is separate from the Data API.
        This is a simplified implementation and may need to be expanded.
        
        Returns:
            List of dictionaries containing property information
        """
        # This is a placeholder - in a real implementation, you would use the Analytics Admin API
        # to retrieve the list of properties the user has access to
        
        # For now, we'll return a message indicating this needs to be implemented
        print("Note: Retrieving available properties requires the Analytics Admin API.")
        print("This functionality needs to be implemented separately.")
        
        return []
    
    def run_report(self, 
                  property_id: str,
                  date_range: Union[Dict[str, str], List[Dict[str, str]]],
                  metrics: List[str],
                  dimensions: Optional[List[str]] = None,
                  limit: Optional[int] = None,
                  offset: Optional[int] = None) -> pd.DataFrame:
        """
        Run a report on Google Analytics data and return results as a DataFrame.
        
        Args:
            property_id: GA4 property ID (format: "properties/XXXXXX")
            date_range: Date range for the query, either a single dict or list of dicts
                        with 'start_date' and 'end_date' keys (format: 'YYYY-MM-DD')
            metrics: List of metrics to retrieve (e.g., 'sessions', 'activeUsers')
            dimensions: Optional list of dimensions (e.g., 'date', 'country', 'deviceCategory')
            limit: Optional row limit for the report
            offset: Optional offset for pagination
            
        Returns:
            pandas.DataFrame containing the report data
        """
        if not self.client:
            raise ValueError("Client not initialized. Call authenticate() first.")
        
        # Ensure property_id has the correct format
        if not property_id.startswith('properties/'):
            property_id = f'properties/{property_id}'
        
        # Prepare date ranges
        date_ranges = []
        if isinstance(date_range, dict):
            date_ranges.append(DateRange(
                start_date=date_range['start_date'],
                end_date=date_range['end_date']
            ))
        else:
            for dr in date_range:
                date_ranges.append(DateRange(
                    start_date=dr['start_date'],
                    end_date=dr['end_date']
                ))
        
        # Prepare dimensions
        dimension_list = []
        if dimensions:
            for dim in dimensions:
                dimension_list.append(Dimension(name=dim))
        
        # Prepare metrics
        metric_list = []
        for metric in metrics:
            metric_list.append(Metric(name=metric))
        
        # Create the request
        request = RunReportRequest(
            property=property_id,
            date_ranges=date_ranges,
            dimensions=dimension_list,
            metrics=metric_list,
            limit=limit,
            offset=offset
        )
        
        # Make the API call
        response = self.client.run_report(request)
        
        # Convert the response to a DataFrame
        return self._response_to_dataframe(response)
    
    def batch_run_reports(self,
                         property_id: str,
                         requests: List[Dict[str, Any]]) -> List[pd.DataFrame]:
        """
        Run multiple reports in a single batch request.
        
        Args:
            property_id: GA4 property ID (format: "properties/XXXXXX")
            requests: List of report request configurations
            
        Returns:
            List of pandas.DataFrame objects containing the report data
        """
        if not self.client:
            raise ValueError("Client not initialized. Call authenticate() first.")
        
        # Ensure property_id has the correct format
        if not property_id.startswith('properties/'):
            property_id = f'properties/{property_id}'
        
        # Prepare individual report requests
        report_requests = []
        for req in requests:
            # Prepare date ranges
            date_ranges = []
            date_range = req.get('date_range', {'start_date': '7daysAgo', 'end_date': 'today'})
            
            if isinstance(date_range, dict):
                date_ranges.append(DateRange(
                    start_date=date_range['start_date'],
                    end_date=date_range['end_date']
                ))
            else:
                for dr in date_range:
                    date_ranges.append(DateRange(
                        start_date=dr['start_date'],
                        end_date=dr['end_date']
                    ))
            
            # Prepare dimensions
            dimension_list = []
            dimensions = req.get('dimensions', [])
            for dim in dimensions:
                dimension_list.append(Dimension(name=dim))
            
            # Prepare metrics
            metric_list = []
            metrics = req.get('metrics', [])
            for metric in metrics:
                metric_list.append(Metric(name=metric))
            
            # Create the request
            report_request = RunReportRequest(
                dimensions=dimension_list,
                metrics=metric_list,
                date_ranges=date_ranges,
                limit=req.get('limit'),
                offset=req.get('offset')
            )
            
            report_requests.append(report_request)
        
        # Create the batch request
        batch_request = BatchRunReportsRequest(
            property=property_id,
            requests=report_requests
        )
        
        # Make the API call
        response = self.client.batch_run_reports(batch_request)
        
        # Convert the responses to DataFrames
        dataframes = []
        for report in response.reports:
            dataframes.append(self._response_to_dataframe(report))
            
        return dataframes
    
    def _response_to_dataframe(self, response: Union[RunReportResponse, Any]) -> pd.DataFrame:
        """
        Convert a GA4 API response to a pandas DataFrame.
        
        Args:
            response: The API response object
            
        Returns:
            pandas.DataFrame containing the report data
        """
        # Extract dimension and metric headers
        dimension_headers = [header.name for header in response.dimension_headers]
        metric_headers = [header.name for header in response.metric_headers]
        
        all_rows = []
        for row in response.rows:
            row_data = {}
            
            # Add dimensions
            for i, dimension in enumerate(row.dimension_values):
                row_data[dimension_headers[i]] = dimension.value
            
            # Add metrics
            for i, metric in enumerate(row.metric_values):
                row_data[metric_headers[i]] = metric.value
            
            all_rows.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(all_rows)
        
        # Convert numeric columns
        for metric in metric_headers:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors='ignore')
        
        return df
    
    def import_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Import GA data from a file (CSV or JSON).
        
        Args:
            file_path: Path to the file to import
            
        Returns:
            pandas.DataFrame containing the imported data
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .csv or .json")


class GADataProcessor:
    """
    Class for processing and analyzing Google Analytics data.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the data processor.
        
        Args:
            data: Optional initial DataFrame with GA data
        """
        self.data = data
    
    def set_data(self, data: pd.DataFrame):
        """
        Set the data to be processed.
        
        Args:
            data: DataFrame with GA data
        """
        self.data = data
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the data.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.data is None:
            raise ValueError("No data available. Set data first.")
        
        # Identify numeric columns
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        # Calculate summary statistics
        summary = {
            'row_count': len(self.data),
            'metrics': {}
        }
        
        for col in numeric_cols:
            summary['metrics'][col] = {
                'sum': self.data[col].sum(),
                'mean': self.data[col].mean(),
                'median': self.data[col].median(),
                'min': self.data[col].min(),
                'max': self.data[col].max()
            }
        
        return summary
    
    def detect_anomalies(self, column: str, threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect anomalies in a specific column using Z-score.
        
        Args:
            column: Column name to check for anomalies
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame with rows containing anomalies
        """
        if self.data is None:
            raise ValueError("No data available. Set data first.")
        
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        # Calculate Z-scores
        mean = self.data[column].mean()
        std = self.data[column].std()
        
        if std == 0:
            return pd.DataFrame()  # No variation, no anomalies
        
        self.data['z_score'] = (self.data[column] - mean) / std
        
        # Filter anomalies
        anomalies = self.data[abs(self.data['z_score']) > threshold].copy()
        
        # Remove temporary column
        self.data.drop('z_score', axis=1, inplace=True)
        
        return anomalies
    
    def aggregate_by_dimension(self, 
                              dimension: str, 
                              metrics: List[str],
                              agg_func: str = 'sum') -> pd.DataFrame:
        """
        Aggregate data by a specific dimension.
        
        Args:
            dimension: Dimension to aggregate by
            metrics: List of metrics to aggregate
            agg_func: Aggregation function ('sum', 'mean', 'median', 'min', 'max')
            
        Returns:
            Aggregated DataFrame
        """
        if self.data is None:
            raise ValueError("No data available. Set data first.")
        
        if dimension not in self.data.columns:
            raise ValueError(f"Dimension '{dimension}' not found in data")
        
        for metric in metrics:
            if metric not in self.data.columns:
                raise ValueError(f"Metric '{metric}' not found in data")
        
        # Perform aggregation
        return self.data.groupby(dimension)[metrics].agg(agg_func).reset_index()
    
    def calculate_growth(self, 
                        metric: str, 
                        date_column: str = 'date',
                        periods: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate growth rates for a metric over time.
        
        Args:
            metric: Metric to calculate growth for
            date_column: Column containing dates
            periods: Optional list of period names to include
            
        Returns:
            DataFrame with growth rates
        """
        if self.data is None:
            raise ValueError("No data available. Set data first.")
        
        if metric not in self.data.columns:
            raise ValueError(f"Metric '{metric}' not found in data")
        
        if date_column not in self.data.columns:
            raise ValueError(f"Date column '{date_column}' not found in data")
        
        # Ensure date column is datetime
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        # Sort by date
        sorted_data = self.data.sort_values(date_column)
        
        # Calculate growth
        sorted_data[f'{metric}_growth'] = sorted_data[metric].pct_change() * 100
        
        # Filter periods if specified
        if periods:
            return sorted_data[sorted_data[date_column].isin(periods)]
        
        return sorted_data
    
    def prepare_data_for_llm(self) -> Dict[str, Any]:
        """
        Prepare data in a format suitable for sending to an LLM.
        
        Returns:
            Dictionary with processed data and metadata
        """
        if self.data is None:
            raise ValueError("No data available. Set data first.")
        
        # Get basic info about the data
        dimensions = self.data.select_dtypes(exclude=['number']).columns.tolist()
        metrics = self.data.select_dtypes(include=['number']).columns.tolist()
        
        # Get summary statistics
        summary = self.get_summary_statistics()
        
        # Prepare sample data (limit rows to avoid token limits)
        sample_rows = min(20, len(self.data))
        sample_data = self.data.head(sample_rows).to_dict(orient='records')
        
        # Prepare result
        result = {
            'metadata': {
                'dimensions': dimensions,
                'metrics': metrics,
                'row_count': len(self.data),
                'sample_size': sample_rows
            },
            'summary': summary,
            'sample_data': sample_data
        }
        
        return result


# Example usage functions

def example_oauth_authentication():
    """Example of authenticating with OAuth."""
    # Path to your OAuth client ID JSON file
    credentials_path = 'path/to/your/client_secret.json'
    
    # Create connector
    connector = GoogleAnalyticsConnector(credentials_path=credentials_path)
    
    # Authenticate
    if connector.authenticate():
        print("Authentication successful!")
    else:
        print("Authentication failed.")
    
    return connector

def example_service_account_authentication():
    """Example of authenticating with a service account."""
    # Path to your service account JSON file
    credentials_path = 'path/to/your/service-account.json'
    
    # Create connector
    connector = GoogleAnalyticsConnector(
        credentials_path=credentials_path,
        use_service_account=True
    )
    
    # Authenticate
    if connector.authenticate():
        print("Authentication successful!")
    else:
        print("Authentication failed.")
    
    return connector

def example_run_report(connector, property_id):
    """Example of running a basic report."""
    # Define date range
    date_range = {
        'start_date': '30daysAgo',
        'end_date': 'today'
    }
    
    # Define metrics and dimensions
    metrics = ['activeUsers', 'sessions', 'screenPageViews']
    dimensions = ['date', 'deviceCategory']
    
    # Run the report
    df = connector.run_report(
        property_id=property_id,
        date_range=date_range,
        metrics=metrics,
        dimensions=dimensions
    )
    
    print(f"Retrieved {len(df)} rows of data.")
    print(df.head())
    
    return df

def example_data_processing(df):
    """Example of processing GA data."""
    # Create processor
    processor = GADataProcessor(df)
    
    # Get summary statistics
    summary = processor.get_summary_statistics()
    print("Summary statistics:")
    print(json.dumps(summary, indent=2))
    
    # Aggregate by device category
    agg_data = processor.aggregate_by_dimension(
        dimension='deviceCategory',
        metrics=['activeUsers', 'sessions', 'screenPageViews']
    )
    print("\nAggregated by device category:")
    print(agg_data)
    
    # Detect anomalies in sessions
    anomalies = processor.detect_anomalies(column='sessions')
    print("\nAnomaly detection for sessions:")
    print(anomalies)
    
    # Prepare data for LLM
    llm_data = processor.prepare_data_for_llm()
    print("\nData prepared for LLM:")
    print(json.dumps(llm_data, indent=2)[:500] + "...")  # Truncated for brevity
    
    return llm_data

def main():
    """Main function to demonstrate the GA integration."""
    # This is just an example - in a real application, you would integrate this
    # with your Flask/FastAPI backend and database
    
    print("Google Analytics Integration Example")
    print("-----------------------------------")
    
    # Uncomment and modify these lines to run the examples
    # connector = example_oauth_authentication()
    # property_id = "YOUR_GA4_PROPERTY_ID"  # e.g., "123456789"
    # df = example_run_report(connector, property_id)
    # llm_data = example_data_processing(df)
    
    print("\nNote: This is a demonstration module. In a real application,")
    print("you would need to provide valid credentials and property ID.")

if __name__ == "__main__":
    main()
