"""
Google Analytics Integration Module

This module provides functionality to connect to Google Analytics Data API (GA4)
and retrieve analytics data for further processing and analysis.

Features:
- OAuth2 authentication with Google
- Service account authentication
- Data retrieval from GA4 properties
- Support for various metrics and dimensions
- Data preprocessing and formatting
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


class GoogleAnalyticsData:
    """
    Class for retrieving and processing Google Analytics 4 data.
    This class provides a simplified interface for the Streamlit application.
    """
    
    def __init__(self, property_id: str, service_account_key: Optional[Dict] = None):
        """
        Initialize the Google Analytics Data client.
        
        Args:
            property_id: GA4 property ID (format: "123456789" or "properties/123456789")
            service_account_key: Optional service account key as a dictionary
        """
        self.property_id = property_id
        self.service_account_key = service_account_key
        self.connector = None
        self._initialize_connector()
    
    def _initialize_connector(self):
        """
        Initialize the GA connector with service account authentication.
        """
        try:
            # If service account key is provided as a dictionary
            if self.service_account_key:
                # Create a temporary file to store the service account key
                temp_key_path = os.path.join(os.path.expanduser('~'), '.temp_sa_key.json')
                with open(temp_key_path, 'w') as f:
                    json.dump(self.service_account_key, f)
                
                # Initialize connector with service account
                self.connector = GoogleAnalyticsConnector(
                    credentials_path=temp_key_path,
                    use_service_account=True
                )
                
                # Authenticate
                success = self.connector.authenticate()
                
                # Remove temporary file
                if os.path.exists(temp_key_path):
                    os.remove(temp_key_path)
                
                if not success:
                    raise ValueError("Failed to authenticate with service account")
            else:
                # For testing or development without service account
                self.connector = None
                print("Warning: No service account key provided. Some functionality may be limited.")
        
        except Exception as e:
            print(f"Error initializing GA connector: {str(e)}")
            self.connector = None
    
    def get_report(self, 
                  start_date: str, 
                  end_date: str, 
                  dimensions: List[str], 
                  metrics: List[str]) -> pd.DataFrame:
        """
        Get a report from Google Analytics.
        
        Args:
            start_date: Start date in format 'YYYY-MM-DD' or GA4 relative format (e.g., '7daysAgo')
            end_date: End date in format 'YYYY-MM-DD' or GA4 relative format (e.g., 'yesterday')
            dimensions: List of dimensions to include in the report
            metrics: List of metrics to include in the report
            
        Returns:
            pandas.DataFrame containing the report data
        """
        if not self.connector:
            # Return empty DataFrame if connector is not initialized
            print("GA connector not initialized. Returning empty DataFrame.")
            return pd.DataFrame()
        
        try:
            # Create date range
            date_range = {
                'start_date': start_date,
                'end_date': end_date
            }
            
            # Run the report
            df = self.connector.run_report(
                property_id=self.property_id,
                date_range=date_range,
                dimensions=dimensions,
                metrics=metrics
            )
            
            return df
            
        except Exception as e:
            print(f"Error getting report: {str(e)}")
            # Return empty DataFrame on error
            return pd.DataFrame()
    
    def get_available_metrics_and_dimensions(self) -> Dict[str, List[str]]:
        """
        Get lists of available metrics and dimensions for GA4.
        
        Note: This is a simplified implementation that returns common metrics and dimensions.
        In a production environment, you would use the Analytics Admin API to get the actual
        available metrics and dimensions for the specific property.
        
        Returns:
            Dictionary with 'metrics' and 'dimensions' keys
        """
        # Common GA4 metrics
        metrics = [
            'activeUsers',
            'newUsers',
            'sessions',
            'averageSessionDuration',
            'screenPageViews',
            'engagementRate',
            'eventCount',
            'conversions',
            'totalRevenue',
            'itemsViewed',
            'itemsAddedToCart',
            'itemsCheckedOut',
            'itemsPurchased',
            'itemRevenue'
        ]
        
        # Common GA4 dimensions
        dimensions = [
            'date',
            'deviceCategory',
            'country',
            'region',
            'city',
            'sessionSource',
            'sessionMedium',
            'sessionCampaignName',
            'pageTitle',
            'pagePathPlusQueryString',
            'itemName',
            'itemId',
            'itemCategory',
            'itemBrand',
            'transactionId'
        ]
        
        return {
            'metrics': metrics,
            'dimensions': dimensions
        }
