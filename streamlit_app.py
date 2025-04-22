import streamlit as st
import os
import json
import datetime
import pandas as pd
import plotly.express as px
from google.oauth2 import service_account
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest,
    DateRange,
    Dimension,
    Metric
)
from openai import OpenAI  # Updated import for OpenAI
import re
import uuid
import shutil
import sys

# Add the current directory to the path to import the GA4 metrics mapping module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from ga4_metrics_mapping_improved import (
        select_metrics_from_prompt, 
        get_analysis_config_for_type, 
        GA4_METRICS_MAPPING, 
        VALID_GA4_METRICS,
        VALID_GA4_DIMENSIONS,
        is_compatible_combination,
        get_safe_metrics_and_dimensions
    )
except ImportError:
    # Fallback if the module is not available
    # Define basic mappings directly in the code
    GA4_METRICS_MAPPING = {
        "General Overview": {
            "dimensions": ["date"],
            "metrics": ["activeUsers", "sessions", "screenPageViews", "engagementRate"]
        },
        "Audience Analysis": {
            "dimensions": ["deviceCategory", "country", "city"],
            "metrics": ["activeUsers", "newUsers", "sessions", "averageSessionDuration"]
        },
        "Acquisition Analysis": {
            "dimensions": ["sessionSource", "sessionMedium", "sessionCampaignName"],
            "metrics": ["activeUsers", "sessions", "engagementRate", "averageSessionDuration"]
        },
        "Behavior Analysis": {
            "dimensions": ["pagePath", "deviceCategory"],
            "metrics": ["screenPageViews", "averageSessionDuration", "engagementRate", "eventCount"]
        },
        "Conversion Analysis": {
            "dimensions": ["date"],
            "metrics": ["conversions", "eventCount", "eventValue"]
        }
    }
    
    VALID_GA4_METRICS = {
        "activeUsers": "The number of distinct users who visited your site or app.",
        "newUsers": "The number of users who visited your site or app for the first time.",
        "totalUsers": "The total number of users who visited your site or app.",
        "sessions": "The number of sessions that began on your site or app.",
        "averageSessionDuration": "The average duration (in seconds) of users' sessions.",
        "screenPageViews": "The number of app screens or web pages your users viewed.",
        "engagedSessions": "The number of sessions that lasted longer than 10 seconds, or had a conversion event, or had 2 or more screen or page views.",
        "engagementRate": "The percentage of engaged sessions (Engaged sessions divided by Sessions).",
        "eventCount": "The count of events.",
        "eventValue": "The sum of the event parameter named value.",
        "conversions": "The number of conversions."
    }
    
    VALID_GA4_DIMENSIONS = {
        "date": "The date of the session (YYYYMMDD).",
        "deviceCategory": "The device category (mobile, tablet, desktop).",
        "country": "The country from which sessions originated.",
        "city": "The city from which sessions originated.",
        "sessionSource": "The source of the session (e.g., 'google', 'facebook').",
        "sessionMedium": "The medium of the session (e.g., 'organic', 'cpc').",
        "sessionCampaignName": "The campaign name of the session.",
        "pagePath": "The path of the page."
    }
    
    def is_compatible_combination(dimensions, metrics):
        """Fallback function if module is not available"""
        return True
    
    def get_safe_metrics_and_dimensions(dimensions, metrics):
        """Fallback function if module is not available"""
        return {
            "dimensions": ["date"],
            "metrics": ["activeUsers", "sessions"]
        }
    
    def select_metrics_from_prompt(prompt: str) -> dict:
        """Fallback function if module is not available"""
        return GA4_METRICS_MAPPING["General Overview"]
    
    def get_analysis_config_for_type(analysis_type: str) -> dict:
        """Fallback function if module is not available"""
        if analysis_type in GA4_METRICS_MAPPING:
            return GA4_METRICS_MAPPING[analysis_type]
        return GA4_METRICS_MAPPING["General Overview"]

# Set page configuration
st.set_page_config(
    page_title="GA Analytics AI",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'page' not in st.session_state:
    st.session_state.page = "login"
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        'openai': {'api_key': ''}
    }
if 'ga_accounts' not in st.session_state:
    st.session_state.ga_accounts = {}
if 'reports' not in st.session_state:
    st.session_state.reports = []
if 'prompts' not in st.session_state:
    st.session_state.prompts = {
        'Default Prompt': "Analyze the following Google Analytics data and provide insights. Focus on trends, anomalies, and actionable recommendations."
    }

# Create directories if they don't exist
os.makedirs('credentials', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Save session state to disk
def save_session_state():
    # Save API keys
    with open('credentials/api_keys.json', 'w') as f:
        json.dump(st.session_state.api_keys, f)
    
    # Save GA accounts (excluding credentials path which is stored separately)
    ga_accounts_to_save = {}
    for account_name, account_data in st.session_state.ga_accounts.items():
        ga_accounts_to_save[account_name] = {
            'property_id': account_data['property_id']
        }
    
    with open('credentials/ga_accounts.json', 'w') as f:
        json.dump(ga_accounts_to_save, f)
    
    # Save prompts
    with open('credentials/prompts.json', 'w') as f:
        json.dump(st.session_state.prompts, f)
    
    # Save reports
    with open('reports/reports.json', 'w') as f:
        json.dump(st.session_state.reports, f)

# Load session state from disk
def load_session_state():
    # Load API keys
    try:
        with open('credentials/api_keys.json', 'r') as f:
            st.session_state.api_keys = json.load(f)
    except FileNotFoundError:
        pass
    
    # Load GA accounts
    try:
        with open('credentials/ga_accounts.json', 'r') as f:
            ga_accounts_from_file = json.load(f)
            
            # Reconstruct GA accounts with credentials paths
            for account_name, account_data in ga_accounts_from_file.items():
                credentials_path = f'credentials/ga_{account_name.lower().replace(" ", "_")}.json'
                if os.path.exists(credentials_path):
                    if account_name not in st.session_state.ga_accounts:
                        st.session_state.ga_accounts[account_name] = {
                            'property_id': account_data['property_id'],
                            'credentials_path': credentials_path
                        }
    except FileNotFoundError:
        pass
    
    # Load prompts
    try:
        with open('credentials/prompts.json', 'r') as f:
            st.session_state.prompts = json.load(f)
    except FileNotFoundError:
        pass
    
    # Load reports
    try:
        with open('reports/reports.json', 'r') as f:
            st.session_state.reports = json.load(f)
    except FileNotFoundError:
        pass

# Load session state at startup
load_session_state()

# Function to convert web client credentials to installed client format
def convert_web_to_installed_client(web_credentials):
    if 'web' in web_credentials:
        # This is a web client credentials file, convert to installed client format
        installed_credentials = {
            "installed": {
                "client_id": web_credentials["web"]["client_id"],
                "project_id": web_credentials["web"]["project_id"],
                "auth_uri": web_credentials["web"]["auth_uri"],
                "token_uri": web_credentials["web"]["token_uri"],
                "auth_provider_x509_cert_url": web_credentials["web"]["auth_provider_x509_cert_url"],
                "client_secret": web_credentials["web"]["client_secret"],
                "redirect_uris": web_credentials["web"]["redirect_uris"]
            }
        }
        return installed_credentials
    return web_credentials

# Function to check if a JSON file is a service account key
def is_service_account_key(json_data):
    required_fields = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id"]
    return all(field in json_data for field in required_fields) and json_data.get("type") == "service_account"

# Navigation functions
def navigate_to(page):
    st.session_state.page = page

# Render sidebar navigation
def render_sidebar():
    with st.sidebar:
        st.title("Navigation")
        
        if st.button("Dashboard"):
            navigate_to("dashboard")
        
        if st.button("New Analysis"):
            navigate_to("new_analysis")
        
        if st.button("Report History"):
            navigate_to("report_history")
        
        if st.button("Prompt Library"):
            navigate_to("prompt_library")
        
        if st.button("Settings"):
            navigate_to("settings")
        
        st.markdown("---")
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            navigate_to("login")

# Login page
def render_login():
    st.title("Login")
    
    email = st.text_input("Email")
    name = st.text_input("Name")
    
    if st.button("Login"):
        if email and name:
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.session_state.user_name = name
            navigate_to("dashboard")
        else:
            st.error("Please enter both email and name")

# Dashboard page
def render_dashboard():
    render_sidebar()
    
    st.title("Dashboard")
    
    if not st.session_state.ga_accounts:
        st.warning("No Google Analytics accounts configured. Please add an account in Settings.")
        return
    
    # Display recent reports
    st.header("Recent Reports")
    if st.session_state.reports:
        for i, report in enumerate(reversed(st.session_state.reports[-5:])):
            with st.expander(f"{report['title']} - {report['date']}"):
                st.write(f"**Analysis Type:** {report['analysis_type']}")
                st.write(f"**Description:** {report['description']}")
                st.write(f"**Account:** {report['account']}")
                st.write(f"**Date Range:** {report['date_range']}")
                st.write("**Insights:**")
                st.write(report['insights'])
    else:
        st.info("No reports yet. Create a new analysis to generate reports.")
    
    # Display GA accounts
    st.header("Connected Google Analytics Accounts")
    for account_name in st.session_state.ga_accounts:
        st.write(f"- {account_name} (Property ID: {st.session_state.ga_accounts[account_name]['property_id']})")

# New Analysis page
def render_new_analysis():
    render_sidebar()
    
    st.title("New Analysis")
    
    if not st.session_state.ga_accounts:
        st.warning("No Google Analytics accounts configured. Please add an account in Settings.")
        return
    
    # Analysis configuration
    st.header("Select Analysis Type")
    analysis_type = st.selectbox(
        "Select Analysis Type",
        list(GA4_METRICS_MAPPING.keys())
    )
    
    # Prompt selection
    st.header("Analysis Prompt")
    prompt_name = st.selectbox(
        "Select Prompt",
        list(st.session_state.prompts.keys())
    )
    prompt = st.session_state.prompts[prompt_name]
    
    # Custom prompt option
    use_custom_prompt = st.checkbox("Use custom prompt")
    if use_custom_prompt:
        custom_prompt = st.text_area("Custom Prompt", prompt)
        prompt = custom_prompt
    
    # Report details
    st.header("Report Details")
    report_title = st.text_input("Report Title", f"{analysis_type} - {datetime.datetime.now().strftime('%Y-%m-%d')}")
    report_description = st.text_area("Report Description", f"Analysis of {analysis_type.lower()} data from Google Analytics.")
    
    # GA account selection
    st.header("Google Analytics Account")
    account_name = st.selectbox(
        "Select Account",
        list(st.session_state.ga_accounts.keys())
    )
    
    # Date range selection
    st.header("Date Range")
    date_range = st.selectbox(
        "Select Date Range",
        ["Last 7 days", "Last 30 days", "Last 90 days", "Last 12 months"]
    )
    
    # Run analysis button
    if st.button("Run Analysis"):
        with st.spinner("Running analysis... This may take a few moments."):
            try:
                # Set date range
                end_date = datetime.datetime.now().date()
                if date_range == "Last 7 days":
                    start_date = end_date - datetime.timedelta(days=7)
                elif date_range == "Last 30 days":
                    start_date = end_date - datetime.timedelta(days=30)
                elif date_range == "Last 90 days":
                    start_date = end_date - datetime.timedelta(days=90)
                else:  # Last 12 months
                    start_date = end_date - datetime.timedelta(days=365)
                
                # Set metrics and dimensions based on analysis type or prompt
                if use_custom_prompt:
                    # Use intelligent metric selection based on prompt
                    analysis_config = select_metrics_from_prompt(prompt)
                    st.info(f"Selected metrics based on prompt: {', '.join(analysis_config['metrics'])}")
                    st.info(f"Selected dimensions based on prompt: {', '.join(analysis_config['dimensions'])}")
                else:
                    # Use predefined metrics for the selected analysis type
                    analysis_config = get_analysis_config_for_type(analysis_type)
                
                # Verify compatibility of metrics and dimensions
                if not is_compatible_combination(analysis_config['dimensions'], analysis_config['metrics']):
                    st.warning("The selected combination of metrics and dimensions is not compatible. Using safe alternatives.")
                    analysis_config = get_safe_metrics_and_dimensions(analysis_config['dimensions'], analysis_config['metrics'])
                    st.info(f"Using metrics: {', '.join(analysis_config['metrics'])}")
                    st.info(f"Using dimensions: {', '.join(analysis_config['dimensions'])}")
                
                # Get GA account details
                account_details = st.session_state.ga_accounts[account_name]
                property_id = account_details['property_id']
                credentials_path = account_details['credentials_path']
                
                # Initialize GA client with service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/analytics.readonly"]
                )
                client = BetaAnalyticsDataClient(credentials=credentials)
                
                # Build dimensions and metrics for the request
                dimensions_list = [Dimension(name=dim) for dim in analysis_config['dimensions']]
                metrics_list = [Metric(name=metric) for metric in analysis_config['metrics']]
                
                # Create the report request
                request = RunReportRequest(
                    property=f"properties/{property_id}",
                    dimensions=dimensions_list,
                    metrics=metrics_list,
                    date_ranges=[DateRange(start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))]
                )
                
                # Run the report
                response = client.run_report(request)
                
                # Process the response
                data = []
                for row in response.rows:
                    row_data = {}
                    for i, dimension in enumerate(row.dimension_values):
                        row_data[analysis_config['dimensions'][i]] = dimension.value
                    for i, metric in enumerate(row.metric_values):
                        row_data[analysis_config['metrics'][i]] = metric.value
                    data.append(row_data)
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                
                # Generate insights using OpenAI
                api_key = st.session_state.api_keys['openai']['api_key']
                if not api_key:
                    st.error("OpenAI API key not configured. Please add it in Settings.")
                    return
                
                # Initialize OpenAI client with new API
                client = OpenAI(api_key=api_key)
                
                # Prepare data for OpenAI
                df_str = df.to_string()
                
                # Create prompt for OpenAI
                openai_prompt = f"""
                {prompt}
                
                Analysis Type: {analysis_type}
                Date Range: {date_range} ({start_date} to {end_date})
                
                Data:
                {df_str}
                
                Please provide a detailed analysis with insights and recommendations.
                """
                
                # Call OpenAI API with new interface
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a Google Analytics expert who provides insightful analysis."},
                        {"role": "user", "content": openai_prompt}
                    ]
                )
                
                # Extract insights with new API response format
                insights = response.choices[0].message.content
                
                # Save report
                report = {
                    'id': str(uuid.uuid4()),
                    'title': report_title,
                    'description': report_description,
                    'analysis_type': analysis_type,
                    'account': account_name,
                    'date_range': f"{start_date} to {end_date}",
                    'prompt': prompt,
                    'data': data,
                    'insights': insights,
                    'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                st.session_state.reports.append(report)
                save_session_state()
                
                # Display insights
                st.subheader("Analysis Results")
                st.write(insights)
                
                # Display data visualization
                if len(data) > 0:
                    st.subheader("Data Visualization")
                    if 'date' in df.columns:
                        # Convert date string to datetime
                        df['date'] = pd.to_datetime(df['date'])
                        
                        # Create time series chart for each metric
                        for metric in analysis_config['metrics']:
                            if metric in df.columns:
                                try:
                                    df[metric] = pd.to_numeric(df[metric])
                                    fig = px.line(df, x='date', y=metric, title=f"{metric} over time")
                                    st.plotly_chart(fig)
                                except:
                                    st.write(f"Could not create visualization for {metric}")
                    elif len(analysis_config['dimensions']) > 0 and len(analysis_config['metrics']) > 0:
                        # Create bar chart for first dimension and first metric
                        dimension = analysis_config['dimensions'][0]
                        metric = analysis_config['metrics'][0]
                        if dimension in df.columns and metric in df.columns:
                            try:
                                df[metric] = pd.to_numeric(df[metric])
                                fig = px.bar(df, x=dimension, y=metric, title=f"{metric} by {dimension}")
                                st.plotly_chart(fig)
                            except:
                                st.write(f"Could not create visualization for {dimension} and {metric}")
            
            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")
                st.error("Please try a different combination of metrics and dimensions or a different prompt.")

# Report History page
def render_report_history():
    render_sidebar()
    
    st.title("Report History")
    
    if not st.session_state.reports:
        st.info("No reports yet. Create a new analysis to generate reports.")
        return
    
    # Display reports
    for i, report in enumerate(reversed(st.session_state.reports)):
        with st.expander(f"{report['title']} - {report['date']}"):
            st.write(f"**Analysis Type:** {report['analysis_type']}")
            st.write(f"**Description:** {report['description']}")
            st.write(f"**Account:** {report['account']}")
            st.write(f"**Date Range:** {report['date_range']}")
            st.write("**Insights:**")
            st.write(report['insights'])
            
            # Delete report button
            if st.button(f"Delete Report", key=f"delete_{i}"):
                st.session_state.reports.remove(report)
                save_session_state()
                st.rerun()

# Prompt Library page
def render_prompt_library():
    render_sidebar()
    
    st.title("Prompt Library")
    
    # Display existing prompts
    st.header("Existing Prompts")
    for prompt_name, prompt_text in st.session_state.prompts.items():
        with st.expander(prompt_name):
            st.write(prompt_text)
            
            # Delete prompt button (prevent deleting the default prompt)
            if prompt_name != "Default Prompt":
                if st.button(f"Delete", key=f"delete_{prompt_name}"):
                    del st.session_state.prompts[prompt_name]
                    save_session_state()
                    st.rerun()
    
    # Add new prompt
    st.header("Add New Prompt")
    new_prompt_name = st.text_input("Prompt Name")
    new_prompt_text = st.text_area("Prompt Text")
    
    if st.button("Add Prompt"):
        if new_prompt_name and new_prompt_text:
            if new_prompt_name in st.session_state.prompts:
                st.error("A prompt with this name already exists")
            else:
                st.session_state.prompts[new_prompt_name] = new_prompt_text
                save_session_state()
                st.success(f"Prompt '{new_prompt_name}' added successfully")
                st.rerun()
        else:
            st.error("Please enter both prompt name and text")

# Settings page
def render_settings():
    render_sidebar()
    
    st.title("Settings")
    
    # API Keys
    st.header("API Keys")
    openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.api_keys['openai']['api_key'], type="password")
    
    if st.button("Save API Keys"):
        st.session_state.api_keys['openai']['api_key'] = openai_api_key
        save_session_state()
        st.success("API keys saved successfully")
    
    # Google Analytics Accounts
    st.header("Google Analytics")
    
    # Display existing accounts
    if st.session_state.ga_accounts:
        st.subheader("Existing Accounts")
        for account_name, account_data in st.session_state.ga_accounts.items():
            with st.expander(f"{account_name} (Property ID: {account_data['property_id']})"):
                st.write(f"Credentials Path: {account_data['credentials_path']}")
                
                # Delete account button
                if st.button(f"Delete Account", key=f"delete_{account_name}"):
                    # Remove credentials file
                    if os.path.exists(account_data['credentials_path']):
                        os.remove(account_data['credentials_path'])
                    
                    # Remove from session state
                    del st.session_state.ga_accounts[account_name]
                    save_session_state()
                    st.success(f"Account '{account_name}' deleted successfully")
                    st.rerun()
    
    # Add new account
    st.subheader("Add New Account")
    new_account_name = st.text_input("Account Name")
    new_property_id = st.text_input("Property ID (numbers only, without 'properties/')")
    
    # File uploader for service account credentials
    uploaded_file = st.file_uploader("Upload Service Account JSON Credentials", type=["json"])
    
    if st.button("Add Account"):
        if new_account_name and new_property_id and uploaded_file:
            try:
                # Read and validate the uploaded file
                credentials_json = json.load(uploaded_file)
                
                # Check if it's a service account key
                if is_service_account_key(credentials_json):
                    # Save credentials to file
                    credentials_path = f'credentials/ga_{new_account_name.lower().replace(" ", "_")}.json'
                    with open(credentials_path, 'w') as f:
                        json.dump(credentials_json, f)
                    
                    # Add to session state
                    st.session_state.ga_accounts[new_account_name] = {
                        'property_id': new_property_id,
                        'credentials_path': credentials_path
                    }
                    
                    save_session_state()
                    st.success(f"Account '{new_account_name}' added successfully")
                    st.rerun()
                else:
                    # Check if it's a web client credentials file that needs conversion
                    if 'web' in credentials_json:
                        # Convert to installed client format
                        converted_credentials = convert_web_to_installed_client(credentials_json)
                        
                        # Save converted credentials to file
                        credentials_path = f'credentials/ga_{new_account_name.lower().replace(" ", "_")}.json'
                        with open(credentials_path, 'w') as f:
                            json.dump(converted_credentials, f)
                        
                        # Add to session state
                        st.session_state.ga_accounts[new_account_name] = {
                            'property_id': new_property_id,
                            'credentials_path': credentials_path
                        }
                        
                        save_session_state()
                        st.success(f"Account '{new_account_name}' added successfully (converted web client credentials)")
                        st.rerun()
                    else:
                        st.error("Invalid credentials format. Please upload a service account key JSON file.")
            except Exception as e:
                st.error(f"Error adding account: {str(e)}")
        else:
            st.error("Please enter account name, property ID, and upload credentials file")
    
    # Help section for GA4 metrics and dimensions
    st.header("Help with Google Analytics 4 Metrics and Dimensions")
    with st.expander("Common GA4 Metrics"):
        st.markdown("""
        Here are some common GA4 metrics you can use in your analyses:
        """)
        
        # Create a table of metrics and descriptions
        metrics_table = []
        for metric, description in VALID_GA4_METRICS.items():
            metrics_table.append(f"| `{metric}` | {description} |")
        
        st.markdown("| Metric | Description |")
        st.markdown("| --- | --- |")
        for row in metrics_table:
            st.markdown(row)
    
    with st.expander("Common GA4 Dimensions"):
        st.markdown("""
        Here are some common GA4 dimensions you can use in your analyses:
        """)
        
        # Create a table of dimensions and descriptions
        dimensions_table = []
        for dimension, description in VALID_GA4_DIMENSIONS.items():
            dimensions_table.append(f"| `{dimension}` | {description} |")
        
        st.markdown("| Dimension | Description |")
        st.markdown("| --- | --- |")
        for row in dimensions_table:
            st.markdown(row)
    
    with st.expander("Compatibility Information"):
        st.markdown("""
        ### Metric and Dimension Compatibility
        
        Not all metrics and dimensions can be used together in GA4. The application will automatically check for compatibility and use safe alternatives if needed.
        
        Common incompatible combinations:
        - `itemName` and `grossItemRevenue` cannot be used together
        - Some e-commerce metrics require specific e-commerce tracking setup in GA4
        
        If you encounter compatibility errors, try:
        1. Using a different analysis type
        2. Using a custom prompt that focuses on general metrics
        3. Checking your GA4 property setup to ensure all required features are enabled
        """)

# Main app logic
def main():
    if not st.session_state.logged_in:
        render_login()
    else:
        if st.session_state.page == "dashboard":
            render_dashboard()
        elif st.session_state.page == "new_analysis":
            render_new_analysis()
        elif st.session_state.page == "report_history":
            render_report_history()
        elif st.session_state.page == "prompt_library":
            render_prompt_library()
        elif st.session_state.page == "settings":
            render_settings()
        else:
            render_dashboard()

if __name__ == "__main__":
    main()
