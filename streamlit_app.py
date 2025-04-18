"""
Streamlit App for Google Analytics AI Analyzer

This is the main application file for the Google Analytics AI Analyzer.
It provides a web interface for analyzing Google Analytics data using AI.
"""

import os
import sys
import json
import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import base64
import tempfile
import uuid
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our custom modules with error handling
try:
    from ga_integration import GoogleAnalyticsConnector, GADataProcessor
except ImportError as e:
    logger.error(f"Error importing ga_integration: {e}")
    # Create mock classes if imports fail
    class GoogleAnalyticsConnector:
        def __init__(self, *args, **kwargs):
            pass
        def authenticate(self, *args, **kwargs):
            return False
        def get_data(self, *args, **kwargs):
            return pd.DataFrame()
        def import_from_file(self, *args, **kwargs):
            return pd.DataFrame()

    class GADataProcessor:
        def __init__(self, *args, **kwargs):
            pass
        def get_summary_statistics(self, *args, **kwargs):
            return {}
        def prepare_data_for_llm(self, *args, **kwargs):
            return {}

try:
    from llm_integration import LLMFactory, AnalyticsInsightGenerator, PromptTemplate, AnalyticsPromptLibrary
except ImportError as e:
    logger.error(f"Error importing llm_integration: {e}")
    # Create mock classes if imports fail
    class PromptTemplate:
        def __init__(self, *args, **kwargs):
            pass
        def render(self, *args, **kwargs):
            return ""

    class AnalyticsPromptLibrary:
        def get_general_analysis_prompt(self, *args, **kwargs):
            return PromptTemplate("", {})
        def get_traffic_analysis_prompt(self, *args, **kwargs):
            return PromptTemplate("", {})
        def get_conversion_analysis_prompt(self, *args, **kwargs):
            return PromptTemplate("", {})
        def get_user_behavior_analysis_prompt(self, *args, **kwargs):
            return PromptTemplate("", {})
        def get_anomaly_detection_prompt(self, *args, **kwargs):
            return PromptTemplate("", {})
        def get_comparative_analysis_prompt(self, *args, **kwargs):
            return PromptTemplate("", {})

    class LLMFactory:
        @staticmethod
        def create_provider(*args, **kwargs):
            return None

    class AnalyticsInsightGenerator:
        def __init__(self, *args, **kwargs):
            pass
        def generate_insights(self, *args, **kwargs):
            return "Mock insights (LLM integration not available)"
        def generate_general_analysis(self, *args, **kwargs):
            return "Mock general analysis (LLM integration not available)"
        def generate_traffic_analysis(self, *args, **kwargs):
            return "Mock traffic analysis (LLM integration not available)"
        def generate_conversion_analysis(self, *args, **kwargs):
            return "Mock conversion analysis (LLM integration not available)"
        def generate_user_behavior_analysis(self, *args, **kwargs):
            return "Mock user behavior analysis (LLM integration not available)"
        def generate_anomaly_analysis(self, *args, **kwargs):
            return "Mock anomaly analysis (LLM integration not available)"
        def generate_comparative_analysis(self, *args, **kwargs):
            return "Mock comparative analysis (LLM integration not available)"

try:
    from analysis_pipeline import AnalysisPipeline, ReportGenerator, AnalysisWorkflow
except ImportError as e:
    logger.error(f"Error importing analysis_pipeline: {e}")
    # Create mock classes if imports fail
    class AnalysisWorkflow:
        @staticmethod
        def get_standard_metrics(*args, **kwargs):
            return ["sessions", "users", "pageviews"]
        
        @staticmethod
        def get_standard_dimensions(*args, **kwargs):
            return ["date", "deviceCategory"]
        
        @staticmethod
        def get_weekly_report_config(*args, **kwargs):
            return {
                "metrics": ["sessions", "users", "pageviews"],
                "dimensions": ["date", "deviceCategory"],
                "analysis_type": "general"
            }
        
        @staticmethod
        def get_traffic_overview_config(*args, **kwargs):
            return {
                "metrics": ["sessions", "users", "pageviews"],
                "dimensions": ["source", "medium"],
                "analysis_type": "traffic"
            }
        
        @staticmethod
        def get_conversion_overview_config(*args, **kwargs):
            return {
                "metrics": ["conversions", "conversionRate"],
                "dimensions": ["date", "deviceCategory"],
                "analysis_type": "conversion"
            }
        
        @staticmethod
        def get_user_behavior_config(*args, **kwargs):
            return {
                "metrics": ["sessionDuration", "bounceRate"],
                "dimensions": ["date", "deviceCategory"],
                "analysis_type": "user_behavior"
            }
        
        @staticmethod
        def get_comparative_report_config(*args, **kwargs):
            return {
                "metrics": ["sessions", "users", "pageviews"],
                "dimensions": ["date", "deviceCategory"],
                "analysis_type": "comparative"
            }

    class ReportGenerator:
        def __init__(self, *args, **kwargs):
            pass
        
        def generate_markdown_report(self, *args, **kwargs):
            return "# Mock Report\n\nThis is a mock report generated when the analysis pipeline is not available."
        
        def generate_html_report(self, *args, **kwargs):
            return "<h1>Mock Report</h1><p>This is a mock report generated when the analysis pipeline is not available.</p>"
        
        def save_report(self, *args, **kwargs):
            return "/tmp/mock_report.md"

    class AnalysisPipeline:
        def __init__(self, *args, **kwargs):
            pass
        
        def authenticate_ga(self, *args, **kwargs):
            return False
        
        def import_ga_data(self, *args, **kwargs):
            return pd.DataFrame()
        
        def import_ga_data_from_file(self, *args, **kwargs):
            return pd.DataFrame()
        
        def analyze_data(self, *args, **kwargs):
            return {}
        
        def generate_insights(self, *args, **kwargs):
            return "Mock insights (Analysis pipeline not available)"
        
        def run_complete_analysis(self, *args, **kwargs):
            return {
                "raw_data": pd.DataFrame(),
                "processed_data": {},
                "insights": "Mock insights (Analysis pipeline not available)",
                "metadata": {
                    "timestamp": datetime.datetime.now().isoformat()
                }
            }

# Set page configuration
st.set_page_config(
    page_title="GA Analytics AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
REPORTS_DIR = BASE_DIR / "reports"
PROMPTS_DIR = BASE_DIR / "prompts"
CONFIG_DIR = BASE_DIR / "config"
CREDENTIALS_DIR = BASE_DIR / "credentials"  # Directory for storing credentials

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR, REPORTS_DIR, PROMPTS_DIR, CONFIG_DIR, CREDENTIALS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Define database file paths
USERS_DB = DATA_DIR / "users.json"
REPORTS_DB = DATA_DIR / "reports.json"
PROMPTS_DB = DATA_DIR / "prompts.json"
GA_ACCOUNTS_DB = DATA_DIR / "ga_accounts.json"
API_KEYS_DB = DATA_DIR / "api_keys.json"

# Initialize database files if they don't exist
def init_db_files():
    """Initialize database files if they don't exist."""
    if not USERS_DB.exists():
        with open(USERS_DB, 'w') as f:
            json.dump([], f)
    
    if not REPORTS_DB.exists():
        with open(REPORTS_DB, 'w') as f:
            json.dump([], f)
    
    if not PROMPTS_DB.exists():
        with open(PROMPTS_DB, 'w') as f:
            json.dump([], f)
    
    if not GA_ACCOUNTS_DB.exists():
        with open(GA_ACCOUNTS_DB, 'w') as f:
            json.dump([], f)
    
    if not API_KEYS_DB.exists():
        with open(API_KEYS_DB, 'w') as f:
            json.dump([], f)

# Call initialization
init_db_files()

# Database functions
def load_json_db(file_path: Path) -> List[Dict[str, Any]]:
    """Load a JSON database file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_db(file_path: Path, data: List[Dict[str, Any]]):
    """Save data to a JSON database file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get a user by email."""
    users = load_json_db(USERS_DB)
    for user in users:
        if user['email'] == email:
            return user
    return None

def add_user(email: str, name: str) -> Dict[str, Any]:
    """Add a new user."""
    users = load_json_db(USERS_DB)
    
    # Check if user already exists
    for user in users:
        if user['email'] == email:
            return user
    
    # Create new user
    new_user = {
        'id': str(uuid.uuid4()),
        'email': email,
        'name': name,
        'created_at': datetime.datetime.now().isoformat()
    }
    
    users.append(new_user)
    save_json_db(USERS_DB, users)
    
    return new_user

def get_ga_accounts() -> List[Dict[str, Any]]:
    """Get all GA accounts."""
    return load_json_db(GA_ACCOUNTS_DB)

def get_ga_account(account_id: str) -> Optional[Dict[str, Any]]:
    """Get a GA account by ID."""
    accounts = load_json_db(GA_ACCOUNTS_DB)
    for account in accounts:
        if account['id'] == account_id:
            return account
    return None

def is_service_account_json(json_content: bytes) -> bool:
    """
    Check if the JSON content is a service account key.
    
    Args:
        json_content: JSON content as bytes
        
    Returns:
        True if it's a service account key, False otherwise
    """
    try:
        json_data = json.loads(json_content)
        # Service account keys typically have these fields
        required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
        return all(field in json_data for field in required_fields) and json_data.get('type') == 'service_account'
    except Exception:
        return False

def add_ga_account(name: str, property_id: str, credentials_file) -> Dict[str, Any]:
    """Add a new GA account."""
    accounts = load_json_db(GA_ACCOUNTS_DB)
    
    # Generate a unique ID for the account
    account_id = str(uuid.uuid4())
    
    # Save the credentials file to the credentials directory
    credentials_filename = f"ga_credentials_{account_id}.json"
    credentials_path = CREDENTIALS_DIR / credentials_filename
    
    # Read the uploaded file and save it
    credentials_content = credentials_file.read()
    
    # Check if it's a service account JSON
    is_service_account = is_service_account_json(credentials_content)
    
    # Save the credentials file
    with open(credentials_path, 'wb') as f:
        f.write(credentials_content)
    
    # Create new account
    new_account = {
        'id': account_id,
        'name': name,
        'property_id': property_id,
        'credentials_path': str(credentials_path),
        'is_service_account': is_service_account,
        'created_at': datetime.datetime.now().isoformat()
    }
    
    accounts.append(new_account)
    save_json_db(GA_ACCOUNTS_DB, accounts)
    
    return new_account

def delete_ga_account(account_id: str) -> bool:
    """Delete a GA account by ID."""
    accounts = load_json_db(GA_ACCOUNTS_DB)
    
    # Find the account to delete
    for i, account in enumerate(accounts):
        if account['id'] == account_id:
            # Delete the credentials file
            credentials_path = Path(account['credentials_path'])
            if credentials_path.exists():
                credentials_path.unlink()
            
            # Remove the account from the list
            accounts.pop(i)
            save_json_db(GA_ACCOUNTS_DB, accounts)
            return True
    
    return False

def get_api_keys() -> Dict[str, Dict[str, str]]:
    """Get all API keys."""
    api_keys = load_json_db(API_KEYS_DB)
    
    # Convert to dictionary format
    result = {}
    for key in api_keys:
        provider = key.get('provider', '')
        if provider:
            result[provider] = {
                'api_key': key.get('api_key', ''),
                'updated_at': key.get('updated_at', '')
            }
    
    return result

def update_api_key(provider: str, api_key: str) -> Dict[str, Any]:
    """Update an API key."""
    api_keys = load_json_db(API_KEYS_DB)
    
    # Check if key already exists
    for key in api_keys:
        if key.get('provider') == provider:
            key['api_key'] = api_key
            key['updated_at'] = datetime.datetime.now().isoformat()
            save_json_db(API_KEYS_DB, api_keys)
            return key
    
    # Create new key
    new_key = {
        'provider': provider,
        'api_key': api_key,
        'created_at': datetime.datetime.now().isoformat(),
        'updated_at': datetime.datetime.now().isoformat()
    }
    
    api_keys.append(new_key)
    save_json_db(API_KEYS_DB, api_keys)
    
    return new_key

def get_prompts() -> List[Dict[str, Any]]:
    """Get all prompt templates."""
    return load_json_db(PROMPTS_DB)

def get_prompt(prompt_id: str) -> Optional[Dict[str, Any]]:
    """Get a prompt template by ID."""
    prompts = load_json_db(PROMPTS_DB)
    for prompt in prompts:
        if prompt['id'] == prompt_id:
            return prompt
    return None

def add_prompt(name: str, description: str, prompt_template: str) -> Dict[str, Any]:
    """Add a new prompt template."""
    prompts = load_json_db(PROMPTS_DB)
    
    # Create new prompt
    new_prompt = {
        'id': str(uuid.uuid4()),
        'name': name,
        'description': description,
        'prompt_template': prompt_template,
        'created_at': datetime.datetime.now().isoformat(),
        'updated_at': datetime.datetime.now().isoformat()
    }
    
    prompts.append(new_prompt)
    save_json_db(PROMPTS_DB, prompts)
    
    return new_prompt

def update_prompt(prompt_id: str, name: str, description: str, prompt_template: str) -> Optional[Dict[str, Any]]:
    """Update a prompt template."""
    prompts = load_json_db(PROMPTS_DB)
    
    for prompt in prompts:
        if prompt['id'] == prompt_id:
            prompt['name'] = name
            prompt['description'] = description
            prompt['prompt_template'] = prompt_template
            prompt['updated_at'] = datetime.datetime.now().isoformat()
            
            save_json_db(PROMPTS_DB, prompts)
            return prompt
    
    return None

def delete_prompt(prompt_id: str) -> bool:
    """Delete a prompt template."""
    prompts = load_json_db(PROMPTS_DB)
    
    for i, prompt in enumerate(prompts):
        if prompt['id'] == prompt_id:
            prompts.pop(i)
            save_json_db(PROMPTS_DB, prompts)
            return True
    
    return False

def get_reports() -> List[Dict[str, Any]]:
    """Get all reports."""
    return load_json_db(REPORTS_DB)

def get_report(report_id: str) -> Optional[Dict[str, Any]]:
    """Get a report by ID."""
    reports = load_json_db(REPORTS_DB)
    for report in reports:
        if report['id'] == report_id:
            return report
    return None

def add_report(title: str, description: str, report_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Add a new report."""
    reports = load_json_db(REPORTS_DB)
    
    # Create new report
    new_report = {
        'id': str(uuid.uuid4()),
        'title': title,
        'description': description,
        'user_id': user_id,
        'created_at': datetime.datetime.now().isoformat(),
        'updated_at': datetime.datetime.now().isoformat(),
        'data': report_data
    }
    
    reports.append(new_report)
    save_json_db(REPORTS_DB, reports)
    
    return new_report

def delete_report(report_id: str) -> bool:
    """Delete a report."""
    reports = load_json_db(REPORTS_DB)
    
    for i, report in enumerate(reports):
        if report['id'] == report_id:
            reports.pop(i)
            save_json_db(REPORTS_DB, reports)
            return True
    
    return False

# Navigation functions
def navigate_to(page: str):
    """Navigate to a different page in the app."""
    st.session_state.page = page
    # No st.rerun() call to prevent infinite loop

def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.title("Navigation")
        
        if st.button("Dashboard", key="nav_dashboard"):
            navigate_to('dashboard')
            st.rerun()
        
        if st.button("New Analysis", key="nav_new_analysis"):
            navigate_to('new_analysis')
            st.rerun()
        
        if st.button("Report History", key="nav_report_history"):
            navigate_to('report_history')
            st.rerun()
        
        if st.button("Prompt Library", key="nav_prompt_library"):
            navigate_to('prompt_library')
            st.rerun()
        
        if st.button("Settings", key="nav_settings"):
            navigate_to('settings')
            st.rerun()
        
        st.markdown("---")
        
        if st.button("Logout", key="nav_logout"):
            st.session_state.user = None
            navigate_to('login')
            st.rerun()

# Page rendering functions
def render_login():
    """Render the login page."""
    st.title("Login")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        name = st.text_input("Name")
        
        submitted = st.form_submit_button("Login")
        
        if submitted and email and name:
            user = add_user(email, name)
            st.session_state.user = user
            navigate_to('dashboard')
            st.rerun()

def render_dashboard():
    """Render the dashboard page."""
    st.title(f"Welcome, {st.session_state.user['name']}!")
    
    # Render sidebar navigation
    render_sidebar()
    
    # Check if any GA accounts are configured
    accounts = get_ga_accounts()
    if not accounts:
        st.warning("No Google Analytics accounts configured. Please add an account in the Settings page.")
        return
    
    # Check if OpenAI API key is configured
    api_keys = get_api_keys()
    if 'openai' not in api_keys or not api_keys['openai'].get('api_key'):
        st.warning("OpenAI API key not configured. Please add your API key in the Settings page.")
    
    # Display recent reports
    reports = get_reports()
    if reports:
        st.subheader("Recent Reports")
        
        # Sort reports by creation date (newest first)
        reports.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Display the 5 most recent reports
        for report in reports[:5]:
            with st.expander(f"{report['title']} ({report['created_at'][:10]})"):
                st.write(report['description'])
                
                if 'data' in report and 'insights' in report['data']:
                    st.markdown(report['data']['insights'])
                
                st.button("View Full Report", key=f"view_report_{report['id']}")
    else:
        st.info("No reports yet. Create your first analysis!")
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("New Analysis", key="dashboard_new_analysis"):
            navigate_to('new_analysis')
            st.rerun()
    
    with col2:
        if st.button("View Reports", key="dashboard_view_reports"):
            navigate_to('report_history')
            st.rerun()
    
    with col3:
        if st.button("Settings", key="dashboard_settings"):
            navigate_to('settings')
            st.rerun()

def render_new_analysis():
    """Render the new analysis page."""
    st.title("New Analysis")
    
    # Render sidebar navigation
    render_sidebar()
    
    # Check if any GA accounts are configured
    accounts = get_ga_accounts()
    if not accounts:
        st.warning("No Google Analytics accounts configured. Please add an account in the Settings page.")
        return
    
    # Check if OpenAI API key is configured
    api_keys = get_api_keys()
    if 'openai' not in api_keys or not api_keys['openai'].get('api_key'):
        st.warning("OpenAI API key not configured. Please add your API key in the Settings page.")
        return
    
    # Analysis form
    st.subheader("Select Analysis Type")
    analysis_type = st.selectbox(
        "Select Analysis Type",
        options=["General Overview", "Traffic Analysis", "Conversion Analysis", "User Behavior", "Anomaly Detection", "Comparative Analysis"],
        index=0
    )
    
    st.subheader("Analysis Prompt")
    prompts = get_prompts()
    prompt_options = ["Default Prompt"] + [p['name'] for p in prompts]
    selected_prompt_name = st.selectbox("Select Prompt", options=prompt_options)
    
    selected_prompt = None
    if selected_prompt_name != "Default Prompt":
        for p in prompts:
            if p['name'] == selected_prompt_name:
                selected_prompt = p
                break
    
    st.subheader("Report Details")
    report_title = st.text_input("Report Title", value=f"{analysis_type} - {datetime.datetime.now().strftime('%Y-%m-%d')}")
    report_description = st.text_area("Report Description", value=f"Analysis of general data from Google Analytics.")
    
    # Account selection
    st.subheader("Google Analytics Account")
    account_options = [f"{a['name']} (Property: {a['property_id']})" for a in accounts]
    selected_account_name = st.selectbox("Select Account", options=account_options)
    
    # Find the selected account
    selected_account = None
    for i, name in enumerate(account_options):
        if name == selected_account_name:
            selected_account = accounts[i]
            break
    
    # Date range selection
    st.subheader("Date Range")
    date_range_options = ["Last 7 days", "Last 30 days", "Last 90 days", "Custom Range"]
    selected_date_range = st.selectbox("Select Date Range", options=date_range_options)
    
    start_date = None
    end_date = None
    
    if selected_date_range == "Custom Range":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.datetime.now() - datetime.timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", value=datetime.datetime.now())
    else:
        # Calculate date range based on selection
        end_date = datetime.datetime.now().date()
        if selected_date_range == "Last 7 days":
            start_date = end_date - datetime.timedelta(days=7)
        elif selected_date_range == "Last 30 days":
            start_date = end_date - datetime.timedelta(days=30)
        elif selected_date_range == "Last 90 days":
            start_date = end_date - datetime.timedelta(days=90)
    
    # Format dates for GA API
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # Run analysis button
    if st.button("Run Analysis"):
        if selected_account:
            with st.spinner("Running analysis... This may take a few moments."):
                try:
                    # Prepare analysis configuration
                    analysis_config = {}
                    
                    # Set date range
                    analysis_config['date_range'] = {
                        'start_date': start_date_str,
                        'end_date': end_date_str
                    }
                    
                    # Set metrics and dimensions based on analysis type
                    if analysis_type == "General Overview":
                        analysis_config['metrics'] = ["sessions", "totalUsers", "screenPageViews", "bounceRate", "averageSessionDuration"]
                        analysis_config['dimensions'] = ["date", "deviceCategory"]
                        analysis_config['analysis_type'] = "general"
                    elif analysis_type == "Traffic Analysis":
                        analysis_config['metrics'] = ["sessions", "totalUsers", "newUsers", "screenPageViews", "averageSessionDuration"]
                        analysis_config['dimensions'] = ["sessionSource", "sessionMedium", "date"]
                        analysis_config['analysis_type'] = "traffic"
                    elif analysis_type == "Conversion Analysis":
                        analysis_config['metrics'] = ["conversions", "conversionRate", "eventCount"]
                        analysis_config['dimensions'] = ["date", "deviceCategory", "sessionSource"]
                        analysis_config['analysis_type'] = "conversion"
                    elif analysis_type == "User Behavior":
                        analysis_config['metrics'] = ["screenPageViews", "averageSessionDuration", "bounceRate", "eventCount"]
                        analysis_config['dimensions'] = ["pagePath", "deviceCategory"]
                        analysis_config['analysis_type'] = "user_behavior"
                    elif analysis_type == "Anomaly Detection":
                        analysis_config['metrics'] = ["sessions", "totalUsers", "screenPageViews", "bounceRate"]
                        analysis_config['dimensions'] = ["date"]
                        analysis_config['analysis_type'] = "anomaly"
                    elif analysis_type == "Comparative Analysis":
                        analysis_config['metrics'] = ["sessions", "totalUsers", "screenPageViews", "bounceRate"]
                        analysis_config['dimensions'] = ["date", "deviceCategory"]
                        analysis_config['analysis_type'] = "comparative"
                    
                    # Initialize the analysis pipeline
                    pipeline = AnalysisPipeline(
                        ga_credentials_path=selected_account['credentials_path'],
                        llm_provider="openai",
                        llm_api_key=api_keys['openai']['api_key'],
                        use_service_account=selected_account.get('is_service_account', False)
                    )
                    
                    # Authenticate with Google Analytics
                    auth_success = pipeline.authenticate_ga()
                    if not auth_success:
                        st.error("Failed to authenticate with Google Analytics. Please check your credentials.")
                        return
                    
                    # Run the analysis
                    results = pipeline.run_complete_analysis(
                        property_id=selected_account['property_id'],
                        metrics=analysis_config['metrics'],
                        dimensions=analysis_config['dimensions'],
                        date_range=analysis_config['date_range'],
                        analysis_type=analysis_config['analysis_type'],
                        custom_prompt=selected_prompt['prompt_template'] if selected_prompt else None
                    )
                    
                    # Save the report
                    report = add_report(
                        title=report_title,
                        description=report_description,
                        report_data=results,
                        user_id=st.session_state.user['id']
                    )
                    
                    # Display the results
                    st.success("Analysis completed successfully!")
                    st.markdown(results['insights'])
                    
                    # Show raw data if available
                    if 'raw_data' in results and not results['raw_data'].empty:
                        with st.expander("View Raw Data"):
                            st.dataframe(results['raw_data'])
                    
                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")
        else:
            st.error("Please select a Google Analytics account.")

def render_report_history():
    """Render the report history page."""
    st.title("Report History")
    
    # Render sidebar navigation
    render_sidebar()
    
    # Get all reports
    reports = get_reports()
    
    if not reports:
        st.info("No reports found. Create your first analysis!")
        return
    
    # Sort reports by creation date (newest first)
    reports.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    # Display reports
    for report in reports:
        with st.expander(f"{report['title']} ({report['created_at'][:10]})"):
            st.write(report['description'])
            
            if 'data' in report and 'insights' in report['data']:
                st.markdown(report['data']['insights'])
            
            # Actions
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("View Full Report", key=f"view_full_report_{report['id']}"):
                    # TODO: Implement full report view
                    st.info("Full report view not implemented yet.")
            
            with col2:
                if st.button("Delete Report", key=f"delete_report_{report['id']}"):
                    if delete_report(report['id']):
                        st.success("Report deleted successfully.")
                        st.rerun()
                    else:
                        st.error("Failed to delete report.")

def render_prompt_library():
    """Render the prompt library page."""
    st.title("Prompt Library")
    
    # Render sidebar navigation
    render_sidebar()
    
    # Get all prompts
    prompts = get_prompts()
    
    # Add new prompt form
    with st.expander("Add New Prompt"):
        with st.form("add_prompt_form"):
            prompt_name = st.text_input("Prompt Name")
            prompt_description = st.text_area("Description")
            prompt_template = st.text_area("Prompt Template")
            
            submitted = st.form_submit_button("Add Prompt")
            
            if submitted and prompt_name and prompt_template:
                new_prompt = add_prompt(prompt_name, prompt_description, prompt_template)
                if new_prompt:
                    st.success(f"Prompt '{prompt_name}' added successfully.")
                    st.rerun()
                else:
                    st.error("Failed to add prompt.")
    
    # Display existing prompts
    if prompts:
        st.subheader("Existing Prompts")
        
        for prompt in prompts:
            with st.expander(f"{prompt['name']}"):
                st.write(prompt['description'])
                st.text_area(f"Template for {prompt['name']}", value=prompt['prompt_template'], height=200, key=f"template_{prompt['id']}")
                
                # Actions
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Edit", key=f"edit_prompt_{prompt['id']}"):
                        st.session_state.editing_prompt = prompt
                
                with col2:
                    if st.button("Delete", key=f"delete_prompt_{prompt['id']}"):
                        if delete_prompt(prompt['id']):
                            st.success(f"Prompt '{prompt['name']}' deleted successfully.")
                            st.rerun()
                        else:
                            st.error("Failed to delete prompt.")
        
        # Edit prompt form
        if hasattr(st.session_state, 'editing_prompt') and st.session_state.editing_prompt:
            prompt = st.session_state.editing_prompt
            
            st.subheader(f"Edit Prompt: {prompt['name']}")
            
            with st.form("edit_prompt_form"):
                edited_name = st.text_input("Prompt Name", value=prompt['name'])
                edited_description = st.text_area("Description", value=prompt['description'])
                edited_template = st.text_area("Prompt Template", value=prompt['prompt_template'], height=300)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    submitted = st.form_submit_button("Save Changes")
                
                with col2:
                    cancelled = st.form_submit_button("Cancel")
                
                if submitted and edited_name and edited_template:
                    updated_prompt = update_prompt(prompt['id'], edited_name, edited_description, edited_template)
                    if updated_prompt:
                        st.success(f"Prompt '{edited_name}' updated successfully.")
                        st.session_state.editing_prompt = None
                        st.rerun()
                    else:
                        st.error("Failed to update prompt.")
                
                if cancelled:
                    st.session_state.editing_prompt = None
                    st.rerun()
    else:
        st.info("No prompts found. Add your first prompt using the form above.")

def render_settings():
    """Render the settings page."""
    st.title("Settings")
    
    # Render sidebar navigation
    render_sidebar()
    
    # API Keys section
    st.header("API Keys")
    
    api_keys = get_api_keys()
    
    with st.form("api_keys_form"):
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=api_keys.get('openai', {}).get('api_key', ''),
            type="password"
        )
        
        submitted = st.form_submit_button("Save API Keys")
        
        if submitted:
            if openai_api_key:
                update_api_key('openai', openai_api_key)
                st.success("API keys saved successfully.")
            else:
                st.error("OpenAI API key is required.")
    
    # Google Analytics Accounts section
    st.header("Google Analytics")
    
    # Add new account form
    with st.expander("Add New Account"):
        with st.form("add_ga_account_form"):
            account_name = st.text_input("Account Name")
            property_id = st.text_input("Property ID", help="Format: 123456789")
            credentials_file = st.file_uploader(
                "Service Account JSON Key", 
                type=["json"], 
                help="Upload your Google Analytics service account key JSON file."
            )
            
            submitted = st.form_submit_button("Add Account")
            
            if submitted:
                if account_name and property_id and credentials_file:
                    try:
                        new_account = add_ga_account(account_name, property_id, credentials_file)
                        if new_account:
                            st.success(f"Account '{account_name}' added successfully.")
                            st.rerun()
                        else:
                            st.error("Failed to add account.")
                    except Exception as e:
                        st.error(f"Error adding account: {str(e)}")
                else:
                    st.error("All fields are required.")
    
    # Display existing accounts
    accounts = get_ga_accounts()
    
    if accounts:
        st.subheader("Existing Accounts")
        
        for account in accounts:
            with st.expander(f"{account['name']} (Property: {account['property_id']})"):
                st.write(f"ID: {account['id']}")
                st.write(f"Created: {account['created_at'][:10]}")
                st.write(f"Credentials Path: {account['credentials_path']}")
                st.write(f"Authentication Type: {'Service Account' if account.get('is_service_account', False) else 'OAuth'}")
                
                if st.button("Delete Account", key=f"delete_account_{account['id']}"):
                    if delete_ga_account(account['id']):
                        st.success(f"Account '{account['name']}' deleted successfully.")
                        st.rerun()
                    else:
                        st.error("Failed to delete account.")
    else:
        st.info("No Google Analytics accounts configured. Add your first account using the form above.")
    
    # Service Account Help
    with st.expander("Help with Service Account Setup"):
        st.markdown("""
        ## Service Account Setup Help
        
        This application requires a service account to access your Google Analytics data. Service accounts are ideal for server-to-server applications like this one, as they don't require user interaction.
        
        ### Steps to create a service account:
        
        1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select an existing one
        3. Enable the Google Analytics Data API
        4. Go to "IAM & Admin" > "Service Accounts"
        5. Click "Create Service Account"
        6. Enter a name and description for the service account
        7. Click "Create and Continue"
        8. For the role, select "Viewer" (or a more specific role if needed)
        9. Click "Continue" and then "Done"
        10. Click on the service account you just created
        11. Go to the "Keys" tab
        12. Click "Add Key" > "Create new key"
        13. Select "JSON" as the key type and click "Create"
        14. The key file will be downloaded to your computer
        15. Upload this key file in the form above
        
        ### Grant access to your Google Analytics property:
        
        1. Go to your Google Analytics account
        2. Navigate to "Admin" > "Account Access Management"
        3. Click "+"
        4. Add the service account email (it looks like: service-account-name@project-id.iam.gserviceaccount.com)
        5. Assign "Viewer" role (or higher if needed)
        
        ### Common issues:
        
        - Make sure you've enabled the Google Analytics Data API in your Google Cloud project
        - Ensure the service account has the correct permissions in Google Analytics
        - Verify that the property ID is correct (it should be just the number, without "properties/")
        """)
    
    # GA4 Metrics and Dimensions Help
    with st.expander("Help with Google Analytics 4 Metrics and Dimensions"):
        st.markdown("""
        ## Google Analytics 4 Metrics and Dimensions
        
        Google Analytics 4 (GA4) uses different metrics and dimensions than Universal Analytics. Here are some common GA4 metrics and dimensions that you can use in this application:
        
        ### Common GA4 Metrics:
        
        - **sessions**: Number of sessions
        - **totalUsers**: Total number of users
        - **newUsers**: Number of new users
        - **screenPageViews**: Number of page views
        - **averageSessionDuration**: Average session duration
        - **bounceRate**: Bounce rate
        - **eventCount**: Number of events
        - **conversions**: Number of conversions
        - **conversionRate**: Conversion rate
        - **engagementRate**: Engagement rate
        - **activeUsers**: Number of active users
        - **userEngagementDuration**: User engagement duration
        
        ### Common GA4 Dimensions:
        
        - **date**: Date
        - **deviceCategory**: Device category (mobile, tablet, desktop)
        - **sessionSource**: Source of the session
        - **sessionMedium**: Medium of the session
        - **pagePath**: Page path
        - **country**: Country
        - **city**: City
        - **browser**: Browser
        - **operatingSystem**: Operating system
        - **language**: Language
        - **screenResolution**: Screen resolution
        
        For a complete list of GA4 metrics and dimensions, see the [Google Analytics Data API documentation](https://developers.google.com/analytics/devguides/reporting/data/v1/api-schema).
        """)

# Main app logic
def main():
    """Main application entry point."""
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Check if user is logged in
    if st.session_state.user is None and st.session_state.page != 'login':
        st.session_state.page = 'login'
    
    # Render the appropriate page
    if st.session_state.page == 'login':
        render_login()
    elif st.session_state.page == 'dashboard':
        render_dashboard()
    elif st.session_state.page == 'new_analysis':
        render_new_analysis()
    elif st.session_state.page == 'report_history':
        render_report_history()
    elif st.session_state.page == 'prompt_library':
        render_prompt_library()
    elif st.session_state.page == 'settings':
        render_settings()
    else:
        st.error(f"Unknown page: {st.session_state.page}")
        st.session_state.page = 'login'
        st.rerun()

if __name__ == "__main__":
    main()
