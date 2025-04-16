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
            self.ga_connector = GoogleAnalyticsConnector()
            self.insight_generator = AnalyticsInsightGenerator()
        
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
CREDENTIALS_DIR = BASE_DIR / "credentials"  # New directory for storing OAuth credentials

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
        'created_at': datetime.datetime.now().isoformat(),
        'last_login': datetime.datetime.now().isoformat()
    }
    
    users.append(new_user)
    save_json_db(USERS_DB, users)
    
    return new_user

def update_user_login(user_id: str):
    """Update user's last login time."""
    users = load_json_db(USERS_DB)
    
    for user in users:
        if user['id'] == user_id:
            user['last_login'] = datetime.datetime.now().isoformat()
            break
    
    save_json_db(USERS_DB, users)

def get_ga_accounts(user_id: str) -> List[Dict[str, Any]]:
    """Get Google Analytics accounts for a user."""
    accounts = load_json_db(GA_ACCOUNTS_DB)
    return [account for account in accounts if account['user_id'] == user_id]

def add_ga_account(user_id: str, account_name: str, property_id: str, credentials_path: str) -> Dict[str, Any]:
    """Add a Google Analytics account for a user."""
    accounts = load_json_db(GA_ACCOUNTS_DB)
    
    # Check if account already exists
    for account in accounts:
        if account['user_id'] == user_id and account['property_id'] == property_id:
            return account
    
    # Create new account
    new_account = {
        'id': str(uuid.uuid4()),
        'user_id': user_id,
        'account_name': account_name,
        'property_id': property_id,
        'credentials_path': credentials_path,
        'created_at': datetime.datetime.now().isoformat(),
        'last_used': datetime.datetime.now().isoformat(),
        'is_active': True
    }
    
    accounts.append(new_account)
    save_json_db(GA_ACCOUNTS_DB, accounts)
    
    return new_account

def delete_ga_account(account_id: str) -> bool:
    """Delete a Google Analytics account."""
    accounts = load_json_db(GA_ACCOUNTS_DB)
    
    # Find the account to delete
    for i, account in enumerate(accounts):
        if account['id'] == account_id:
            # Delete the credentials file if it exists
            if os.path.exists(account['credentials_path']):
                try:
                    os.remove(account['credentials_path'])
                except Exception as e:
                    logger.error(f"Error deleting credentials file: {e}")
            
            # Remove the account from the list
            del accounts[i]
            save_json_db(GA_ACCOUNTS_DB, accounts)
            return True
    
    return False

def get_api_keys(user_id: str) -> Dict[str, Dict[str, Any]]:
    """Get API keys for a user."""
    keys = load_json_db(API_KEYS_DB)
    user_keys = {}
    
    for key in keys:
        if key['user_id'] == user_id:
            user_keys[key['service']] = key
    
    return user_keys

def add_api_key(user_id: str, service: str, api_key: str) -> Dict[str, Any]:
    """Add an API key for a user."""
    keys = load_json_db(API_KEYS_DB)
    
    # Check if key already exists
    for key in keys:
        if key['user_id'] == user_id and key['service'] == service:
            # Update existing key
            key['api_key'] = api_key  # In a real app, this would be encrypted
            key['updated_at'] = datetime.datetime.now().isoformat()
            save_json_db(API_KEYS_DB, keys)
            return key
    
    # Create new key
    new_key = {
        'id': str(uuid.uuid4()),
        'user_id': user_id,
        'service': service,
        'api_key': api_key,  # In a real app, this would be encrypted
        'created_at': datetime.datetime.now().isoformat(),
        'updated_at': datetime.datetime.now().isoformat(),
        'is_active': True
    }
    
    keys.append(new_key)
    save_json_db(API_KEYS_DB, keys)
    
    return new_key

def get_reports(user_id: str) -> List[Dict[str, Any]]:
    """Get reports for a user."""
    reports = load_json_db(REPORTS_DB)
    return [report for report in reports if report['user_id'] == user_id]

def add_report(user_id: str, title: str, description: str, analysis_type: str, 
               report_content: str, metadata: Dict[str, Any], 
               file_path: Optional[str] = None) -> Dict[str, Any]:
    """Add a report for a user."""
    reports = load_json_db(REPORTS_DB)
    
    # Create new report
    new_report = {
        'id': str(uuid.uuid4()),
        'user_id': user_id,
        'title': title,
        'description': description,
        'analysis_type': analysis_type,
        'created_at': datetime.datetime.now().isoformat(),
        'file_path': file_path,
        'metadata': metadata,
        'is_favorite': False,
        'tags': []
    }
    
    reports.append(new_report)
    save_json_db(REPORTS_DB, reports)
    
    return new_report

def update_report(report_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a report."""
    reports = load_json_db(REPORTS_DB)
    
    for report in reports:
        if report['id'] == report_id:
            for key, value in updates.items():
                report[key] = value
            save_json_db(REPORTS_DB, reports)
            return report
    
    return None

def get_prompts(user_id: str) -> List[Dict[str, Any]]:
    """Get prompts for a user."""
    prompts = load_json_db(PROMPTS_DB)
    return [prompt for prompt in prompts if prompt['user_id'] == user_id or prompt.get('is_system', False)]

def add_prompt(user_id: str, title: str, description: str, prompt_template: str, 
               parameters: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Add a prompt for a user."""
    prompts = load_json_db(PROMPTS_DB)
    
    # Create new prompt
    new_prompt = {
        'id': str(uuid.uuid4()),
        'user_id': user_id,
        'title': title,
        'description': description,
        'prompt_template': prompt_template,
        'parameters': parameters,
        'category': category,
        'created_at': datetime.datetime.now().isoformat(),
        'updated_at': datetime.datetime.now().isoformat(),
        'usage_count': 0,
        'is_favorite': False,
        'is_system': False,
        'tags': []
    }
    
    prompts.append(new_prompt)
    save_json_db(PROMPTS_DB, prompts)
    
    return new_prompt

def update_prompt(prompt_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a prompt."""
    prompts = load_json_db(PROMPTS_DB)
    
    for prompt in prompts:
        if prompt['id'] == prompt_id:
            for key, value in updates.items():
                prompt[key] = value
            prompt['updated_at'] = datetime.datetime.now().isoformat()
            save_json_db(PROMPTS_DB, prompts)
            return prompt
    
    return None

def increment_prompt_usage(prompt_id: str) -> Optional[Dict[str, Any]]:
    """Increment the usage count for a prompt."""
    prompts = load_json_db(PROMPTS_DB)
    
    for prompt in prompts:
        if prompt['id'] == prompt_id:
            prompt['usage_count'] = prompt.get('usage_count', 0) + 1
            save_json_db(PROMPTS_DB, prompts)
            return prompt
    
    return None

# Helper functions
def get_file_download_link(file_path: str, link_text: str) -> str:
    """Generate a download link for a file."""
    with open(file_path, 'r') as f:
        data = f.read()
    
    b64 = base64.b64encode(data.encode()).decode()
    filename = os.path.basename(file_path)
    
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'

def save_uploaded_file(uploaded_file, directory: Path, filename: Optional[str] = None) -> str:
    """Save an uploaded file to the specified directory."""
    if not filename:
        filename = uploaded_file.name
    
    file_path = directory / filename
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)

def navigate_to(page: str):
    """Navigate to a different page in the app."""
    st.session_state.page = page
    # Removed st.rerun() call to prevent infinite loop

# Main app
def main():
    """Main application function."""
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Render the appropriate page
    if st.session_state.page == 'login':
        render_login()
    elif not st.session_state.user:
        # If not logged in, redirect to login
        st.session_state.page = 'login'
        render_login()
    elif st.session_state.page == 'dashboard':
        render_dashboard()
    elif st.session_state.page == 'new_analysis':
        render_new_analysis()
    elif st.session_state.page == 'report_history':
        render_report_history()
    elif st.session_state.page == 'view_report':
        render_view_report()
    elif st.session_state.page == 'prompt_library':
        render_prompt_library()
    elif st.session_state.page == 'settings':
        render_settings()
    else:
        # Default to dashboard
        st.session_state.page = 'dashboard'
        render_dashboard()

def render_login():
    """Render the login page."""
    st.title("GA Analytics AI")
    st.subheader("Login")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        name = st.text_input("Name")
        
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if email and name:
                # Add or get user
                user = add_user(email, name)
                
                # Update last login
                update_user_login(user['id'])
                
                # Set user in session state
                st.session_state.user = user
                
                # Navigate to dashboard
                navigate_to('dashboard')
                # Removed st.rerun() call to prevent infinite loop
            else:
                st.error("Please enter both email and name")

def render_dashboard():
    """Render the dashboard page."""
    st.title(f"Welcome, {st.session_state.user['name']}!")
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        
        if st.button("Dashboard", key="nav_dashboard"):
            navigate_to('dashboard')
            # Removed st.rerun() call to prevent infinite loop
            return
        
        if st.button("New Analysis", key="nav_new_analysis"):
            navigate_to('new_analysis')
            # Removed st.rerun() call to prevent infinite loop
            return
        
        if st.button("Report History", key="nav_report_history"):
            navigate_to('report_history')
            # Removed st.rerun() call to prevent infinite loop
            return
        
        if st.button("Prompt Library", key="nav_prompt_library"):
            navigate_to('prompt_library')
            # Removed st.rerun() call to prevent infinite loop
            return
        
        if st.button("Settings", key="nav_settings"):
            navigate_to('settings')
            # Removed st.rerun() call to prevent infinite loop
            return
        
        st.markdown("---")
        
        if st.button("Logout", key="nav_logout"):
            st.session_state.user = None
            navigate_to('login')
            # Removed st.rerun() call to prevent infinite loop
            return
    
    # Dashboard content
    st.header("Dashboard")
    
    # Check if user has GA accounts
    ga_accounts = get_ga_accounts(st.session_state.user['id'])
    
    if not ga_accounts:
        st.warning("You haven't connected any Google Analytics accounts yet.")
        
        if st.button("Connect Google Analytics"):
            navigate_to('settings')
            # Removed st.rerun() call to prevent infinite loop
            return
        
        return
    
    # Check if user has API keys
    api_keys = get_api_keys(st.session_state.user['id'])
    
    if not api_keys:
        st.warning("You haven't added any API keys yet.")
        
        if st.button("Add API Keys"):
            navigate_to('settings')
            # Removed st.rerun() call to prevent infinite loop
            return
        
        return
    
    # Recent reports
    st.subheader("Recent Reports")
    
    reports = get_reports(st.session_state.user['id'])
    
    if not reports:
        st.info("You haven't created any reports yet.")
        
        if st.button("Create New Analysis"):
            navigate_to('new_analysis')
            # Removed st.rerun() call to prevent infinite loop
            return
    else:
        # Sort reports by creation date (newest first)
        reports.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Display the 5 most recent reports
        for report in reports[:5]:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{report['title']}**")
                st.write(f"Created: {report['created_at'][:10]}")
            
            with col2:
                if st.button("View", key=f"view_{report['id']}"):
                    st.session_state.selected_report = report
                    navigate_to('view_report')
                    # Removed st.rerun() call to prevent infinite loop
                    return
        
        if st.button("View All Reports"):
            navigate_to('report_history')
            # Removed st.rerun() call to prevent infinite loop
            return
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("New Analysis"):
            navigate_to('new_analysis')
            # Removed st.rerun() call to prevent infinite loop
            return
    
    with col2:
        if st.button("Manage Prompts"):
            navigate_to('prompt_library')
            # Removed st.rerun() call to prevent infinite loop
            return
    
    with col3:
        if st.button("Account Settings"):
            navigate_to('settings')
            # Removed st.rerun() call to prevent infinite loop
            return

def render_new_analysis():
    """Render the new analysis page."""
    st.title("New Analysis")
    
    # Check if user has GA accounts
    ga_accounts = get_ga_accounts(st.session_state.user['id'])
    
    if not ga_accounts:
        st.warning("You haven't connected any Google Analytics accounts yet.")
        
        if st.button("Connect Google Analytics"):
            navigate_to('settings')
            # Removed st.rerun() call to prevent infinite loop
            return
        
        return
    
    # Check if user has API keys
    api_keys = get_api_keys(st.session_state.user['id'])
    
    if not api_keys or 'openai' not in api_keys:
        st.warning("You need to add an OpenAI API key to use this feature.")
        
        if st.button("Add API Keys"):
            navigate_to('settings')
            # Removed st.rerun() call to prevent infinite loop
            return
        
        return
    
    # Analysis form
    st.header("Analysis Configuration")
    
    with st.form("analysis_form"):
        # GA account selection
        st.subheader("Google Analytics Account")
        selected_account = st.selectbox(
            "Select Account",
            options=ga_accounts,
            format_func=lambda x: f"{x['account_name']} ({x['property_id']})"
        )
        
        # Date range
        st.subheader("Date Range")
        date_range_options = {
            "last_7_days": "Last 7 Days",
            "last_30_days": "Last 30 Days",
            "last_90_days": "Last 90 Days",
            "last_year": "Last Year",
            "custom": "Custom Range"
        }
        
        selected_date_range = st.selectbox(
            "Select Date Range",
            options=list(date_range_options.keys()),
            format_func=lambda x: date_range_options[x]
        )
        
        if selected_date_range == "custom":
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input("Start Date")
            
            with col2:
                end_date = st.date_input("End Date")
        
        # Analysis type
        st.subheader("Analysis Type")
        analysis_types = {
            "general": "General Overview",
            "traffic": "Traffic Analysis",
            "conversion": "Conversion Analysis",
            "user_behavior": "User Behavior Analysis",
            "anomaly": "Anomaly Detection",
            "comparative": "Comparative Analysis"
        }
        
        selected_analysis_key = st.selectbox(
            "Select Analysis Type",
            options=list(analysis_types.keys()),
            format_func=lambda x: analysis_types[x]
        )
        
        # Prompt selection
        st.subheader("Analysis Prompt")
        
        # Get prompts for the selected analysis type
        prompts = get_prompts(st.session_state.user['id'])
        filtered_prompts = [p for p in prompts if p.get('category') == selected_analysis_key]
        
        prompt_options = [{"id": None, "title": "Default Prompt"}] + filtered_prompts
        
        selected_prompt_index = st.selectbox(
            "Select Prompt",
            options=range(len(prompt_options)),
            format_func=lambda i: prompt_options[i]['title']
        )
        
        selected_prompt = prompt_options[selected_prompt_index] if selected_prompt_index > 0 else None
        
        # Report details
        st.subheader("Report Details")
        report_title = st.text_input("Report Title", value=f"{analysis_types[selected_analysis_key]} - {datetime.datetime.now().strftime('%Y-%m-%d')}")
        report_description = st.text_area("Report Description", value=f"Analysis of {selected_analysis_key} data from Google Analytics.")
        
        # Submit button
        submitted = st.form_submit_button("Run Analysis")
    
    if submitted:
        try:
            st.info("Running analysis... This may take a few moments.")
            
            # Create date range dict
            if selected_date_range == "custom":
                date_range = {
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d")
                }
            else:
                # Map selected option to GA4 date range format
                date_mapping = {
                    "last_7_days": {"start_date": "7daysAgo", "end_date": "today"},
                    "last_30_days": {"start_date": "30daysAgo", "end_date": "today"},
                    "last_90_days": {"start_date": "90daysAgo", "end_date": "today"},
                    "last_year": {"start_date": "365daysAgo", "end_date": "today"}
                }
                date_range = date_mapping[selected_date_range]
            
            # Get analysis configuration
            if selected_analysis_key == "general":
                analysis_config = AnalysisWorkflow.get_weekly_report_config()
            elif selected_analysis_key == "traffic":
                analysis_config = AnalysisWorkflow.get_traffic_overview_config()
            elif selected_analysis_key == "conversion":
                analysis_config = AnalysisWorkflow.get_conversion_overview_config()
            elif selected_analysis_key == "user_behavior":
                analysis_config = AnalysisWorkflow.get_user_behavior_config()
            elif selected_analysis_key == "comparative":
                analysis_config = AnalysisWorkflow.get_comparative_report_config()
            else:
                analysis_config = AnalysisWorkflow.get_weekly_report_config()
            
            # Add date range to config
            analysis_config["date_range"] = date_range
            
            # Create analysis pipeline
            pipeline = AnalysisPipeline(
                ga_connector=GoogleAnalyticsConnector(
                    credentials_path=selected_account['credentials_path']
                ),
                insight_generator=AnalyticsInsightGenerator(
                    llm_provider=LLMFactory.create_provider(
                        provider="openai",
                        api_key=api_keys['openai']['api_key']
                    )
                )
            )
            
            # Run analysis
            results = pipeline.run_complete_analysis(
                property_id=selected_account['property_id'],
                metrics=analysis_config['metrics'],
                dimensions=analysis_config['dimensions'],
                date_range=analysis_config['date_range'],
                analysis_type=analysis_config['analysis_type'],
                custom_prompt=selected_prompt['prompt_template'] if selected_prompt else None
            )
            
            # Generate report
            report_generator = ReportGenerator()
            report_content = report_generator.generate_markdown_report(
                title=report_title,
                description=report_description,
                data=results['processed_data'],
                insights=results['insights'],
                metadata=results['metadata']
            )
            
            # Save report to file
            report_path = report_generator.save_report(
                content=report_content,
                directory=REPORTS_DIR,
                filename=f"{selected_analysis_key}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            
            # Save report to database
            report = add_report(
                user_id=st.session_state.user['id'],
                title=report_title,
                description=report_description,
                analysis_type=selected_analysis_key,
                report_content=report_content,
                metadata=results['metadata'],
                file_path=report_path
            )
            
            # Increment prompt usage if a prompt was used
            if selected_prompt:
                increment_prompt_usage(selected_prompt['id'])
            
            # Show success message
            st.success("Analysis completed successfully!")
            
            # Display the report
            st.markdown("## Report Preview")
            st.markdown(report_content)
            
            # Provide download link
            st.markdown(get_file_download_link(report_path, "Download Report"), unsafe_allow_html=True)
            
            # Option to view in report history
            if st.button("Go to Report History"):
                navigate_to('report_history')
                # Removed st.rerun() call to prevent infinite loop
                return
            
        except Exception as e:
            st.error(f"Error running analysis: {str(e)}")
            logger.error(f"Analysis error: {str(e)}", exc_info=True)

def render_report_history():
    """Render the report history page."""
    st.title("Report History")
    
    # Get all reports for the user
    reports = get_reports(st.session_state.user['id'])
    
    if not reports:
        st.info("You haven't created any reports yet.")
        
        if st.button("Create New Analysis"):
            navigate_to('new_analysis')
            # Removed st.rerun() call to prevent infinite loop
            return
        
        return
    
    # Sort reports by creation date (newest first)
    reports.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Filtering options
    st.header("Filter Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get unique analysis types
        analysis_types = list(set(report['analysis_type'] for report in reports))
        selected_types = st.multiselect("Analysis Type", analysis_types, default=analysis_types)
    
    with col2:
        # Date range filter
        date_filter = st.selectbox("Date Filter", ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"])
        
        if date_filter != "All Time":
            days = int(date_filter.split()[1])
            cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
            reports = [r for r in reports if r['created_at'] > cutoff_date]
    
    # Filter by selected types
    if selected_types:
        reports = [r for r in reports if r['analysis_type'] in selected_types]
    
    # Display reports
    st.header("Reports")
    
    for report in reports:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"**{report['title']}**")
            st.write(f"Created: {report['created_at'][:10]}")
        
        with col2:
            st.write(f"Type: {report['analysis_type']}")
            if report.get('is_favorite'):
                st.write("⭐ Favorite")
        
        with col3:
            if st.button("View", key=f"view_{report['id']}"):
                st.session_state.selected_report = report
                navigate_to('view_report')
                # Removed st.rerun() call to prevent infinite loop
                return
            
            # Toggle favorite status
            favorite_label = "Remove Favorite" if report.get('is_favorite') else "Add Favorite"
            if st.button(favorite_label, key=f"fav_{report['id']}"):
                update_report(report['id'], {'is_favorite': not report.get('is_favorite', False)})
                # Use a single rerun at the end of the function instead
                return
        
        st.markdown("---")

def render_view_report():
    """Render the view report page."""
    if 'selected_report' not in st.session_state:
        st.error("No report selected")
        
        if st.button("Back to Report History"):
            navigate_to('report_history')
            # Removed st.rerun() call to prevent infinite loop
            return
        
        return
    
    report = st.session_state.selected_report
    
    st.title(report['title'])
    
    st.write(f"**Description:** {report['description']}")
    st.write(f"**Analysis Type:** {report['analysis_type']}")
    st.write(f"**Created:** {report['created_at'][:10]}")
    
    # Display report content
    if report.get('file_path') and os.path.exists(report['file_path']):
        with open(report['file_path'], 'r') as f:
            report_content = f.read()
        
        st.markdown(report_content)
        
        # Provide download link
        st.markdown(get_file_download_link(report['file_path'], "Download Report"), unsafe_allow_html=True)
    else:
        st.warning("Report file not found")
    
    # Actions
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Toggle favorite status
        favorite_label = "Remove from Favorites" if report.get('is_favorite') else "Add to Favorites"
        if st.button(favorite_label):
            update_report(report['id'], {'is_favorite': not report.get('is_favorite', False)})
            st.session_state.selected_report = update_report(report['id'], {})  # Refresh the report
            # Removed st.rerun() call to prevent infinite loop
            return
    
    with col2:
        # Run similar analysis
        if st.button("Run Similar Analysis"):
            navigate_to('new_analysis')
            # Removed st.rerun() call to prevent infinite loop
            return
    
    with col3:
        # Back to report history
        if st.button("Back to Report History"):
            navigate_to('report_history')
            # Removed st.rerun() call to prevent infinite loop
            return

def render_prompt_library():
    """Render the prompt library page."""
    st.title("Prompt Library")
    
    # Get all prompts for the user
    prompts = get_prompts(st.session_state.user['id'])
    
    # Tabs for viewing and creating prompts
    tab1, tab2 = st.tabs(["View Prompts", "Create New Prompt"])
    
    with tab1:
        if not prompts:
            st.info("No prompts found. Create your first prompt!")
        else:
            # Filtering options
            st.header("Filter Prompts")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Get unique categories
                categories = list(set(prompt.get('category', 'uncategorized') for prompt in prompts))
                selected_categories = st.multiselect("Category", categories, default=categories)
            
            with col2:
                # Show system prompts option
                show_system = st.checkbox("Show System Prompts", value=True)
            
            # Filter prompts
            filtered_prompts = prompts
            
            if selected_categories:
                filtered_prompts = [p for p in filtered_prompts if p.get('category') in selected_categories]
            
            if not show_system:
                filtered_prompts = [p for p in filtered_prompts if not p.get('is_system', False)]
            
            # Display prompts
            st.header("Prompts")
            
            for prompt in filtered_prompts:
                with st.expander(f"{prompt['title']} ({prompt.get('category', 'uncategorized')})"):
                    st.write(f"**Description:** {prompt['description']}")
                    
                    if prompt.get('is_system'):
                        st.write("**System Prompt**")
                    
                    st.write(f"**Usage Count:** {prompt.get('usage_count', 0)}")
                    
                    # Display template
                    st.subheader("Template")
                    st.code(prompt['prompt_template'])
                    
                    # Display parameters
                    if prompt.get('parameters'):
                        st.subheader("Parameters")
                        for param_name, param_default in prompt['parameters'].items():
                            st.write(f"- **{param_name}:** {param_default}")
                    
                    # Actions
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Toggle favorite status
                        favorite_label = "Remove from Favorites" if prompt.get('is_favorite') else "Add to Favorites"
                        if not prompt.get('is_system') and st.button(favorite_label, key=f"fav_{prompt['id']}"):
                            update_prompt(prompt['id'], {'is_favorite': not prompt.get('is_favorite', False)})
                            # Removed st.rerun() call to prevent infinite loop
                            return
                    
                    with col2:
                        # Edit prompt (only for user prompts)
                        if not prompt.get('is_system') and st.button("Edit", key=f"edit_{prompt['id']}"):
                            st.session_state.selected_prompt = prompt
                            st.session_state.editing_prompt = True
                            # Removed st.rerun() call to prevent infinite loop
                            return
    
    with tab2:
        # Check if we're editing an existing prompt
        editing = hasattr(st.session_state, 'editing_prompt') and st.session_state.editing_prompt
        selected_prompt = getattr(st.session_state, 'selected_prompt', None) if editing else None
        
        if editing:
            st.header(f"Edit Prompt: {selected_prompt['title']}")
        else:
            st.header("Create New Prompt")
        
        # Prompt form
        title = st.text_input("Title", value=selected_prompt['title'] if editing else "")
        description = st.text_area("Description", value=selected_prompt['description'] if editing else "")
        
        # Categories
        categories = ["general", "traffic", "conversion", "user_behavior", "anomaly", "comparative", "custom"]
        category = st.selectbox("Category", categories, index=categories.index(selected_prompt['category']) if editing and selected_prompt.get('category') in categories else 0)
        
        # Template
        st.subheader("Prompt Template")
        st.write("Use {placeholders} for parameters that will be filled at runtime.")
        prompt_template = st.text_area("Template", value=selected_prompt['prompt_template'] if editing else "", height=200)
        
        # Parameters
        st.subheader("Parameters")
        st.write("Define default values for parameters used in the template.")
        
        # Initialize parameters
        if 'prompt_params' not in st.session_state:
            st.session_state.prompt_params = selected_prompt.get('parameters', {}).copy() if editing else {}
        
        # Display existing parameters
        for param_name, param_value in list(st.session_state.prompt_params.items()):
            col1, col2, col3 = st.columns([2, 3, 1])
            
            with col1:
                st.text(param_name)
            
            with col2:
                new_value = st.text_input(f"Value for {param_name}", value=param_value, key=f"param_{param_name}")
                st.session_state.prompt_params[param_name] = new_value
            
            with col3:
                if st.button("Remove", key=f"remove_{param_name}"):
                    del st.session_state.prompt_params[param_name]
                    # Removed st.rerun() call to prevent infinite loop
                    return
        
        # Add new parameter
        st.subheader("Add Parameter")
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            new_param_name = st.text_input("Parameter Name")
        
        with col2:
            new_param_value = st.text_input("Default Value")
        
        with col3:
            if st.button("Add") and new_param_name:
                st.session_state.prompt_params[new_param_name] = new_param_value
                # Removed st.rerun() call to prevent infinite loop
                return
        
        # Save button
        if editing:
            if st.button("Update Prompt"):
                if title and prompt_template:
                    # Update the prompt
                    update_prompt(selected_prompt['id'], {
                        'title': title,
                        'description': description,
                        'prompt_template': prompt_template,
                        'parameters': st.session_state.prompt_params,
                        'category': category
                    })
                    
                    # Reset editing state
                    st.session_state.editing_prompt = False
                    st.session_state.selected_prompt = None
                    
                    # Clear parameters
                    st.session_state.prompt_params = {}
                    
                    st.success("Prompt updated successfully!")
                    # Removed st.rerun() call to prevent infinite loop
                    return
                else:
                    st.error("Title and template are required")
            
            if st.button("Cancel"):
                # Reset editing state
                st.session_state.editing_prompt = False
                st.session_state.selected_prompt = None
                
                # Clear parameters
                st.session_state.prompt_params = {}
                
                # Removed st.rerun() call to prevent infinite loop
                return
        else:
            if st.button("Create Prompt"):
                if title and prompt_template:
                    # Add the prompt
                    add_prompt(
                        user_id=st.session_state.user['id'],
                        title=title,
                        description=description,
                        prompt_template=prompt_template,
                        parameters=st.session_state.prompt_params,
                        category=category
                    )
                    
                    # Clear parameters
                    st.session_state.prompt_params = {}
                    
                    st.success("Prompt created successfully!")
                    # Removed st.rerun() call to prevent infinite loop
                    return
                else:
                    st.error("Title and template are required")

def render_settings():
    """Render the settings page."""
    st.title("Settings")
    
    # Tabs for different settings
    tab1, tab2, tab3 = st.tabs(["Google Analytics", "API Keys", "User Settings"])
    
    with tab1:
        st.header("Google Analytics Accounts")
        
        # Display existing accounts
        ga_accounts = get_ga_accounts(st.session_state.user['id'])
        
        if ga_accounts:
            st.subheader("Connected Accounts")
            
            for account in ga_accounts:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{account['account_name']}**")
                    st.write(f"Property ID: {account['property_id']}")
                    st.write(f"Connected: {account['created_at'][:10]}")
                
                with col2:
                    # Add delete button
                    if st.button("Delete", key=f"delete_{account['id']}"):
                        if delete_ga_account(account['id']):
                            st.success(f"Account {account['account_name']} deleted successfully!")
                            # Removed st.rerun() call to prevent infinite loop
                            return
                        else:
                            st.error("Failed to delete account")
                
                st.markdown("---")
        else:
            st.info("No Google Analytics accounts connected")
        
        # Add new account
        st.subheader("Add New Account")
        
        with st.form("add_ga_account_form"):
            account_name = st.text_input("Account Name")
            property_id = st.text_input("Property ID")
            
            # Add file uploader for OAuth credentials
            st.write("**OAuth Credentials**")
            st.write("Upload your Google OAuth credentials JSON file. This file contains the client ID and secret needed to authenticate with Google Analytics.")
            uploaded_credentials = st.file_uploader("Upload OAuth Credentials JSON", type=["json"])
            
            # Submit button
            submitted = st.form_submit_button("Add Account")
        
        if submitted:
            if account_name and property_id and uploaded_credentials:
                try:
                    # Create a unique filename for the credentials
                    credentials_filename = f"ga_oauth_{st.session_state.user['id']}_{uuid.uuid4()}.json"
                    
                    # Save the uploaded credentials file
                    credentials_path = save_uploaded_file(
                        uploaded_credentials, 
                        CREDENTIALS_DIR, 
                        credentials_filename
                    )
                    
                    # Add the account
                    add_ga_account(
                        user_id=st.session_state.user['id'],
                        account_name=account_name,
                        property_id=property_id,
                        credentials_path=credentials_path
                    )
                    
                    st.success("Account added successfully!")
                    # Removed st.rerun() call to prevent infinite loop
                    return
                except Exception as e:
                    st.error(f"Error adding account: {str(e)}")
                    logger.error(f"Error adding GA account: {str(e)}", exc_info=True)
            else:
                st.error("Please fill in all fields and upload credentials file")
    
    with tab2:
        st.header("API Keys")
        
        # Display existing API keys
        api_keys = get_api_keys(st.session_state.user['id'])
        
        if api_keys:
            st.subheader("Saved API Keys")
            
            for service, key_data in api_keys.items():
                st.write(f"**{service.capitalize()}**")
                st.write(f"Added: {key_data['created_at'][:10]}")
                st.write(f"Last Updated: {key_data['updated_at'][:10]}")
                
                # Show masked key
                masked_key = "•" * 20 + key_data['api_key'][-5:] if len(key_data['api_key']) > 5 else "•" * 20
                st.code(masked_key)
                
                st.markdown("---")
        else:
            st.info("No API keys added yet")
        
        # Add new API key
        st.subheader("Add/Update API Key")
        
        with st.form("add_api_key_form"):
            service = st.selectbox("Service", ["openai", "anthropic", "google", "ollama"])
            api_key = st.text_input("API Key", type="password")
            
            # Submit button
            submitted = st.form_submit_button("Save API Key")
        
        if submitted:
            if service and api_key:
                # Add the API key
                add_api_key(
                    user_id=st.session_state.user['id'],
                    service=service,
                    api_key=api_key
                )
                
                st.success(f"{service.capitalize()} API key saved successfully!")
                # Removed st.rerun() call to prevent infinite loop
                return
            else:
                st.error("Please select a service and enter an API key")
    
    with tab3:
        st.header("User Settings")
        
        # Display user information
        st.subheader("User Information")
        st.write(f"**Name:** {st.session_state.user['name']}")
        st.write(f"**Email:** {st.session_state.user['email']}")
        st.write(f"**Account Created:** {st.session_state.user['created_at'][:10]}")
        st.write(f"**Last Login:** {st.session_state.user['last_login'][:10]}")
        
        # Update user information
        st.subheader("Update Information")
        
        with st.form("update_user_form"):
            new_name = st.text_input("Name", value=st.session_state.user['name'])
            
            # Submit button
            submitted = st.form_submit_button("Update Information")
        
        if submitted:
            if new_name:
                # Update user information
                users = load_json_db(USERS_DB)
                
                for user in users:
                    if user['id'] == st.session_state.user['id']:
                        user['name'] = new_name
                        break
                
                save_json_db(USERS_DB, users)
                
                # Update session state
                st.session_state.user['name'] = new_name
                
                st.success("User information updated successfully!")
                # Removed st.rerun() call to prevent infinite loop
                return
            else:
                st.error("Name cannot be empty")

# Run the app
if __name__ == "__main__":
    main()
