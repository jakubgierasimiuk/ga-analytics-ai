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

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR, REPORTS_DIR, PROMPTS_DIR, CONFIG_DIR]:
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

# Initialize system prompts
def init_system_prompts():
    """Initialize system prompts if they don't exist."""
    prompts = load_json_db(PROMPTS_DB)
    
    # Check if system prompts exist
    system_prompts_exist = any(prompt.get('is_system', False) for prompt in prompts)
    
    if not system_prompts_exist:
        # Create system prompts
        prompt_library = AnalyticsPromptLibrary()
        
        system_prompts = [
            {
                'title': 'General Analysis',
                'description': 'General analysis of Google Analytics data',
                'prompt_template': prompt_library.get_general_analysis_prompt().template,
                'parameters': prompt_library.get_general_analysis_prompt().parameters,
                'category': 'general'
            },
            {
                'title': 'Traffic Analysis',
                'description': 'Analysis of traffic sources and patterns',
                'prompt_template': prompt_library.get_traffic_analysis_prompt().template,
                'parameters': prompt_library.get_traffic_analysis_prompt().parameters,
                'category': 'traffic'
            },
            {
                'title': 'Conversion Analysis',
                'description': 'Analysis of conversion rates and revenue',
                'prompt_template': prompt_library.get_conversion_analysis_prompt().template,
                'parameters': prompt_library.get_conversion_analysis_prompt().parameters,
                'category': 'conversion'
            },
            {
                'title': 'User Behavior Analysis',
                'description': 'Analysis of user engagement and behavior',
                'prompt_template': prompt_library.get_user_behavior_analysis_prompt().template,
                'parameters': prompt_library.get_user_behavior_analysis_prompt().parameters,
                'category': 'user_behavior'
            },
            {
                'title': 'Anomaly Detection',
                'description': 'Detection and analysis of anomalies in the data',
                'prompt_template': prompt_library.get_anomaly_detection_prompt().template,
                'parameters': prompt_library.get_anomaly_detection_prompt().parameters,
                'category': 'anomaly'
            },
            {
                'title': 'Comparative Analysis',
                'description': 'Comparison of two time periods',
                'prompt_template': prompt_library.get_comparative_analysis_prompt().template,
                'parameters': prompt_library.get_comparative_analysis_prompt().parameters,
                'category': 'comparative'
            }
        ]
        
        for prompt_data in system_prompts:
            new_prompt = {
                'id': str(uuid.uuid4()),
                'user_id': None,
                'title': prompt_data['title'],
                'description': prompt_data['description'],
                'prompt_template': prompt_data['prompt_template'],
                'parameters': prompt_data['parameters'],
                'category': prompt_data['category'],
                'created_at': datetime.datetime.now().isoformat(),
                'updated_at': datetime.datetime.now().isoformat(),
                'usage_count': 0,
                'is_favorite': False,
                'is_system': True,
                'tags': ['system']
            }
            
            prompts.append(new_prompt)
        
        save_json_db(PROMPTS_DB, prompts)

# Call initialization
init_system_prompts()

# Helper functions
def get_file_download_link(file_path: str, link_text: str) -> str:
    """Generate a download link for a file."""
    try:
        with open(file_path, 'r') as f:
            data = f.read()
        
        b64 = base64.b64encode(data.encode()).decode()
        filename = os.path.basename(file_path)
        mime_type = 'text/markdown' if file_path.endswith('.md') else 'text/html'
        
        href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'
        return href
    except Exception as e:
        logger.error(f"Error creating download link: {e}")
        return f"Error creating download link: {str(e)}"

def format_date_for_ga(date_obj: datetime.date) -> str:
    """Format a date for Google Analytics API."""
    return date_obj.strftime('%Y-%m-%d')

# Authentication and session management
def init_session_state():
    """Initialize session state variables."""
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'login'
    
    if 'ga_connector' not in st.session_state:
        st.session_state.ga_connector = None
    
    if 'analysis_pipeline' not in st.session_state:
        st.session_state.analysis_pipeline = None
    
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = None

def login_user(email: str, name: str):
    """Log in a user."""
    user = get_user_by_email(email)
    
    if not user:
        user = add_user(email, name)
    else:
        update_user_login(user['id'])
    
    st.session_state.user = user
    st.session_state.authenticated = True
    st.session_state.current_page = 'dashboard'

def logout_user():
    """Log out the current user."""
    st.session_state.user = None
    st.session_state.authenticated = False
    st.session_state.current_page = 'login'
    st.session_state.ga_connector = None
    st.session_state.analysis_pipeline = None
    st.session_state.report_generator = None

# Initialize session state
init_session_state()

# Page navigation
def navigate_to(page: str):
    """Navigate to a specific page."""
    st.session_state.current_page = page

# UI Components
def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("GA Analytics AI")
    
    if st.session_state.authenticated:
        st.sidebar.write(f"Welcome, {st.session_state.user['name']}!")
        
        st.sidebar.header("Navigation")
        
        if st.sidebar.button("Dashboard", key="nav_dashboard"):
            navigate_to('dashboard')
        
        if st.sidebar.button("New Analysis", key="nav_new_analysis"):
            navigate_to('new_analysis')
        
        if st.sidebar.button("Report History", key="nav_report_history"):
            navigate_to('report_history')
        
        if st.sidebar.button("Prompt Library", key="nav_prompt_library"):
            navigate_to('prompt_library')
        
        if st.sidebar.button("Settings", key="nav_settings"):
            navigate_to('settings')
        
        st.sidebar.markdown("---")
        
        if st.sidebar.button("Logout", key="nav_logout"):
            logout_user()
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 GA Analytics AI")

def render_login_page():
    """Render the login page."""
    st.title("Welcome to GA Analytics AI")
    st.write("Your AI-powered Google Analytics assistant")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Login")
        email = st.text_input("Email")
        name = st.text_input("Name")
        
        if st.button("Login"):
            if email and name:
                login_user(email, name)
            else:
                st.error("Please enter both email and name")
    
    with col2:
        st.header("Features")
        st.markdown("""
        - AI-powered analysis of Google Analytics data
        - Automatic insight generation
        - Custom report creation
        - Historical report tracking
        - Prompt library for customized analysis
        """)

def render_dashboard():
    """Render the dashboard page."""
    st.title("Dashboard")
    
    # Check if user has GA accounts
    ga_accounts = get_ga_accounts(st.session_state.user['id'])
    
    if not ga_accounts:
        st.warning("You don't have any Google Analytics accounts connected. Go to Settings to add one.")
        
        if st.button("Go to Settings"):
            navigate_to('settings')
        
        return
    
    # Check if user has API keys
    api_keys = get_api_keys(st.session_state.user['id'])
    
    if not api_keys:
        st.warning("You don't have any LLM API keys configured. Go to Settings to add one.")
        
        if st.button("Go to Settings"):
            navigate_to('settings')
        
        return
    
    # Display recent reports
    st.header("Recent Reports")
    
    reports = get_reports(st.session_state.user['id'])
    reports.sort(key=lambda x: x['created_at'], reverse=True)
    
    if not reports:
        st.info("You haven't created any reports yet. Create your first analysis!")
        
        if st.button("New Analysis"):
            navigate_to('new_analysis')
        
        return
    
    # Display the 5 most recent reports
    recent_reports = reports[:5]
    
    for report in recent_reports:
        with st.expander(f"{report['title']} ({report['created_at'][:10]})"):
            st.write(f"**Description:** {report['description']}")
            st.write(f"**Analysis Type:** {report['analysis_type']}")
            
            if report.get('file_path') and os.path.exists(report['file_path']):
                st.markdown(get_file_download_link(report['file_path'], "Download Report"), unsafe_allow_html=True)
            
            if st.button("View Details", key=f"view_{report['id']}"):
                st.session_state.selected_report = report
                navigate_to('view_report')
    
    st.markdown("---")
    
    # Quick actions
    st.header("Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("New Analysis", key="dashboard_new_analysis"):
            navigate_to('new_analysis')
    
    with col2:
        if st.button("View All Reports", key="dashboard_all_reports"):
            navigate_to('report_history')
    
    with col3:
        if st.button("Prompt Library", key="dashboard_prompt_library"):
            navigate_to('prompt_library')

def render_new_analysis():
    """Render the new analysis page."""
    st.title("New Analysis")
    
    # Check if user has GA accounts
    ga_accounts = get_ga_accounts(st.session_state.user['id'])
    
    if not ga_accounts:
        st.warning("You don't have any Google Analytics accounts connected. Go to Settings to add one.")
        
        if st.button("Go to Settings"):
            navigate_to('settings')
        
        return
    
    # Check if user has API keys
    api_keys = get_api_keys(st.session_state.user['id'])
    
    if not api_keys:
        st.warning("You don't have any LLM API keys configured. Go to Settings to add one.")
        
        if st.button("Go to Settings"):
            navigate_to('settings')
        
        return
    
    # Analysis configuration
    st.header("Analysis Configuration")
    
    # Step 1: Select GA account
    st.subheader("Step 1: Select Google Analytics Account")
    
    account_options = {account['account_name']: account for account in ga_accounts}
    selected_account_name = st.selectbox("Select Account", list(account_options.keys()))
    selected_account = account_options[selected_account_name]
    
    # Step 2: Select date range
    st.subheader("Step 2: Select Date Range")
    
    date_range_type = st.radio("Date Range Type", ["Last N Days", "Custom Range", "Comparison"])
    
    if date_range_type == "Last N Days":
        days = st.slider("Number of Days", 7, 90, 30)
        end_date = datetime.datetime.now().date()
        start_date = end_date - datetime.timedelta(days=days)
        
        date_range = {
            'start_date': format_date_for_ga(start_date),
            'end_date': format_date_for_ga(end_date)
        }
        
        st.write(f"Selected range: {start_date} to {end_date}")
        
    elif date_range_type == "Custom Range":
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", datetime.datetime.now().date() - datetime.timedelta(days=30))
        
        with col2:
            end_date = st.date_input("End Date", datetime.datetime.now().date())
        
        date_range = {
            'start_date': format_date_for_ga(start_date),
            'end_date': format_date_for_ga(end_date)
        }
        
    else:  # Comparison
        st.write("Compare two time periods:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Current Period:")
            current_start = st.date_input("Start Date", datetime.datetime.now().date() - datetime.timedelta(days=30))
            current_end = st.date_input("End Date", datetime.datetime.now().date())
        
        with col2:
            st.write("Previous Period:")
            previous_start = st.date_input("Start Date", current_start - datetime.timedelta(days=30))
            previous_end = st.date_input("End Date", current_end - datetime.timedelta(days=30))
        
        date_range = [
            {
                'start_date': format_date_for_ga(current_start),
                'end_date': format_date_for_ga(current_end)
            },
            {
                'start_date': format_date_for_ga(previous_start),
                'end_date': format_date_for_ga(previous_end)
            }
        ]
    
    # Step 3: Select analysis type
    st.subheader("Step 3: Select Analysis Type")
    
    analysis_types = {
        "general": "General Analysis",
        "traffic": "Traffic Analysis",
        "conversion": "Conversion Analysis",
        "user_behavior": "User Behavior Analysis",
        "anomaly": "Anomaly Detection"
    }
    
    if date_range_type == "Comparison":
        analysis_types["comparative"] = "Comparative Analysis"
    
    selected_analysis_type = st.selectbox("Analysis Type", list(analysis_types.values()))
    selected_analysis_key = list(analysis_types.keys())[list(analysis_types.values()).index(selected_analysis_type)]
    
    # Step 4: Configure metrics and dimensions
    st.subheader("Step 4: Configure Metrics and Dimensions")
    
    # Get predefined configuration based on analysis type
    if selected_analysis_key == "general":
        config = AnalysisWorkflow.get_weekly_report_config()
    elif selected_analysis_key == "traffic":
        config = AnalysisWorkflow.get_traffic_overview_config()
    elif selected_analysis_key == "conversion":
        config = AnalysisWorkflow.get_conversion_overview_config()
    elif selected_analysis_key == "user_behavior":
        config = AnalysisWorkflow.get_user_behavior_config()
    elif selected_analysis_key == "comparative":
        config = AnalysisWorkflow.get_comparative_report_config()
    else:
        # Default to traffic config for anomaly
        config = AnalysisWorkflow.get_traffic_overview_config()
    
    # Allow user to customize metrics and dimensions
    st.write("Metrics:")
    selected_metrics = st.multiselect("Select Metrics", config['metrics'], default=config['metrics'])
    
    st.write("Dimensions:")
    selected_dimensions = st.multiselect("Select Dimensions", config['dimensions'], default=config['dimensions'])
    
    # Step 5: Configure prompt
    st.subheader("Step 5: Configure Analysis Prompt")
    
    # Get prompts for the selected analysis type
    prompts = get_prompts(st.session_state.user['id'])
    type_prompts = [p for p in prompts if p.get('category') == selected_analysis_key]
    
    if type_prompts:
        prompt_options = {f"{p['title']}": p for p in type_prompts}
        selected_prompt_name = st.selectbox("Select Prompt Template", list(prompt_options.keys()))
        selected_prompt = prompt_options[selected_prompt_name]
        
        # Display prompt parameters
        st.write("Prompt Parameters:")
        
        prompt_params = {}
        for param_name, param_default in selected_prompt.get('parameters', {}).items():
            param_value = st.text_input(f"{param_name}", value=param_default)
            prompt_params[param_name] = param_value
    else:
        st.warning(f"No prompts found for {selected_analysis_type}. Using default prompt.")
        selected_prompt = None
        prompt_params = {}
    
    # Step 6: Run analysis
    st.subheader("Step 6: Run Analysis")
    
    report_title = st.text_input("Report Title", f"{selected_analysis_type} - {datetime.datetime.now().strftime('%Y-%m-%d')}")
    report_description = st.text_area("Report Description", f"Analysis of {selected_account_name} data")
    
    if st.button("Run Analysis"):
        with st.spinner("Running analysis..."):
            # Initialize the pipeline if not already done
            if not st.session_state.analysis_pipeline:
                # Get API key for the selected LLM provider
                llm_provider = "openai"  # Default to OpenAI
                llm_api_key = api_keys.get(llm_provider, {}).get('api_key')
                
                if not llm_api_key:
                    st.error(f"No API key found for {llm_provider}. Please add one in Settings.")
                    return
                
                # Initialize the pipeline
                st.session_state.analysis_pipeline = AnalysisPipeline(
                    ga_credentials_path=selected_account['credentials_path'],
                    llm_provider=llm_provider,
                    llm_api_key=llm_api_key,
                    cache_dir=str(CACHE_DIR)
                )
                
                # Initialize the report generator
                st.session_state.report_generator = ReportGenerator(output_dir=str(REPORTS_DIR))
            
            # Authenticate with GA
            if not st.session_state.analysis_pipeline.authenticate_ga():
                st.error("Failed to authenticate with Google Analytics. Please check your credentials.")
                return
            
            try:
                # Run the analysis
                results = st.session_state.analysis_pipeline.run_complete_analysis(
                    property_id=selected_account['property_id'],
                    date_range=date_range,
                    metrics=selected_metrics,
                    dimensions=selected_dimensions,
                    analysis_type=selected_analysis_key,
                    **prompt_params
                )
                
                # Generate the report
                report_content = st.session_state.report_generator.generate_markdown_report(
                    analysis_results=results,
                    include_data_summary=True,
                    include_charts=True
                )
                
                # Save the report
                report_filename = f"{selected_analysis_key}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                report_path = st.session_state.report_generator.save_report(
                    report_content=report_content,
                    filename=report_filename,
                    format="markdown"
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
                
            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")

def render_report_history():
    """Render the report history page."""
    st.title("Report History")
    
    # Get all reports for the user
    reports = get_reports(st.session_state.user['id'])
    
    if not reports:
        st.info("You haven't created any reports yet.")
        
        if st.button("Create New Analysis"):
            navigate_to('new_analysis')
        
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
            
            # Toggle favorite status
            favorite_label = "Remove Favorite" if report.get('is_favorite') else "Add Favorite"
            if st.button(favorite_label, key=f"fav_{report['id']}"):
                update_report(report['id'], {'is_favorite': not report.get('is_favorite', False)})
                st.experimental_rerun()
        
        st.markdown("---")

def render_view_report():
    """Render the view report page."""
    if 'selected_report' not in st.session_state:
        st.error("No report selected")
        
        if st.button("Back to Report History"):
            navigate_to('report_history')
        
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
            st.experimental_rerun()
    
    with col2:
        # Run similar analysis
        if st.button("Run Similar Analysis"):
            navigate_to('new_analysis')
    
    with col3:
        # Back to report history
        if st.button("Back to Report History"):
            navigate_to('report_history')

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
                            st.experimental_rerun()
                    
                    with col2:
                        # Edit prompt (only for user prompts)
                        if not prompt.get('is_system') and st.button("Edit", key=f"edit_{prompt['id']}"):
                            st.session_state.selected_prompt = prompt
                            st.session_state.editing_prompt = True
                            st.experimental_rerun()
    
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
                    st.experimental_rerun()
        
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
                st.experimental_rerun()
        
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
                    st.experimental_rerun()
                else:
                    st.error("Title and template are required")
            
            if st.button("Cancel"):
                # Reset editing state
                st.session_state.editing_prompt = False
                st.session_state.selected_prompt = None
                
                # Clear parameters
                st.session_state.prompt_params = {}
                
                st.experimental_rerun()
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
                    st.experimental_rerun()
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
                st.write(f"**{account['account_name']}**")
                st.write(f"Property ID: {account['property_id']}")
                st.write(f"Connected: {account['created_at'][:10]}")
                st.markdown("---")
        else:
            st.info("No Google Analytics accounts connected")
        
        # Add new account
        st.subheader("Add New Account")
        
        account_name = st.text_input("Account Name")
        property_id = st.text_input("Property ID")
        
        # In a real app, you would handle file upload for credentials
        # Here we'll just use a placeholder path
        credentials_path = os.path.join(CONFIG_DIR, f"ga_credentials_{st.session_state.user['id']}.json")
        
        if st.button("Add Account"):
            if account_name and property_id:
                # In a real app, you would save the uploaded credentials file
                # Here we'll just create a dummy file
                with open(credentials_path, 'w') as f:
                    f.write('{"dummy": "credentials"}')
                
                # Add the account
                add_ga_account(
                    user_id=st.session_state.user['id'],
                    account_name=account_name,
                    property_id=property_id,
                    credentials_path=credentials_path
                )
                
                st.success("Account added successfully!")
                st.experimental_rerun()
            else:
                st.error("Account name and property ID are required")
    
    with tab2:
        st.header("API Keys")
        
        # Display existing keys
        api_keys = get_api_keys(st.session_state.user['id'])
        
        if api_keys:
            st.subheader("Configured API Keys")
            
            for service, key in api_keys.items():
                st.write(f"**{service.capitalize()}**")
                st.write(f"Added: {key['created_at'][:10]}")
                st.markdown("---")
        else:
            st.info("No API keys configured")
        
        # Add new key
        st.subheader("Add/Update API Key")
        
        service = st.selectbox("Service", ["openai", "anthropic", "gemini"])
        api_key = st.text_input("API Key", type="password")
        
        if st.button("Save API Key"):
            if service and api_key:
                # Add the key
                add_api_key(
                    user_id=st.session_state.user['id'],
                    service=service,
                    api_key=api_key
                )
                
                st.success("API key saved successfully!")
                st.experimental_rerun()
            else:
                st.error("Service and API key are required")
    
    with tab3:
        st.header("User Settings")
        
        # Display user info
        st.subheader("User Information")
        
        st.write(f"**Name:** {st.session_state.user['name']}")
        st.write(f"**Email:** {st.session_state.user['email']}")
        st.write(f"**Account Created:** {st.session_state.user['created_at'][:10]}")
        
        # In a real app, you would allow updating user information
        st.subheader("Update Information")
        
        new_name = st.text_input("Name", value=st.session_state.user['name'])
        
        if st.button("Update"):
            if new_name:
                # In a real app, you would update the user information
                st.success("Information updated successfully!")
            else:
                st.error("Name is required")

# Main app
def main():
    """Main function to run the Streamlit app."""
    # Render sidebar
    render_sidebar()
    
    # Render the current page
    if not st.session_state.authenticated:
        render_login_page()
    else:
        if st.session_state.current_page == 'dashboard':
            render_dashboard()
        elif st.session_state.current_page == 'new_analysis':
            render_new_analysis()
        elif st.session_state.current_page == 'report_history':
            render_report_history()
        elif st.session_state.current_page == 'view_report':
            render_view_report()
        elif st.session_state.current_page == 'prompt_library':
            render_prompt_library()
        elif st.session_state.current_page == 'settings':
            render_settings()
        else:
            render_dashboard()  # Default to dashboard

if __name__ == "__main__":
    main()
