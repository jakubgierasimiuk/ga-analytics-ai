import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import tempfile
import re
import csv
from io import StringIO

# Import GA4 integration module
from ga_integration import GoogleAnalyticsData
# Import LLM integration module
from llm_integration import analyze_data_with_llm
# Import GA4 ecommerce metrics module
from ga4_ecommerce_metrics import (
    select_product_metrics_and_dimensions,
    get_safe_metrics_and_dimensions,
    is_compatible_combination,
    PRODUCT_ANALYSIS_TYPES,
    ITEM_SCOPED_METRICS,
    ITEM_SCOPED_DIMENSIONS,
    EVENT_SCOPED_METRICS,
    EVENT_SCOPED_DIMENSIONS
)
# Import enhanced product query understanding
from product_query_understanding import (
    detect_product_analysis_intent_enhanced,
    get_product_analysis_type_enhanced,
    extract_product_entities,
    select_product_metrics_and_dimensions_enhanced,
    generate_product_analysis_prompt,
    suggest_product_metrics_and_dimensions
)

# Set page configuration
st.set_page_config(
    page_title="GA Analytics AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'ga_accounts' not in st.session_state:
    st.session_state.ga_accounts = {}
if 'selected_account' not in st.session_state:
    st.session_state.selected_account = None
if 'reports' not in st.session_state:
    st.session_state.reports = []
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'service_account_key' not in st.session_state:
    st.session_state.service_account_key = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "o3"  # Default to o3 model
# No need to initialize max_tokens as we'll calculate it dynamically
if 'language' not in st.session_state:
    st.session_state.language = "en"  # Default language

# Define available OpenAI models
OPENAI_MODELS = {
    "o3": {"name": "o3", "description": "Najnowszy i najpotężniejszy model rozumowania z wiodącą wydajnością w kodowaniu, matematyce i nauce", "max_context": 128000},
    "gpt-4o": {"name": "gpt-4o", "description": "Najnowszy model GPT-4 z ulepszoną wydajnością i niższym kosztem", "max_context": 128000},
    "gpt-4-turbo": {"name": "gpt-4-turbo", "description": "Szybszy model GPT-4 z dużym kontekstem", "max_context": 128000},
    "gpt-4": {"name": "gpt-4", "description": "Standardowy model GPT-4", "max_context": 8192},
    "gpt-3.5-turbo": {"name": "gpt-3.5-turbo", "description": "Szybki i ekonomiczny model", "max_context": 16385}
}

# Text translations
TRANSLATIONS = {
    "en": {
        "navigation": "Navigation",
        "dashboard": "Dashboard",
        "new_analysis": "New Analysis",
        "report_history": "Report History",
        "prompt_library": "Prompt Library",
        "settings": "Settings",
        "select_analysis_type": "Select Analysis Type",
        "analysis_prompt": "Analysis Prompt",
        "select_prompt": "Select Prompt",
        "default_prompt": "Default Prompt",
        "use_custom_prompt": "Use custom prompt",
        "custom_prompt": "Custom Prompt",
        "report_details": "Report Details",
        "report_title": "Report Title",
        "report_description": "Report Description",
        "ga_account": "Google Analytics Account",
        "select_account": "Select Account",
        "date_range": "Date Range",
        "select_date_range": "Select Date Range",
        "run_analysis": "Run Analysis",
        "selected_metrics": "Selected metrics based on prompt",
        "selected_dimensions": "Selected dimensions based on prompt",
        "analysis_results": "Analysis Results",
        "add_account": "Add Google Analytics Account",
        "property_id": "Property ID",
        "service_account_key": "Service Account Key (JSON)",
        "add": "Add",
        "openai_api_key": "OpenAI API Key",
        "save": "Save",
        "select_model": "Select OpenAI Model",
        "language": "Language",
        "english": "English",
        "polish": "Polski",
        "help_metrics": "Common GA4 Metrics and Dimensions",
        "no_reports": "No reports available. Run an analysis to see results here.",
        "loading": "Loading...",
        "error": "Error",
        "success": "Success",
        "product_comparison": "Product Comparison",
        "top_products": "Top Products by Composite Score",
        "custom_prompt_template": "Your Custom Prompt",
        "save_to_library": "Save to Library",
        "prompt_saved": "Prompt saved to library! (Note: This is a demo feature, prompts are not actually saved between sessions)",
        "last_7_days": "Last 7 days",
        "last_30_days": "Last 30 days",
        "last_90_days": "Last 90 days",
        "last_12_months": "Last 12 months",
        "custom_range": "Custom Range",
        "start_date": "Start Date",
        "end_date": "End Date",
        "use_suggested_prompt": "Use suggested prompt"
    },
    "pl": {
        "navigation": "Nawigacja",
        "dashboard": "Pulpit",
        "new_analysis": "Nowa Analiza",
        "report_history": "Historia Raportów",
        "prompt_library": "Biblioteka Promptów",
        "settings": "Ustawienia",
        "select_analysis_type": "Wybierz Typ Analizy",
        "analysis_prompt": "Prompt Analizy",
        "select_prompt": "Wybierz Prompt",
        "default_prompt": "Domyślny Prompt",
        "use_custom_prompt": "Użyj własnego promptu",
        "custom_prompt": "Własny Prompt",
        "report_details": "Szczegóły Raportu",
        "report_title": "Tytuł Raportu",
        "report_description": "Opis Raportu",
        "ga_account": "Konto Google Analytics",
        "select_account": "Wybierz Konto",
        "date_range": "Zakres Dat",
        "select_date_range": "Wybierz Zakres Dat",
        "run_analysis": "Uruchom Analizę",
        "selected_metrics": "Wybrane metryki na podstawie promptu",
        "selected_dimensions": "Wybrane wymiary na podstawie promptu",
        "analysis_results": "Wyniki Analizy",
        "add_account": "Dodaj Konto Google Analytics",
        "property_id": "ID Właściwości",
        "service_account_key": "Klucz Konta Usługi (JSON)",
        "add": "Dodaj",
        "openai_api_key": "Klucz API OpenAI",
        "save": "Zapisz",
        "select_model": "Wybierz Model OpenAI",
        "language": "Język",
        "english": "English",
        "polish": "Polski",
        "help_metrics": "Popularne Metryki i Wymiary GA4",
        "no_reports": "Brak dostępnych raportów. Uruchom analizę, aby zobaczyć wyniki tutaj.",
        "loading": "Ładowanie...",
        "error": "Błąd",
        "success": "Sukces",
        "product_comparison": "Porównanie Produktów",
        "top_products": "Najlepsze Produkty wg Złożonego Wyniku",
        "custom_prompt_template": "Twój Własny Prompt",
        "save_to_library": "Zapisz do Biblioteki",
        "prompt_saved": "Prompt zapisany do biblioteki! (Uwaga: To funkcja demonstracyjna, prompty nie są faktycznie zapisywane między sesjami)",
        "last_7_days": "Ostatnie 7 dni",
        "last_30_days": "Ostatnie 30 dni",
        "last_90_days": "Ostatnie 90 dni",
        "last_12_months": "Ostatnie 12 miesięcy",
        "custom_range": "Niestandardowy Zakres",
        "start_date": "Data Początkowa",
        "end_date": "Data Końcowa",
        "use_suggested_prompt": "Użyj sugerowanego promptu"
    }
}

# Function to get translated text
def get_text(key):
    lang = st.session_state.language
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))

# Navigation sidebar
st.sidebar.title(get_text("navigation"))
page = st.sidebar.radio("", [
    get_text("dashboard"), 
    get_text("new_analysis"), 
    get_text("report_history"), 
    get_text("prompt_library"), 
    get_text("settings")
])

# Function to add a GA account
def add_ga_account(property_id, service_account_key=None):
    try:
        # Initialize GA client with service account key
        ga_client = GoogleAnalyticsData(property_id, service_account_key)
        
        # Test connection
        test_data = ga_client.get_report(
            start_date="7daysAgo",
            end_date="yesterday",
            dimensions=["date"],
            metrics=["activeUsers"]
        )
        
        if test_data is not None and not test_data.empty:
            # Store account in session state
            account_name = f"GA4: {property_id}"
            st.session_state.ga_accounts[account_name] = {
                "property_id": property_id,
                "service_account_key": service_account_key
            }
            st.session_state.selected_account = account_name
            return True, "Account added successfully!"
        else:
            return False, "Failed to retrieve data from Google Analytics."
    except Exception as e:
        return False, f"Error adding account: {str(e)}"

# Function to format DataFrame for LLM to reduce token usage
def format_dataframe_for_llm(df):
    """
    Format DataFrame in a more token-efficient way for LLM processing.
    Uses CSV format which is more compact than string representation.
    """
    if df is None or df.empty:
        return "No data available."
    
    # Convert DataFrame to CSV string (more compact than df.to_string())
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()
    
    # Calculate approximate token count (rough estimate: 4 chars per token)
    approx_tokens = len(csv_str) // 4
    
    # Add token usage information
    result = f"Data (approx. {approx_tokens} tokens):\n\n{csv_str}"
    return result

# Function to detect language from prompt
def detect_language(prompt):
    """
    Detect language from prompt text.
    Currently supports English and Polish.
    """
    polish_words = ["analiza", "produkty", "koszyk", "zamówienie", "sprzedaż", 
                   "kategoria", "marka", "cena", "rabat", "promocja", "konwersja"]
    
    prompt_lower = prompt.lower()
    polish_word_count = sum(1 for word in polish_words if word in prompt_lower)
    
    # If more than 2 Polish words are found, assume Polish
    return "pl" if polish_word_count > 2 else "en"

# Function to run analysis
def run_analysis(account_name, start_date, end_date, analysis_type, custom_prompt=None, dimensions=None, metrics=None):
    if account_name not in st.session_state.ga_accounts:
        return None, "Account not found."
    
    account_info = st.session_state.ga_accounts[account_name]
    property_id = account_info["property_id"]
    service_account_key = account_info["service_account_key"]
    
    # Initialize GA client
    ga_client = GoogleAnalyticsData(property_id, service_account_key)
    
    # Get analysis configuration based on type
    analysis_config = get_analysis_config_for_type(analysis_type)
    
    # Detect language if custom prompt is provided
    language = "en"
    if custom_prompt:
        language = detect_language(custom_prompt)
        st.session_state.language = language
    
    # If custom prompt is provided, analyze it to determine metrics and dimensions
    if custom_prompt:
        # Check if this is a product-specific analysis with enhanced understanding
        product_analysis = suggest_product_metrics_and_dimensions(custom_prompt, language)
        
        if product_analysis and product_analysis["is_product_query"]:
            # Use enhanced product-specific metrics and dimensions
            analysis_config['metrics'] = product_analysis["suggested_metrics"]
            analysis_config['dimensions'] = product_analysis["suggested_dimensions"]
            analysis_config['analysis_type'] = product_analysis["analysis_type"]
            
            # Use the suggested prompt if available
            if "suggested_prompt" in product_analysis:
                custom_prompt = product_analysis["suggested_prompt"]
        else:
            # Use intelligent metric selection for non-product analysis
            selected_metrics = select_metrics_from_prompt(custom_prompt)
            if selected_metrics:
                analysis_config['metrics'] = selected_metrics
    
    # Override with custom dimensions and metrics if provided
    if dimensions:
        analysis_config['dimensions'] = dimensions
    if metrics:
        analysis_config['metrics'] = metrics
    
    # Ensure metrics and dimensions are compatible
    safe_metrics, safe_dimensions = get_safe_metrics_and_dimensions(
        analysis_config['metrics'], 
        analysis_config['dimensions']
    )
    
    analysis_config['metrics'] = safe_metrics
    analysis_config['dimensions'] = safe_dimensions
    
    try:
        # Get data from GA
        df = ga_client.get_report(
            start_date=start_date,
            end_date=end_date,
            dimensions=analysis_config['dimensions'],
            metrics=analysis_config['metrics']
        )
        
        if df is None or df.empty:
            return None, "No data returned from Google Analytics."
        
        # Format data for LLM
        df_str = format_dataframe_for_llm(df)
        
        # Create prompt for LLM
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = analysis_config['prompt']
        
        # Add context about the analysis
        context = f"""
        Analysis period: {start_date} to {end_date}
        Dimensions: {', '.join(analysis_config['dimensions'])}
        Metrics: {', '.join(analysis_config['metrics'])}
        """
        
        full_prompt = f"{prompt}\n\n{context}\n\n{df_str}"
        
        # Get analysis from LLM
        api_key = st.session_state.openai_api_key
        model = st.session_state.selected_model
        # Auto-determine appropriate max tokens based on model and input size
        input_token_estimate = len(full_prompt) // 4  # Rough estimate
        model_max_tokens = OPENAI_MODELS[model]["max_context"]
        # Use 25% of remaining tokens after input (or 1000 if that's larger)
        max_tokens = max(1000, int((model_max_tokens - input_token_estimate) * 0.25))
        
        analysis_result = analyze_data_with_llm(full_prompt, api_key, model, max_tokens)
        
        # Save report to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = {
            "timestamp": timestamp,
            "account": account_name,
            "start_date": start_date,
            "end_date": end_date,
            "analysis_type": analysis_type,
            "custom_prompt": custom_prompt,
            "dimensions": analysis_config['dimensions'],
            "metrics": analysis_config['metrics'],
            "data": df.to_dict(),
            "result": analysis_result
        }
        st.session_state.reports.append(report)
        
        return analysis_result, None
    except Exception as e:
        return None, f"Error running analysis: {str(e)}"

# Function to get analysis configuration based on type
def get_analysis_config_for_type(analysis_type):
    configs = {
        "general_overview": {
            "dimensions": ["date", "deviceCategory", "country"],
            "metrics": ["activeUsers", "newUsers", "sessions", "averageSessionDuration"],
            "prompt": "Provide a general overview of the website performance based on the provided Google Analytics data. Include trends over time, device breakdown, and geographic insights."
        },
        "traffic_sources": {
            "dimensions": ["sessionSource", "sessionMedium", "sessionCampaignName"],
            "metrics": ["sessions", "engagementRate", "conversions"],
            "prompt": "Analyze the traffic sources based on the provided Google Analytics data. Identify the top performing sources, mediums, and campaigns."
        },
        "content_performance": {
            "dimensions": ["pageTitle", "pagePathPlusQueryString"],
            "metrics": ["screenPageViews", "averageSessionDuration", "bounceRate"],
            "prompt": "Evaluate the content performance based on the provided Google Analytics data. Identify the top performing pages and content."
        },
        "conversion_analysis": {
            "dimensions": ["date", "deviceCategory"],
            "metrics": ["conversions", "eventCount", "eventValue"],
            "prompt": "Analyze the conversion performance based on the provided Google Analytics data. Identify trends, patterns, and opportunities for improvement."
        },
        "user_behavior": {
            "dimensions": ["sessionSource", "deviceCategory"],
            "metrics": ["activeUsers", "engagementRate", "averageSessionDuration"],
            "prompt": "Analyze user behavior based on the provided Google Analytics data. Identify how users interact with the website across different sources and devices."
        },
        "product_performance": {
            "dimensions": ["itemName", "itemCategory"],
            "metrics": ["itemsViewed", "itemsAddedToCart", "itemsPurchased"],
            "prompt": "Analyze product performance based on the provided Google Analytics data. Identify the top performing products, categories, and trends."
        },
        "cart_analysis": {
            "dimensions": ["itemName", "itemCategory"],
            "metrics": ["itemsAddedToCart", "itemsCheckedOut", "itemsPurchased"],
            "prompt": "Analyze shopping cart behavior based on the provided Google Analytics data. Identify products frequently added to cart, cart abandonment patterns, and conversion rates from cart to purchase."
        },
        "checkout_analysis": {
            "dimensions": ["date", "deviceCategory"],
            "metrics": ["checkouts", "purchases", "totalRevenue"],
            "prompt": "Analyze checkout process based on the provided Google Analytics data. Identify conversion rates, abandonment points, and opportunities to optimize the checkout flow."
        },
        "product_list_performance": {
            "dimensions": ["itemListName", "itemListId", "itemName"],
            "metrics": ["itemsViewedInList", "itemsClickedInList", "itemsAddedToCart"],
            "prompt": "Analyze product list performance based on the provided Google Analytics data. Identify which product lists generate the most engagement and conversions."
        },
        "promotion_effectiveness": {
            "dimensions": ["itemPromotionName", "itemPromotionId", "itemName"],
            "metrics": ["itemsViewedInPromotion", "itemsClickedInPromotion", "itemsPurchased"],
            "prompt": "Analyze promotion effectiveness based on the provided Google Analytics data. Identify which promotions generate the most engagement and conversions."
        }
    }
    
    return configs.get(analysis_type, configs["general_overview"])

# Function to select metrics based on prompt
def select_metrics_from_prompt(prompt):
    # Simple keyword-based metric selection
    prompt_lower = prompt.lower()
    
    selected_metrics = []
    
    # Traffic and user metrics
    if any(word in prompt_lower for word in ["traffic", "user", "visitor", "audience", "ruch", "użytkownik", "odwiedzający"]):
        selected_metrics.extend(["activeUsers", "newUsers", "sessions"])
    
    # Engagement metrics
    if any(word in prompt_lower for word in ["engagement", "time", "duration", "bounce", "zaangażowanie", "czas", "odbicie"]):
        selected_metrics.extend(["engagementRate", "averageSessionDuration", "bounceRate"])
    
    # Content metrics
    if any(word in prompt_lower for word in ["content", "page", "view", "treść", "strona", "wyświetlenie"]):
        selected_metrics.extend(["screenPageViews", "averageSessionDuration"])
    
    # Conversion metrics
    if any(word in prompt_lower for word in ["conversion", "goal", "event", "konwersja", "cel", "zdarzenie"]):
        selected_metrics.extend(["conversions", "eventCount", "eventValue"])
    
    # If no metrics were selected, use default metrics
    if not selected_metrics:
        return None
    
    # Remove duplicates
    selected_metrics = list(set(selected_metrics))
    
    return selected_metrics

# Function to compare products based on selected metrics
def compare_products(df, metrics, top_n=10):
    """
    Compare products based on selected metrics and return top N products.
    
    Args:
        df: DataFrame with product data
        metrics: List of metrics to use for comparison
        top_n: Number of top products to return
        
    Returns:
        DataFrame with top N products
    """
    if df is None or df.empty or 'itemName' not in df.columns:
        return None
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Ensure all metrics are numeric
    for metric in metrics:
        if metric in df_copy.columns:
            df_copy[metric] = pd.to_numeric(df_copy[metric], errors='coerce')
    
    # Group by product name and aggregate metrics
    product_df = df_copy.groupby('itemName').agg({metric: 'sum' for metric in metrics if metric in df_copy.columns})
    
    # Calculate a composite score (simple sum of normalized metrics)
    for metric in metrics:
        if metric in product_df.columns:
            # Normalize the metric (0-1 scale)
            max_val = product_df[metric].max()
            min_val = product_df[metric].min()
            if max_val > min_val:
                product_df[f'{metric}_normalized'] = (product_df[metric] - min_val) / (max_val - min_val)
            else:
                product_df[f'{metric}_normalized'] = 0
    
    # Calculate composite score
    normalized_metrics = [f'{metric}_normalized' for metric in metrics if f'{metric}_normalized' in product_df.columns]
    if normalized_metrics:
        product_df['composite_score'] = product_df[normalized_metrics].mean(axis=1)
        
        # Sort by composite score
        product_df = product_df.sort_values('composite_score', ascending=False)
    
    # Return top N products
    return product_df.head(top_n)

# Dashboard page
if page == get_text("dashboard"):
    st.title("GA Analytics AI " + get_text("dashboard"))
    
    if not st.session_state.ga_accounts:
        st.info(get_text("no_reports").split('.')[0] + ". " + get_text("settings") + ".")
    else:
        st.subheader(get_text("quick_analysis"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            account_name = st.selectbox(get_text("select_account"), list(st.session_state.ga_accounts.keys()))
            analysis_type = st.selectbox(get_text("select_analysis_type"), [
                "general_overview", "traffic_sources", "content_performance", 
                "conversion_analysis", "user_behavior", "product_performance",
                "cart_analysis", "checkout_analysis", "product_list_performance",
                "promotion_effectiveness"
            ])
        
        with col2:
            date_range = st.selectbox(get_text("select_date_range"), [
                get_text("last_7_days"), get_text("last_30_days"), get_text("last_90_days"), get_text("last_12_months")
            ])
            
            # Convert date range to start and end dates
            end_date = "yesterday"
            if date_range == get_text("last_7_days"):
                start_date = "7daysAgo"
            elif date_range == get_text("last_30_days"):
                start_date = "30daysAgo"
            elif date_range == get_text("last_90_days"):
                start_date = "90daysAgo"
            else:
                start_date = "365daysAgo"
        
        if st.button(get_text("run_analysis")):
            with st.spinner(get_text("loading")):
                result, error = run_analysis(account_name, start_date, end_date, analysis_type)
                
                if error:
                    st.error(error)
                else:
                    st.subheader(get_text("analysis_results"))
                    st.markdown(result)
        
        # Recent reports
        st.subheader(get_text("report_history"))
        if st.session_state.reports:
            for i, report in enumerate(reversed(st.session_state.reports[-5:])):
                with st.expander(f"{report['timestamp']} - {report['analysis_type']}"):
                    st.markdown(report['result'])
        else:
            st.info(get_text("no_reports"))

# New Analysis page
elif page == get_text("new_analysis"):
    st.title(get_text("new_analysis"))
    
    if not st.session_state.ga_accounts:
        st.info(get_text("no_reports").split('.')[0] + ". " + get_text("settings") + ".")
    else:
        # Analysis type selection
        st.subheader(get_text("select_analysis_type"))
        analysis_type = st.selectbox(get_text("select_analysis_type"), [
            "general_overview", "traffic_sources", "content_performance", 
            "conversion_analysis", "user_behavior", "product_performance",
            "cart_analysis", "checkout_analysis", "product_list_performance",
            "promotion_effectiveness"
        ])
        
        # Prompt selection
        st.subheader(get_text("analysis_prompt"))
        prompt_type = st.selectbox(get_text("select_prompt"), [get_text("default_prompt"), get_text("custom_prompt")])
        
        custom_prompt = None
        if prompt_type == get_text("custom_prompt"):
            use_custom_prompt = st.checkbox(get_text("use_custom_prompt"))
            if use_custom_prompt:
                custom_prompt = st.text_area(get_text("custom_prompt"), height=150)
                
                # Show product analysis detection if prompt is entered
                if custom_prompt:
                    language = detect_language(custom_prompt)
                    is_product_query, confidence = detect_product_analysis_intent_enhanced(custom_prompt, language)
                    
                    if is_product_query:
                        st.info(f"Product analysis detected with {confidence:.2f} confidence.")
                        
                        # Get product analysis details
                        product_analysis = suggest_product_metrics_and_dimensions(custom_prompt, language)
                        
                        if product_analysis:
                            st.success(f"Analysis type: {product_analysis['analysis_type']}")
                            
                            # Show extracted entities if any
                            if product_analysis['extracted_entities']:
                                st.write("Extracted product entities:")
                                for entity_type, entities in product_analysis['extracted_entities'].items():
                                    st.write(f"- {entity_type}: {', '.join(entities)}")
                            
                            # Show suggested prompt
                            if "suggested_prompt" in product_analysis:
                                st.write("Suggested prompt:")
                                st.info(product_analysis["suggested_prompt"])
                                
                                # Option to use suggested prompt
                                if st.button(get_text("use_suggested_prompt")):
                                    custom_prompt = product_analysis["suggested_prompt"]
        
        # Report details
        st.subheader(get_text("report_details"))
        report_title = st.text_input(get_text("report_title"), f"{analysis_type.replace('_', ' ').title()} - {datetime.now().strftime('%Y-%m-%d')}")
        report_description = st.text_area(get_text("report_description"), f"Analysis of {analysis_type.replace('_', ' ')} data from Google Analytics.")
        
        # Account selection
        st.subheader(get_text("ga_account"))
        account_name = st.selectbox(get_text("select_account"), list(st.session_state.ga_accounts.keys()))
        
        # Date range selection
        st.subheader(get_text("date_range"))
        date_range = st.selectbox(get_text("select_date_range"), [
            get_text("last_7_days"), 
            get_text("last_30_days"), 
            get_text("last_90_days"), 
            get_text("last_12_months"), 
            get_text("custom_range")
        ])
        
        if date_range == "Custom Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date_custom = st.date_input("Start Date", datetime.now() - timedelta(days=7))
                start_date = start_date_custom.strftime("%Y-%m-%d")
            with col2:
                end_date_custom = st.date_input("End Date", datetime.now() - timedelta(days=1))
                end_date = end_date_custom.strftime("%Y-%m-%d")
        else:
            end_date = "yesterday"
            if date_range == "Last 7 days":
                start_date = "7daysAgo"
            elif date_range == "Last 30 days":
                start_date = "30daysAgo"
            elif date_range == "Last 90 days":
                start_date = "90daysAgo"
            else:
                start_date = "365daysAgo"
        
        # Run analysis button
        if st.button("Run Analysis"):
            with st.spinner("Running analysis..."):
                # Get analysis configuration
                analysis_config = get_analysis_config_for_type(analysis_type)
                
                # Detect language if custom prompt is provided
                language = "en"
                if custom_prompt and use_custom:
                    language = detect_language(custom_prompt)
                
                # Display selected metrics and dimensions
                if custom_prompt and use_custom:
                    # Check if this is a product-specific analysis with enhanced understanding
                    product_analysis = suggest_product_metrics_and_dimensions(custom_prompt, language)
                    
                    if product_analysis and product_analysis["is_product_query"]:
                        metrics = product_analysis["suggested_metrics"]
                        dimensions = product_analysis["suggested_dimensions"]
                        
                        # Use the suggested prompt if available
                        if "suggested_prompt" in product_analysis:
                            custom_prompt = product_analysis["suggested_prompt"]
                    else:
                        # Use intelligent metric selection for non-product analysis
                        selected_metrics = select_metrics_from_prompt(custom_prompt)
                        if selected_metrics:
                            metrics = selected_metrics
                        else:
                            metrics = analysis_config['metrics']
                        dimensions = analysis_config['dimensions']
                else:
                    metrics = analysis_config['metrics']
                    dimensions = analysis_config['dimensions']
                
                # Ensure metrics and dimensions are compatible
                metrics, dimensions = get_safe_metrics_and_dimensions(metrics, dimensions)
                
                st.info(f"Selected metrics based on prompt: {', '.join(metrics)}")
                st.info(f"Selected dimensions based on prompt: {', '.join(dimensions)}")
                
                # Run the analysis
                result, error = run_analysis(
                    account_name, 
                    start_date, 
                    end_date, 
                    analysis_type,
                    custom_prompt if use_custom_prompt else None,
                    dimensions,
                    metrics
                )
                
                if error:
                    st.error(error)
                    if "incompatible" in error.lower():
                        st.warning("Please try a different combination of metrics and dimensions or a different prompt.")
                else:
                    st.subheader("Analysis Results")
                    st.markdown(result)
                    
                    # If this is a product analysis, show product comparison
                    if product_analysis and product_analysis["is_product_query"] and "itemName" in dimensions:
                        st.subheader("Product Comparison")
                        
                        # Get the data from the most recent report
                        if st.session_state.reports:
                            latest_report = st.session_state.reports[-1]
                            df = pd.DataFrame.from_dict(latest_report["data"])
                            
                            # Compare products
                            comparison_metrics = [m for m in metrics if m in df.columns]
                            if comparison_metrics:
                                top_products = compare_products(df, comparison_metrics)
                                
                                if top_products is not None and not top_products.empty:
                                    st.write("Top Products by Composite Score:")
                                    st.dataframe(top_products)
                                    
                                    # Create a bar chart of top products
                                    if 'composite_score' in top_products.columns:
                                        fig = px.bar(
                                            top_products.reset_index(), 
                                            x='itemName', 
                                            y='composite_score',
                                            title="Top Products by Composite Score",
                                            labels={'itemName': 'Product', 'composite_score': 'Score'}
                                        )
                                        st.plotly_chart(fig)

# Report History page
elif page == "Report History":
    st.title("Report History")
    
    if not st.session_state.reports:
        st.info("No reports yet. Run an analysis to see reports here.")
    else:
        for i, report in enumerate(reversed(st.session_state.reports)):
            with st.expander(f"{report['timestamp']} - {report['analysis_type']}"):
                st.markdown(f"**Account:** {report['account']}")
                st.markdown(f"**Date Range:** {report['start_date']} to {report['end_date']}")
                st.markdown(f"**Analysis Type:** {report['analysis_type']}")
                if report.get('custom_prompt'):
                    st.markdown(f"**Custom Prompt:** {report['custom_prompt']}")
                st.markdown(f"**Dimensions:** {', '.join(report['dimensions'])}")
                st.markdown(f"**Metrics:** {', '.join(report['metrics'])}")
                
                # Convert data dict back to DataFrame for display
                df = pd.DataFrame.from_dict(report['data'])
                st.dataframe(df)
                
                st.markdown("### Analysis Result")
                st.markdown(report['result'])
                
                # If this is a product analysis, offer product comparison
                if "itemName" in report['dimensions'] and any(metric in ITEM_SCOPED_METRICS for metric in report['metrics']):
                    if st.button(f"Compare Products (Report {i})", key=f"compare_{i}"):
                        comparison_metrics = [m for m in report['metrics'] if m in df.columns]
                        if comparison_metrics:
                            top_products = compare_products(df, comparison_metrics)
                            
                            if top_products is not None and not top_products.empty:
                                st.write("Top Products by Composite Score:")
                                st.dataframe(top_products)
                                
                                # Create a bar chart of top products
                                if 'composite_score' in top_products.columns:
                                    fig = px.bar(
                                        top_products.reset_index(), 
                                        x='itemName', 
                                        y='composite_score',
                                        title="Top Products by Composite Score",
                                        labels={'itemName': 'Product', 'composite_score': 'Score'}
                                    )
                                    st.plotly_chart(fig)

# Prompt Library page
elif page == "Prompt Library":
    st.title("Prompt Library")
    
    st.markdown("""
    This page contains a library of prompts that you can use for your analyses. 
    Copy a prompt and use it in the Custom Prompt field on the New Analysis page.
    """)
    
    # General prompts
    st.subheader("General Analysis Prompts")
    
    with st.expander("Website Performance Overview"):
        st.markdown("""
        Provide a comprehensive overview of the website performance based on the Google Analytics data. 
        Include trends in user engagement, traffic sources, and device usage. 
        Identify any significant patterns or anomalies in the data and provide actionable insights.
        """)
    
    with st.expander("Traffic Source Analysis"):
        st.markdown("""
        Analyze the traffic sources based on the provided Google Analytics data. 
        Identify the top performing sources, mediums, and campaigns. 
        Compare the performance of different traffic sources in terms of engagement and conversions. 
        Provide recommendations for optimizing traffic acquisition strategy.
        """)
    
    # E-commerce prompts
    st.subheader("E-commerce Analysis Prompts")
    
    with st.expander("Product Performance Analysis"):
        st.markdown("""
        Analyze the performance of products based on the provided Google Analytics data.
        Identify the top performing products in terms of views, add-to-carts, and purchases.
        Determine which product categories generate the most revenue.
        Provide insights on product performance trends and recommendations for inventory and marketing decisions.
        """)
    
    with st.expander("Shopping Cart Analysis"):
        st.markdown("""
        Analyze shopping cart behavior based on the provided Google Analytics data.
        Identify products frequently added to cart but not purchased.
        Determine the cart abandonment rate and patterns.
        Provide recommendations for reducing cart abandonment and improving checkout conversion.
        """)
    
    with st.expander("Checkout Process Analysis"):
        st.markdown("""
        Analyze the checkout process based on the provided Google Analytics data.
        Identify any bottlenecks or drop-off points in the checkout funnel.
        Compare checkout completion rates across different devices and traffic sources.
        Provide recommendations for optimizing the checkout process and increasing conversion rates.
        """)
    
    with st.expander("Product Category Comparison"):
        st.markdown("""
        Compare the performance of different product categories based on the provided Google Analytics data.
        Analyze views, add-to-carts, and purchases for each category.
        Identify which categories have the highest conversion rates and generate the most revenue.
        Provide insights on category performance trends and recommendations for category management.
        """)
    
    with st.expander("Product Ranking Analysis"):
        st.markdown("""
        Analyze the performance of all products and create a ranking based on the following criteria:
        1. Number of conversions (40% weight)
        2. Number of events (30% weight)
        3. Event value (30% weight)
        
        For each product in the top 10, provide:
        1. Key strengths (what's working well)
        2. Areas for improvement
        3. Specific recommendations to boost performance
        
        Present the results as a ranking with justification for each position and recommendations.
        """)
    
    with st.expander("Cart to Purchase Conversion Analysis"):
        st.markdown("""
        Analyze which products were most frequently added to the cart and which had the highest conversion rate from cart to sale.
        Identify products with high add-to-cart rates but low purchase rates.
        Determine factors that might be causing cart abandonment for specific products.
        Provide recommendations for improving cart-to-purchase conversion rates.
        """)
    
    # Custom prompt creation
    st.subheader("Create Your Own Prompt")
    
    st.markdown("""
    Use the template below to create your own custom prompt:
    
    ```
    Analyze [specific aspect] based on the provided Google Analytics data.
    Focus on [specific metrics or dimensions].
    Identify [specific patterns or insights you're looking for].
    Provide recommendations for [specific goal or improvement area].
    ```
    """)
    
    custom_prompt_template = st.text_area("Your Custom Prompt", height=150)
    
    if st.button("Save to Library"):
        st.success("Prompt saved to library! (Note: This is a demo feature, prompts are not actually saved between sessions)")

# Settings page
elif page == "Settings":
    st.title("Settings")
    
    # OpenAI API key
    st.subheader("OpenAI API Key")
    openai_api_key = st.text_input("Enter your OpenAI API key", value=st.session_state.openai_api_key, type="password")
    if st.button("Save API Key"):
        st.session_state.openai_api_key = openai_api_key
        st.success("API key saved!")
    
    # Model selection
    st.subheader("LLM Model Settings")
    
    # Model selection
    model_options = list(OPENAI_MODELS.keys())
    selected_model = st.selectbox(
        "Select OpenAI Model", 
        options=model_options,
        index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
        format_func=lambda x: f"{x} - {OPENAI_MODELS[x]['description']}"
    )
    

    if st.button("Save Model Settings"):
        st.session_state.selected_model = selected_model
        st.success("Model settings saved!")
    
    # Language settings
    st.subheader("Language Settings")
    language = st.selectbox(
        "Default Language", 
        options=["en", "pl"],
        index=0 if st.session_state.language == "en" else 1,
        format_func=lambda x: "English" if x == "en" else "Polish"
    )
    
    if st.button("Save Language Settings"):
        st.session_state.language = language
        st.success("Language settings saved!")
    
    # Google Analytics account
    st.subheader("Google Analytics Account")
    
    # Display existing accounts
    if st.session_state.ga_accounts:
        st.markdown("**Current Accounts:**")
        for account_name, account_info in st.session_state.ga_accounts.items():
            st.markdown(f"- {account_name} (Property ID: {account_info['property_id']})")
    
    # Add new account
    st.markdown("### Add New Account")
    
    property_id = st.text_input("Google Analytics 4 Property ID (format: 123456789)")
    
    # Service account key upload
    st.markdown("### Service Account Authentication")
    st.markdown("Upload your service account key JSON file:")
    
    uploaded_file = st.file_uploader("Choose a service account key file", type=["json"])
    
    if uploaded_file is not None:
        try:
            service_account_key = json.load(uploaded_file)
            st.success("Service account key loaded successfully!")
        except Exception as e:
            st.error(f"Error loading service account key: {str(e)}")
            service_account_key = None
    else:
        service_account_key = st.session_state.service_account_key
    
    if st.button("Add Account"):
        if not property_id:
            st.error("Please enter a Google Analytics 4 Property ID.")
        elif not service_account_key:
            st.error("Please upload a service account key file.")
        else:
            with st.spinner("Adding account..."):
                success, message = add_ga_account(property_id, service_account_key)
                if success:
                    st.session_state.service_account_key = service_account_key
                    st.success(message)
                else:
                    st.error(message)
    
    # Help section
    st.subheader("GA4 Metrics and Dimensions Help")
    
    with st.expander("Common GA4 Metrics"):
        st.markdown("""
        ### User Metrics
        - `activeUsers`: Number of distinct users who visited your site or app
        - `newUsers`: Number of first-time users
        - `totalUsers`: Total number of users
        
        ### Session Metrics
        - `sessions`: Number of sessions
        - `averageSessionDuration`: Average duration of sessions
        - `engagementRate`: Percentage of engaged sessions
        
        ### Page/Screen Metrics
        - `screenPageViews`: Number of app screens or web pages viewed
        - `screenPageViewsPerSession`: Average number of pages viewed per session
        - `bounceRate`: Percentage of sessions with no engagement
        
        ### Event Metrics
        - `eventCount`: Total number of events
        - `eventCountPerUser`: Average number of events per user
        - `conversions`: Total number of conversions
        
        ### E-commerce Metrics
        - `itemsViewed`: Number of items viewed
        - `itemsAddedToCart`: Number of items added to cart
        - `itemsPurchased`: Number of items purchased
        - `itemRevenue`: Revenue from items
        """)
    
    with st.expander("Common GA4 Dimensions"):
        st.markdown("""
        ### User Dimensions
        - `deviceCategory`: Device category (mobile, tablet, desktop)
        - `browser`: Browser used
        - `operatingSystem`: Operating system used
        
        ### Geographic Dimensions
        - `country`: User's country
        - `region`: User's region
        - `city`: User's city
        
        ### Traffic Source Dimensions
        - `sessionSource`: Source of the session
        - `sessionMedium`: Medium of the session
        - `sessionCampaignName`: Campaign name
        
        ### Content Dimensions
        - `pageTitle`: Title of the page
        - `pagePathPlusQueryString`: Page path with query parameters
        - `landingPage`: First page in a session
        
        ### E-commerce Dimensions
        - `itemName`: Name of the item
        - `itemId`: ID of the item
        - `itemCategory`: Category of the item
        - `itemBrand`: Brand of the item
        - `transactionId`: ID of the transaction
        """)
    
    with st.expander("E-commerce Event Types"):
        st.markdown("""
        ### E-commerce Events in GA4
        - `view_item_list`: User views a list of items
        - `select_item`: User selects an item from a list
        - `view_item`: User views item details
        - `add_to_cart`: User adds item to cart
        - `remove_from_cart`: User removes item from cart
        - `view_cart`: User views their cart
        - `begin_checkout`: User begins checkout process
        - `add_shipping_info`: User adds shipping information
        - `add_payment_info`: User adds payment information
        - `purchase`: User completes purchase
        - `refund`: Purchase is refunded
        """)
    
    with st.expander("E-commerce Metrics and Dimensions Compatibility"):
        st.markdown("""
        ### Important Compatibility Notes
        
        GA4 has two types of e-commerce metrics:
        
        1. **Event-scoped metrics** count the number of times an e-commerce event was triggered.
           Examples: `addToCarts`, `checkouts`, `purchases`
        
        2. **Item-scoped metrics** count the number of times users interacted with items in an e-commerce event.
           Examples: `itemsAddedToCart`, `itemsCheckedOut`, `itemsPurchased`
        
        **Compatibility Rules:**
        
        - Item-scoped dimensions (like `itemName`, `itemId`) must be used with item-scoped metrics
        - Event-scoped dimensions (like `transactionId`) must be used with event-scoped metrics
        - Mixing item-scoped and event-scoped elements will cause errors
        
        For example:
        - ✅ `itemName` with `itemsAddedToCart` (both item-scoped)
        - ✅ `transactionId` with `purchases` (both event-scoped)
        - ❌ `itemName` with `addToCarts` (mixing item-scoped dimension with event-scoped metric)
        
        The application will automatically ensure compatibility between selected metrics and dimensions.
        """)
