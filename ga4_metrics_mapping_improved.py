"""
GA4 Metrics Mapping Module

This module provides mappings between analysis types and appropriate GA4 metrics,
as well as functions for intelligent metric selection based on prompts and
compatibility checking for GA4 metrics and dimensions.
"""

# Dictionary mapping analysis types to appropriate GA4 metrics and dimensions
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
    },
    "E-commerce Analysis": {
        "dimensions": ["date"],  # Removed itemName to avoid compatibility issues
        "metrics": ["addToCarts", "checkouts", "ecommercePurchases"]  # Removed grossItemRevenue
    },
    "User Engagement": {
        "dimensions": ["date"],
        "metrics": ["engagedSessions", "engagementRate", "averageSessionDuration", "screenPageViewsPerSession"]
    },
    "Product Performance": {
        "dimensions": ["date"],
        "metrics": ["conversions", "eventCount", "eventValue"]
    }
}

# Dictionary mapping common prompt keywords to analysis types
PROMPT_KEYWORD_MAPPING = {
    # Audience related keywords
    "użytkownicy": "Audience Analysis",
    "users": "Audience Analysis",
    "audience": "Audience Analysis",
    "demographics": "Audience Analysis",
    "geografia": "Audience Analysis",
    "geography": "Audience Analysis",
    "urządzenia": "Audience Analysis",
    "devices": "Audience Analysis",
    
    # Acquisition related keywords
    "źródła": "Acquisition Analysis",
    "sources": "Acquisition Analysis",
    "acquisition": "Acquisition Analysis",
    "traffic": "Acquisition Analysis",
    "ruch": "Acquisition Analysis",
    "kampanie": "Acquisition Analysis",
    "campaigns": "Acquisition Analysis",
    "kanały": "Acquisition Analysis",
    "channels": "Acquisition Analysis",
    
    # Behavior related keywords
    "zachowanie": "Behavior Analysis",
    "behavior": "Behavior Analysis",
    "strony": "Behavior Analysis",
    "pages": "Behavior Analysis",
    "content": "Behavior Analysis",
    "treść": "Behavior Analysis",
    "engagement": "User Engagement",
    "zaangażowanie": "User Engagement",
    
    # Conversion related keywords
    "konwersje": "Conversion Analysis",
    "conversions": "Conversion Analysis",
    "goals": "Conversion Analysis",
    "cele": "Conversion Analysis",
    "events": "Conversion Analysis",
    "zdarzenia": "Conversion Analysis",
    
    # E-commerce related keywords
    "ecommerce": "Conversion Analysis",  # Changed to use safer metrics
    "e-commerce": "Conversion Analysis",  # Changed to use safer metrics
    "produkty": "Product Performance",
    "products": "Product Performance",
    "sprzedaż": "Conversion Analysis",
    "sales": "Conversion Analysis",
    "revenue": "Conversion Analysis",
    "przychód": "Conversion Analysis",
    "koszyk": "Conversion Analysis",
    "cart": "Conversion Analysis",
    "zakupy": "Conversion Analysis",
    "purchases": "Conversion Analysis",
    "ranking": "Product Performance",
    "efektywność": "Product Performance",
    "performance": "Product Performance"
}

# Dictionary of valid GA4 metrics with descriptions
VALID_GA4_METRICS = {
    "activeUsers": "The number of distinct users who visited your site or app.",
    "newUsers": "The number of users who visited your site or app for the first time.",
    "totalUsers": "The total number of users who visited your site or app.",
    "sessions": "The number of sessions that began on your site or app.",
    "averageSessionDuration": "The average duration (in seconds) of users' sessions.",
    "screenPageViews": "The number of app screens or web pages your users viewed.",
    "screenPageViewsPerSession": "The average number of pages viewed during a session.",
    "engagedSessions": "The number of sessions that lasted longer than 10 seconds, or had a conversion event, or had 2 or more screen or page views.",
    "engagementRate": "The percentage of engaged sessions (Engaged sessions divided by Sessions).",
    "eventCount": "The count of events.",
    "eventCountPerUser": "The average number of events per user (Event count divided by Active users).",
    "eventValue": "The sum of the event parameter named value.",
    "conversions": "The number of conversions.",
    "addToCarts": "The number of times users added items to their shopping carts.",
    "checkouts": "The number of times users started the checkout process.",
    "ecommercePurchases": "The number of times users completed a purchase.",
    "bounceRate": "The percentage of sessions that were not engaged (Sessions Minus Engaged sessions) divided by Sessions)."
}

# Dictionary of valid GA4 dimensions
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

# Known incompatible combinations of metrics and dimensions
INCOMPATIBLE_COMBINATIONS = [
    {"dimension": "itemName", "metric": "grossItemRevenue"},
    {"dimension": "itemId", "metric": "grossItemRevenue"},
    {"dimension": "itemCategory", "metric": "grossItemRevenue"}
]

# Safe fallback configuration if incompatible combinations are detected
SAFE_FALLBACK_CONFIG = {
    "dimensions": ["date"],
    "metrics": ["activeUsers", "sessions", "screenPageViews", "engagementRate"]
}

def is_compatible_combination(dimensions, metrics):
    """
    Check if the combination of dimensions and metrics is compatible.
    
    Args:
        dimensions: List of dimensions
        metrics: List of metrics
        
    Returns:
        Boolean indicating if the combination is compatible
    """
    # Check for known incompatible combinations
    for incompatible in INCOMPATIBLE_COMBINATIONS:
        if incompatible["dimension"] in dimensions and incompatible["metric"] in metrics:
            return False
    
    # Check if all dimensions are valid
    for dimension in dimensions:
        if dimension not in VALID_GA4_DIMENSIONS and dimension != "itemName":
            return False
    
    # Check if all metrics are valid
    for metric in metrics:
        if metric not in VALID_GA4_METRICS and metric != "grossItemRevenue":
            return False
    
    return True

def get_safe_metrics_and_dimensions(dimensions, metrics):
    """
    Get a safe combination of metrics and dimensions by removing incompatible ones.
    
    Args:
        dimensions: List of dimensions
        metrics: List of metrics
        
    Returns:
        Dictionary with safe dimensions and metrics
    """
    safe_dimensions = []
    safe_metrics = []
    
    # Filter out dimensions that are not in the valid list
    for dimension in dimensions:
        if dimension in VALID_GA4_DIMENSIONS:
            safe_dimensions.append(dimension)
    
    # Filter out metrics that are not in the valid list
    for metric in metrics:
        if metric in VALID_GA4_METRICS:
            safe_metrics.append(metric)
    
    # Check if we have any dimensions and metrics left
    if not safe_dimensions:
        safe_dimensions = ["date"]
    
    if not safe_metrics:
        safe_metrics = ["activeUsers", "sessions"]
    
    # Check for incompatible combinations and remove them
    for incompatible in INCOMPATIBLE_COMBINATIONS:
        if incompatible["dimension"] in safe_dimensions and incompatible["metric"] in safe_metrics:
            if incompatible["dimension"] in safe_dimensions:
                safe_dimensions.remove(incompatible["dimension"])
            if incompatible["metric"] in safe_metrics:
                safe_metrics.remove(incompatible["metric"])
    
    # If we've removed too much, use the safe fallback
    if not safe_dimensions or not safe_metrics:
        return SAFE_FALLBACK_CONFIG
    
    return {
        "dimensions": safe_dimensions,
        "metrics": safe_metrics
    }

def select_metrics_from_prompt(prompt: str) -> dict:
    """
    Intelligently select metrics and dimensions based on the prompt.
    
    Args:
        prompt: The analysis prompt
        
    Returns:
        Dictionary with selected dimensions and metrics
    """
    prompt_lower = prompt.lower()
    
    # Default to General Overview
    selected_analysis_type = "General Overview"
    
    # Check for keywords in the prompt
    for keyword, analysis_type in PROMPT_KEYWORD_MAPPING.items():
        if keyword.lower() in prompt_lower:
            selected_analysis_type = analysis_type
            break
    
    # Get the metrics and dimensions for the selected analysis type
    config = GA4_METRICS_MAPPING[selected_analysis_type].copy()
    
    # Check if the combination is compatible
    if not is_compatible_combination(config["dimensions"], config["metrics"]):
        # If not compatible, get a safe combination
        config = get_safe_metrics_and_dimensions(config["dimensions"], config["metrics"])
    
    return config

def get_analysis_config_for_type(analysis_type: str) -> dict:
    """
    Get the metrics and dimensions configuration for a specific analysis type.
    
    Args:
        analysis_type: The type of analysis
        
    Returns:
        Dictionary with dimensions and metrics for the analysis type
    """
    if analysis_type not in GA4_METRICS_MAPPING:
        return SAFE_FALLBACK_CONFIG
    
    config = GA4_METRICS_MAPPING[analysis_type].copy()
    
    # Check if the combination is compatible
    if not is_compatible_combination(config["dimensions"], config["metrics"]):
        # If not compatible, get a safe combination
        config = get_safe_metrics_and_dimensions(config["dimensions"], config["metrics"])
    
    return config
