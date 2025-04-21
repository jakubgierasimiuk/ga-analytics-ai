"""
GA4 Metrics Mapping Module

This module provides mappings between analysis types and appropriate GA4 metrics,
as well as functions for intelligent metric selection based on prompts.
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
        "dimensions": ["date", "itemName"],
        "metrics": ["addToCarts", "checkouts", "ecommercePurchases", "grossItemRevenue"]
    },
    "User Engagement": {
        "dimensions": ["date"],
        "metrics": ["engagedSessions", "engagementRate", "averageSessionDuration", "screenPageViewsPerSession"]
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
    "engagement": "Behavior Analysis",
    "zaangażowanie": "Behavior Analysis",
    
    # Conversion related keywords
    "konwersje": "Conversion Analysis",
    "conversions": "Conversion Analysis",
    "goals": "Conversion Analysis",
    "cele": "Conversion Analysis",
    "events": "Conversion Analysis",
    "zdarzenia": "Conversion Analysis",
    
    # E-commerce related keywords
    "ecommerce": "E-commerce Analysis",
    "e-commerce": "E-commerce Analysis",
    "produkty": "E-commerce Analysis",
    "products": "E-commerce Analysis",
    "sprzedaż": "E-commerce Analysis",
    "sales": "E-commerce Analysis",
    "revenue": "E-commerce Analysis",
    "przychód": "E-commerce Analysis",
    "koszyk": "E-commerce Analysis",
    "cart": "E-commerce Analysis",
    "zakupy": "E-commerce Analysis",
    "purchases": "E-commerce Analysis"
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
    "grossItemRevenue": "The total revenue from items only. Gross item revenue is the product of its price and quantity.",
    "grossPurchaseRevenue": "The sum of revenue from purchases made in your app or site.",
    "bounceRate": "The percentage of sessions that were not engaged (Sessions Minus Engaged sessions) divided by Sessions).",
    "firstTimePurchaserRate": "The percentage of active users who made their first purchase.",
    "cartToViewRate": "The number of users who added a product(s) to their cart divided by the number of users who viewed the same product(s)."
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
    
    # Special case for e-commerce product analysis
    if "produkt" in prompt_lower and "konwersj" in prompt_lower:
        # If prompt asks about product conversions, use E-commerce Analysis
        selected_analysis_type = "E-commerce Analysis"
        
    # Special case for user engagement
    if "zaangażowanie" in prompt_lower or "engagement" in prompt_lower:
        selected_analysis_type = "User Engagement"
    
    # Return the metrics and dimensions for the selected analysis type
    return GA4_METRICS_MAPPING[selected_analysis_type]
