import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# GA4 E-commerce Metrics and Dimensions
# Based on GA4 documentation: https://support.google.com/analytics/answer/13428834

# Item-scoped metrics (count interactions with specific items)
ITEM_SCOPED_METRICS = [
    "itemsAddedToCart",       # Number of items added to cart
    "itemsCheckedOut",        # Number of items in checkout events
    "itemsClickedInList",     # Number of items clicked in a list
    "itemsClickedInPromotion", # Number of items clicked in a promotion
    "itemsPurchased",         # Number of items purchased
    "itemsRefunded",          # Number of items refunded
    "itemsViewed",            # Number of items viewed
    "itemsViewedInList",      # Number of items viewed in a list
    "itemsViewedInPromotion", # Number of items viewed in a promotion
    "itemQuantity",           # Number of units for a single item in an event
]

# Event-scoped metrics (count number of ecommerce events)
EVENT_SCOPED_METRICS = [
    "addToCarts",             # Number of add_to_cart events
    "checkouts",              # Number of begin_checkout events
    "itemListClickEvents",    # Number of select_item events
    "itemListViewEvents",     # Number of view_item_list events
    "itemViewEvents",         # Number of view_item events
    "promotionClicks",        # Number of select_promotion events
    "promotionViews",         # Number of view_promotion events
    "purchases",              # Number of purchase events
    "refunds",                # Number of refund events
    "ecommerceQuantity",      # Total number of items in ecommerce events
]

# Item-scoped dimensions (describe properties of specific items)
ITEM_SCOPED_DIMENSIONS = [
    "itemId",                 # Unique identifier for the item
    "itemName",               # Name of the item
    "itemBrand",              # Brand of the item
    "itemCategory",           # Primary category of the item
    "itemCategory2",          # Second level category
    "itemCategory3",          # Third level category
    "itemCategory4",          # Fourth level category
    "itemCategory5",          # Fifth level category
    "itemVariant",            # Variant of the item
    "itemListName",           # Name of the list where the item was presented
    "itemListId",             # ID of the list where the item was presented
    "itemPromotionName",      # Name of the promotion associated with the item
    "itemPromotionId",        # ID of the promotion associated with the item
    "itemAffiliation",        # Store or affiliation from which the item was purchased
]

# Event-scoped dimensions (describe properties of ecommerce events)
EVENT_SCOPED_DIMENSIONS = [
    "transactionId",          # Unique identifier for the transaction
    "affiliation",            # Store or affiliation from which the purchase was made
    "coupon",                 # Coupon code used for the purchase
    "currency",               # Currency code for the purchase
    "discount",               # Monetary discount value
    "paymentType",            # Method of payment
    "shippingTier",           # Shipping method selected
]

# Monetary metrics (revenue-related)
MONETARY_METRICS = [
    "itemRevenue",            # Revenue from a specific item
    "itemRefundAmount",       # Refund amount for a specific item
    "discount",               # Discount amount applied
    "tax",                    # Tax amount
    "shipping",               # Shipping cost
    "totalRevenue",           # Total revenue including tax and shipping
]

# Mapping of common product analysis types to appropriate metrics and dimensions
PRODUCT_ANALYSIS_TYPES = {
    "product_performance": {
        "metrics": ["itemsViewed", "itemsAddedToCart", "itemsPurchased", "itemRevenue"],
        "dimensions": ["itemName", "itemId", "itemCategory", "itemBrand"]
    },
    "cart_analysis": {
        "metrics": ["itemsAddedToCart", "itemsCheckedOut", "itemsPurchased"],
        "dimensions": ["itemName", "itemId", "itemCategory"]
    },
    "checkout_analysis": {
        "metrics": ["checkouts", "purchases", "totalRevenue"],
        "dimensions": ["paymentType", "shippingTier", "coupon"]
    },
    "product_list_performance": {
        "metrics": ["itemsViewedInList", "itemsClickedInList", "itemsAddedToCart"],
        "dimensions": ["itemListName", "itemListId", "itemName"]
    },
    "promotion_effectiveness": {
        "metrics": ["itemsViewedInPromotion", "itemsClickedInPromotion", "itemsPurchased"],
        "dimensions": ["itemPromotionName", "itemPromotionId", "itemName"]
    }
}

# Keywords that indicate product-specific analysis is needed
PRODUCT_ANALYSIS_KEYWORDS = {
    "en": [
        "product", "item", "cart", "basket", "checkout", "purchase", "conversion", 
        "revenue", "sales", "bestseller", "best-selling", "popular", "promotion",
        "discount", "offer", "campaign", "category", "brand", "variant", "refund"
    ],
    "pl": [
        "produkt", "przedmiot", "koszyk", "zakup", "konwersja", "przychód", 
        "sprzedaż", "bestseller", "najpopularniejszy", "promocja", "zniżka", 
        "oferta", "kampania", "kategoria", "marka", "wariant", "zwrot"
    ]
}

def detect_product_analysis_intent(prompt: str, language: str = "en") -> bool:
    """
    Detect if the prompt is asking for product-specific analysis.
    
    Args:
        prompt: The analysis prompt text
        language: Language code (en or pl)
        
    Returns:
        True if product analysis is detected, False otherwise
    """
    prompt_lower = prompt.lower()
    keywords = PRODUCT_ANALYSIS_KEYWORDS.get(language, PRODUCT_ANALYSIS_KEYWORDS["en"])
    
    for keyword in keywords:
        if keyword.lower() in prompt_lower:
            return True
    
    return False

def get_product_analysis_type(prompt: str) -> str:
    """
    Determine the most appropriate product analysis type based on the prompt.
    
    Args:
        prompt: The analysis prompt text
        
    Returns:
        The name of the most appropriate product analysis type
    """
    prompt_lower = prompt.lower()
    
    # Check for specific analysis types
    if any(word in prompt_lower for word in ["cart", "basket", "koszyk"]):
        return "cart_analysis"
    elif any(word in prompt_lower for word in ["checkout", "payment", "płatność"]):
        return "checkout_analysis"
    elif any(word in prompt_lower for word in ["list", "category", "kategoria", "lista"]):
        return "product_list_performance"
    elif any(word in prompt_lower for word in ["promotion", "campaign", "promocja", "kampania"]):
        return "promotion_effectiveness"
    else:
        # Default to general product performance
        return "product_performance"

def get_compatible_metrics_and_dimensions(metrics: List[str], dimensions: List[str]) -> Tuple[List[str], List[str]]:
    """
    Ensure that the selected metrics and dimensions are compatible with each other.
    
    Args:
        metrics: List of selected metrics
        dimensions: List of selected dimensions
        
    Returns:
        Tuple of (compatible_metrics, compatible_dimensions)
    """
    # Check if we're mixing item-scoped and event-scoped metrics/dimensions
    has_item_metrics = any(metric in ITEM_SCOPED_METRICS for metric in metrics)
    has_event_metrics = any(metric in EVENT_SCOPED_METRICS for metric in metrics)
    has_item_dimensions = any(dimension in ITEM_SCOPED_DIMENSIONS for dimension in dimensions)
    has_event_dimensions = any(dimension in EVENT_SCOPED_DIMENSIONS for dimension in dimensions)
    
    # If we have both item and event scoped elements, prioritize item-scoped
    if (has_item_metrics or has_item_dimensions) and (has_event_metrics or has_event_dimensions):
        # Filter to keep only item-scoped metrics and dimensions
        compatible_metrics = [m for m in metrics if m in ITEM_SCOPED_METRICS or m in MONETARY_METRICS]
        compatible_dimensions = [d for d in dimensions if d in ITEM_SCOPED_DIMENSIONS]
        
        # If we've filtered out all metrics or dimensions, add some defaults
        if not compatible_metrics:
            compatible_metrics = ["itemsViewed", "itemsAddedToCart", "itemsPurchased"]
        if not compatible_dimensions:
            compatible_dimensions = ["itemName", "itemId"]
    else:
        # No compatibility issues
        compatible_metrics = metrics
        compatible_dimensions = dimensions
    
    return compatible_metrics, compatible_dimensions

def select_product_metrics_and_dimensions(prompt: str, language: str = "en") -> Dict[str, List[str]]:
    """
    Select appropriate metrics and dimensions for product analysis based on the prompt.
    
    Args:
        prompt: The analysis prompt text
        language: Language code (en or pl)
        
    Returns:
        Dictionary with 'metrics' and 'dimensions' keys
    """
    # First check if this is a product analysis request
    if not detect_product_analysis_intent(prompt, language):
        return None
    
    # Determine the type of product analysis
    analysis_type = get_product_analysis_type(prompt)
    
    # Get the default metrics and dimensions for this analysis type
    metrics = PRODUCT_ANALYSIS_TYPES[analysis_type]["metrics"].copy()
    dimensions = PRODUCT_ANALYSIS_TYPES[analysis_type]["dimensions"].copy()
    
    # Ensure compatibility between metrics and dimensions
    metrics, dimensions = get_compatible_metrics_and_dimensions(metrics, dimensions)
    
    return {
        "metrics": metrics,
        "dimensions": dimensions,
        "analysis_type": analysis_type
    }

# Function to check if a specific combination of metrics and dimensions is compatible
def is_compatible_combination(metrics: List[str], dimensions: List[str]) -> bool:
    """
    Check if the given combination of metrics and dimensions is compatible according to GA4 rules.
    
    Args:
        metrics: List of metrics
        dimensions: List of dimensions
        
    Returns:
        True if compatible, False otherwise
    """
    # Check for known incompatible combinations
    has_item_metrics = any(metric in ITEM_SCOPED_METRICS for metric in metrics)
    has_event_metrics = any(metric in EVENT_SCOPED_METRICS for metric in metrics)
    has_item_dimensions = any(dimension in ITEM_SCOPED_DIMENSIONS for dimension in dimensions)
    has_event_dimensions = any(dimension in EVENT_SCOPED_DIMENSIONS for dimension in dimensions)
    
    # Item-scoped metrics are incompatible with event-scoped dimensions and vice versa
    if (has_item_metrics and has_event_dimensions) or (has_event_metrics and has_item_dimensions):
        return False
    
    # Check for specific incompatible combinations
    if "itemName" in dimensions and "grossItemRevenue" in metrics:
        return False
    
    return True

# Function to get safe metrics and dimensions that are compatible
def get_safe_metrics_and_dimensions(metrics: List[str], dimensions: List[str]) -> Tuple[List[str], List[str]]:
    """
    Get a safe set of metrics and dimensions that are compatible with GA4.
    
    Args:
        metrics: List of requested metrics
        dimensions: List of requested dimensions
        
    Returns:
        Tuple of (safe_metrics, safe_dimensions)
    """
    if is_compatible_combination(metrics, dimensions):
        return metrics, dimensions
    
    # If incompatible, try to fix by prioritizing item-scoped or event-scoped
    has_item_dimensions = any(dimension in ITEM_SCOPED_DIMENSIONS for dimension in dimensions)
    
    if has_item_dimensions:
        # Prioritize item-scoped metrics
        safe_metrics = [m for m in metrics if m in ITEM_SCOPED_METRICS or m in MONETARY_METRICS]
        if not safe_metrics:
            safe_metrics = ["itemsViewed", "itemsAddedToCart", "itemsPurchased"]
    else:
        # Prioritize event-scoped metrics
        safe_metrics = [m for m in metrics if m in EVENT_SCOPED_METRICS]
        if not safe_metrics:
            safe_metrics = ["screenPageViews", "conversions", "eventCount"]
    
    # Remove specific incompatible combinations
    if "itemName" in dimensions and "grossItemRevenue" in safe_metrics:
        safe_metrics.remove("grossItemRevenue")
    
    return safe_metrics, dimensions
