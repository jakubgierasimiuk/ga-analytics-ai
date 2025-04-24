import re
from typing import List, Dict, Any, Optional, Tuple, Set

# Import GA4 ecommerce metrics module
from ga4_ecommerce_metrics import (
    PRODUCT_ANALYSIS_KEYWORDS,
    ITEM_SCOPED_METRICS,
    ITEM_SCOPED_DIMENSIONS,
    EVENT_SCOPED_METRICS,
    EVENT_SCOPED_DIMENSIONS,
    MONETARY_METRICS,
    PRODUCT_ANALYSIS_TYPES
)

# Enhanced product-specific keywords for better prompt understanding
ENHANCED_PRODUCT_KEYWORDS = {
    "en": {
        "product_performance": [
            "product performance", "best selling", "top products", "worst products", 
            "product ranking", "product sales", "product revenue", "product views",
            "popular products", "unpopular products", "product comparison"
        ],
        "cart_analysis": [
            "cart analysis", "shopping cart", "basket analysis", "add to cart", 
            "cart abandonment", "items in cart", "cart to purchase", "cart conversion",
            "abandoned cart", "cart recovery", "cart behavior"
        ],
        "checkout_analysis": [
            "checkout analysis", "checkout process", "checkout funnel", "payment method",
            "shipping method", "checkout abandonment", "checkout conversion", 
            "purchase completion", "checkout steps", "checkout optimization"
        ],
        "product_list_performance": [
            "product list", "category page", "product catalog", "collection page",
            "product grid", "listing page", "product feed", "product showcase",
            "product display", "product gallery"
        ],
        "promotion_effectiveness": [
            "promotion", "discount", "offer", "coupon", "sale", "deal", "campaign",
            "special offer", "promotional banner", "marketing campaign", "promo code"
        ]
    },
    "pl": {
        "product_performance": [
            "wydajność produktu", "najlepiej sprzedające się", "najlepsze produkty", 
            "najgorsze produkty", "ranking produktów", "sprzedaż produktów", 
            "przychód z produktów", "wyświetlenia produktów", "popularne produkty", 
            "niepopularne produkty", "porównanie produktów"
        ],
        "cart_analysis": [
            "analiza koszyka", "koszyk zakupowy", "analiza koszyka", "dodaj do koszyka", 
            "porzucenie koszyka", "produkty w koszyku", "koszyk do zakupu", 
            "konwersja koszyka", "porzucony koszyk", "odzyskiwanie koszyka", 
            "zachowanie koszyka"
        ],
        "checkout_analysis": [
            "analiza zamówienia", "proces zamówienia", "lejek zamówienia", "metoda płatności",
            "metoda dostawy", "porzucenie zamówienia", "konwersja zamówienia", 
            "ukończenie zakupu", "kroki zamówienia", "optymalizacja zamówienia"
        ],
        "product_list_performance": [
            "lista produktów", "strona kategorii", "katalog produktów", "strona kolekcji",
            "siatka produktów", "strona z listą", "feed produktów", "prezentacja produktów",
            "wyświetlanie produktów", "galeria produktów"
        ],
        "promotion_effectiveness": [
            "promocja", "zniżka", "oferta", "kupon", "wyprzedaż", "okazja", "kampania",
            "oferta specjalna", "baner promocyjny", "kampania marketingowa", "kod promocyjny"
        ]
    }
}

# Product-specific entities that can be extracted from prompts
PRODUCT_ENTITIES = {
    "en": {
        "product": ["product", "item", "merchandise", "good", "sku"],
        "category": ["category", "collection", "type", "group", "department", "section"],
        "brand": ["brand", "manufacturer", "maker", "vendor", "supplier"],
        "price": ["price", "cost", "value", "amount", "fee"],
        "discount": ["discount", "sale", "offer", "deal", "promotion", "coupon"],
        "cart": ["cart", "basket", "bag", "shopping cart"],
        "checkout": ["checkout", "payment", "purchase", "order", "transaction"],
        "conversion": ["conversion", "purchase rate", "success rate", "completion rate"]
    },
    "pl": {
        "product": ["produkt", "przedmiot", "towar", "artykuł", "sku"],
        "category": ["kategoria", "kolekcja", "typ", "grupa", "dział", "sekcja"],
        "brand": ["marka", "producent", "wytwórca", "dostawca"],
        "price": ["cena", "koszt", "wartość", "kwota", "opłata"],
        "discount": ["zniżka", "rabat", "promocja", "oferta", "kupon"],
        "cart": ["koszyk", "kosz", "torba", "koszyk zakupowy"],
        "checkout": ["zamówienie", "płatność", "zakup", "transakcja"],
        "conversion": ["konwersja", "współczynnik zakupu", "współczynnik sukcesu", "współczynnik ukończenia"]
    }
}

# Enhanced function to detect product analysis intent with confidence score
def detect_product_analysis_intent_enhanced(prompt: str, language: str = "en") -> Tuple[bool, float]:
    """
    Enhanced detection of product-specific analysis intent with confidence score.
    
    Args:
        prompt: The analysis prompt text
        language: Language code (en or pl)
        
    Returns:
        Tuple of (is_product_analysis, confidence_score)
    """
    prompt_lower = prompt.lower()
    
    # Get keywords for the specified language or default to English
    keywords = PRODUCT_ANALYSIS_KEYWORDS.get(language, PRODUCT_ANALYSIS_KEYWORDS["en"])
    
    # Count keyword matches
    keyword_matches = 0
    for keyword in keywords:
        if keyword.lower() in prompt_lower:
            keyword_matches += 1
    
    # Get enhanced keywords for all analysis types
    enhanced_keywords = ENHANCED_PRODUCT_KEYWORDS.get(language, ENHANCED_PRODUCT_KEYWORDS["en"])
    
    # Count enhanced keyword matches
    enhanced_matches = 0
    for analysis_type, type_keywords in enhanced_keywords.items():
        for keyword in type_keywords:
            if keyword.lower() in prompt_lower:
                enhanced_matches += 1
    
    # Count entity matches
    entity_matches = 0
    entities = PRODUCT_ENTITIES.get(language, PRODUCT_ENTITIES["en"])
    for entity_type, entity_keywords in entities.items():
        for keyword in entity_keywords:
            if keyword.lower() in prompt_lower:
                entity_matches += 1
    
    # Calculate confidence score (0.0 to 1.0)
    # Weight factors can be adjusted based on importance
    base_weight = 0.4
    enhanced_weight = 0.4
    entity_weight = 0.2
    
    max_base_keywords = len(keywords)
    max_enhanced_keywords = sum(len(keywords) for keywords in enhanced_keywords.values()) / 5  # Average per type
    max_entity_keywords = sum(len(keywords) for keywords in entities.values()) / 8  # Average per entity
    
    base_score = min(1.0, keyword_matches / max_base_keywords) if max_base_keywords > 0 else 0
    enhanced_score = min(1.0, enhanced_matches / max_enhanced_keywords) if max_enhanced_keywords > 0 else 0
    entity_score = min(1.0, entity_matches / max_entity_keywords) if max_entity_keywords > 0 else 0
    
    confidence_score = (base_weight * base_score) + (enhanced_weight * enhanced_score) + (entity_weight * entity_score)
    
    # Determine if this is a product analysis based on confidence threshold
    is_product_analysis = confidence_score >= 0.3  # Threshold can be adjusted
    
    return is_product_analysis, confidence_score

# Enhanced function to determine the most appropriate product analysis type
def get_product_analysis_type_enhanced(prompt: str, language: str = "en") -> Tuple[str, float]:
    """
    Enhanced determination of the most appropriate product analysis type with confidence score.
    
    Args:
        prompt: The analysis prompt text
        language: Language code (en or pl)
        
    Returns:
        Tuple of (analysis_type, confidence_score)
    """
    prompt_lower = prompt.lower()
    
    # Get enhanced keywords for all analysis types
    enhanced_keywords = ENHANCED_PRODUCT_KEYWORDS.get(language, ENHANCED_PRODUCT_KEYWORDS["en"])
    
    # Calculate match scores for each analysis type
    type_scores = {}
    for analysis_type, type_keywords in enhanced_keywords.items():
        matches = 0
        for keyword in type_keywords:
            if keyword.lower() in prompt_lower:
                matches += 1
        
        # Calculate score as percentage of matched keywords
        score = matches / len(type_keywords) if len(type_keywords) > 0 else 0
        type_scores[analysis_type] = score
    
    # Find the analysis type with the highest score
    best_type = max(type_scores.items(), key=lambda x: x[1])
    analysis_type, confidence_score = best_type
    
    # If confidence is too low, default to general product performance
    if confidence_score < 0.2:  # Threshold can be adjusted
        return "product_performance", confidence_score
    
    return analysis_type, confidence_score

# Function to extract product-specific entities from prompt
def extract_product_entities(prompt: str, language: str = "en") -> Dict[str, List[str]]:
    """
    Extract product-specific entities from the prompt.
    
    Args:
        prompt: The analysis prompt text
        language: Language code (en or pl)
        
    Returns:
        Dictionary of entity types and their extracted values
    """
    prompt_lower = prompt.lower()
    entities = PRODUCT_ENTITIES.get(language, PRODUCT_ENTITIES["en"])
    
    extracted_entities = {}
    
    # Simple extraction based on patterns
    for entity_type, entity_keywords in entities.items():
        extracted = []
        
        for keyword in entity_keywords:
            # Look for patterns like "product X" or "X product"
            pattern1 = rf"{keyword}\s+(\w+)"
            pattern2 = rf"(\w+)\s+{keyword}"
            
            matches1 = re.findall(pattern1, prompt_lower)
            matches2 = re.findall(pattern2, prompt_lower)
            
            extracted.extend(matches1)
            extracted.extend(matches2)
        
        if extracted:
            extracted_entities[entity_type] = list(set(extracted))
    
    return extracted_entities

# Enhanced function to select product metrics and dimensions based on prompt
def select_product_metrics_and_dimensions_enhanced(prompt: str, language: str = "en") -> Dict[str, Any]:
    """
    Enhanced selection of appropriate metrics and dimensions for product analysis based on the prompt.
    
    Args:
        prompt: The analysis prompt text
        language: Language code (en or pl)
        
    Returns:
        Dictionary with 'metrics', 'dimensions', 'analysis_type', and 'confidence' keys
    """
    # First check if this is a product analysis request
    is_product_analysis, base_confidence = detect_product_analysis_intent_enhanced(prompt, language)
    
    if not is_product_analysis:
        return None
    
    # Determine the type of product analysis
    analysis_type, type_confidence = get_product_analysis_type_enhanced(prompt, language)
    
    # Extract entities for potential customization
    entities = extract_product_entities(prompt, language)
    
    # Get the default metrics and dimensions for this analysis type
    metrics = PRODUCT_ANALYSIS_TYPES[analysis_type]["metrics"].copy()
    dimensions = PRODUCT_ANALYSIS_TYPES[analysis_type]["dimensions"].copy()
    
    # Customize dimensions based on extracted entities
    if "category" in entities and "itemCategory" not in dimensions:
        dimensions.append("itemCategory")
    
    if "brand" in entities and "itemBrand" not in dimensions:
        dimensions.append("itemBrand")
    
    # Ensure compatibility between metrics and dimensions
    from ga4_ecommerce_metrics import get_compatible_metrics_and_dimensions
    metrics, dimensions = get_compatible_metrics_and_dimensions(metrics, dimensions)
    
    # Calculate overall confidence
    confidence = (base_confidence + type_confidence) / 2
    
    return {
        "metrics": metrics,
        "dimensions": dimensions,
        "analysis_type": analysis_type,
        "confidence": confidence,
        "entities": entities
    }

# Function to generate a product-specific analysis prompt based on the analysis type
def generate_product_analysis_prompt(analysis_type: str, entities: Dict[str, List[str]] = None, language: str = "en") -> str:
    """
    Generate a product-specific analysis prompt based on the analysis type and extracted entities.
    
    Args:
        analysis_type: The type of product analysis
        entities: Dictionary of extracted entities
        language: Language code (en or pl)
        
    Returns:
        Generated analysis prompt
    """
    if language == "pl":
        prompts = {
            "product_performance": "Przeanalizuj wydajność produktów na podstawie dostarczonych danych Google Analytics. "
                                  "Zidentyfikuj najlepiej i najgorzej sprzedające się produkty, trendy w czasie oraz "
                                  "czynniki wpływające na wydajność produktów.",
            
            "cart_analysis": "Przeanalizuj zachowania związane z koszykiem na podstawie dostarczonych danych Google Analytics. "
                            "Zidentyfikuj produkty często dodawane do koszyka, wzorce porzucania koszyka oraz "
                            "współczynniki konwersji od dodania do koszyka do zakupu.",
            
            "checkout_analysis": "Przeanalizuj proces zamówienia na podstawie dostarczonych danych Google Analytics. "
                               "Zidentyfikuj współczynniki konwersji, punkty porzucania oraz "
                               "możliwości optymalizacji procesu zamówienia.",
            
            "product_list_performance": "Przeanalizuj wydajność list produktów na podstawie dostarczonych danych Google Analytics. "
                                      "Zidentyfikuj, które listy produktów generują najwięcej interakcji i konwersji.",
            
            "promotion_effectiveness": "Przeanalizuj skuteczność promocji produktów na podstawie dostarczonych danych Google Analytics. "
                                     "Zidentyfikuj, które promocje generują najwięcej interakcji i konwersji."
        }
    else:  # default to English
        prompts = {
            "product_performance": "Analyze product performance based on the provided Google Analytics data. "
                                  "Identify top and worst performing products, trends over time, and "
                                  "factors influencing product performance.",
            
            "cart_analysis": "Analyze shopping cart behavior based on the provided Google Analytics data. "
                            "Identify products frequently added to cart, cart abandonment patterns, and "
                            "conversion rates from cart to purchase.",
            
            "checkout_analysis": "Analyze checkout process based on the provided Google Analytics data. "
                               "Identify conversion rates, abandonment points, and "
                               "opportunities to optimize the checkout flow.",
            
            "product_list_performance": "Analyze product list performance based on the provided Google Analytics data. "
                                      "Identify which product lists generate the most engagement and conversions.",
            
            "promotion_effectiveness": "Analyze promotion effectiveness based on the provided Google Analytics data. "
                                     "Identify which promotions generate the most engagement and conversions."
        }
    
    # Get base prompt
    base_prompt = prompts.get(analysis_type, prompts["product_performance"])
    
    # Customize prompt based on extracted entities if available
    if entities:
        custom_parts = []
        
        if language == "pl":
            if "product" in entities and len(entities["product"]) > 0:
                products = ", ".join(entities["product"])
                custom_parts.append(f"Zwróć szczególną uwagę na produkty związane z: {products}.")
            
            if "category" in entities and len(entities["category"]) > 0:
                categories = ", ".join(entities["category"])
                custom_parts.append(f"Przeanalizuj wydajność kategorii: {categories}.")
            
            if "brand" in entities and len(entities["brand"]) > 0:
                brands = ", ".join(entities["brand"])
                custom_parts.append(f"Uwzględnij analizę marek: {brands}.")
        else:
            if "product" in entities and len(entities["product"]) > 0:
                products = ", ".join(entities["product"])
                custom_parts.append(f"Pay special attention to products related to: {products}.")
            
            if "category" in entities and len(entities["category"]) > 0:
                categories = ", ".join(entities["category"])
                custom_parts.append(f"Analyze the performance of categories: {categories}.")
            
            if "brand" in entities and len(entities["brand"]) > 0:
                brands = ", ".join(entities["brand"])
                custom_parts.append(f"Include analysis of brands: {brands}.")
        
        # Add custom parts to base prompt
        if custom_parts:
            base_prompt += " " + " ".join(custom_parts)
    
    return base_prompt

# Function to suggest product-specific metrics and dimensions based on user query
def suggest_product_metrics_and_dimensions(query: str, language: str = "en") -> Dict[str, Any]:
    """
    Suggest product-specific metrics and dimensions based on user query.
    
    Args:
        query: User query text
        language: Language code (en or pl)
        
    Returns:
        Dictionary with suggestions or None if not a product query
    """
    # Check if this is a product-related query
    is_product_query, confidence = detect_product_analysis_intent_enhanced(query, language)
    
    if not is_product_query:
        return None
    
    # Get product analysis configuration
    product_analysis = select_product_metrics_and_dimensions_enhanced(query, language)
    
    if not product_analysis:
        return None
    
    # Extract entities
    entities = extract_product_entities(query, language)
    
    # Generate a better prompt
    suggested_prompt = generate_product_analysis_prompt(
        product_analysis["analysis_type"], 
        entities,
        language
    )
    
    # Return suggestions
    return {
        "is_product_query": True,
        "confidence": confidence,
        "analysis_type": product_analysis["analysis_type"],
        "suggested_metrics": product_analysis["metrics"],
        "suggested_dimensions": product_analysis["dimensions"],
        "extracted_entities": entities,
        "suggested_prompt": suggested_prompt
    }

# Test the enhanced product analysis functions
if __name__ == "__main__":
    test_prompts = [
        "Analyze which products were most frequently added to the cart and which had the highest conversion rate from cart to sale",
        "Show me the top 10 best selling products by revenue",
        "What is the cart abandonment rate for each product category?",
        "Which promotions generated the most sales last month?",
        "Compare checkout completion rates across different device categories",
        "Analyze website traffic sources",  # Non-product query
        "Które produkty były najczęściej dodawane do koszyka?",  # Polish: Which products were most frequently added to cart?
        "Pokaż mi najlepiej sprzedające się produkty według przychodów"  # Polish: Show me the best selling products by revenue
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        
        # Detect language (simplified)
        language = "pl" if any(word in prompt.lower() for word in ["produkty", "koszyka", "sprzedające"]) else "en"
        
        # Test enhanced detection
        is_product, confidence = detect_product_analysis_intent_enhanced(prompt, language)
        print(f"Is product analysis: {is_product} (confidence: {confidence:.2f})")
        
        if is_product:
            # Test enhanced analysis type detection
            analysis_type, type_confidence = get_product_analysis_type_enhanced(prompt, language)
            print(f"Analysis type: {analysis_type} (confidence: {type_confidence:.2f})")
            
            # Test entity extraction
            entities = extract_product_entities(prompt, language)
            print(f"Extracted entities: {entities}")
            
            # Test enhanced metrics and dimensions selection
            product_analysis = select_product_metrics_and_dimensions_enhanced(prompt, language)
            print(f"Metrics: {product_analysis['metrics']}")
            print(f"Dimensions: {product_analysis['dimensions']}")
            
            # Test prompt generation
            generated_prompt = generate_product_analysis_prompt(analysis_type, entities, language)
            print(f"Generated prompt: {generated_prompt}")
