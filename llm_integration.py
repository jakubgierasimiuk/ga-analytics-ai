"""
LLM Integration Module

This module provides integration with Large Language Models (LLMs)
for generating insights from Google Analytics data.

Features:
- Support for OpenAI API
- Compatibility with various OpenAI models including o3
- Prompt formatting for analytics data
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_data_with_llm(prompt: str, api_key: str, model: str = "o3", max_tokens: int = 1000) -> str:
    """
    Analyze data using OpenAI's LLM models.
    
    Args:
        prompt: The prompt containing the data and analysis instructions
        api_key: OpenAI API key
        model: Model name to use (default: o3)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated analysis text
    """
    try:
        # Initialize OpenAI client with API key
        client = OpenAI(api_key=api_key)
        
        # Create chat completion
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        
        # Extract and return the generated text
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error generating analysis with OpenAI: {e}")
        error_message = f"""
        **Error generating analysis**
        
        There was an error when trying to generate the analysis using the OpenAI API:
        
        ```
        {str(e)}
        ```
        
        Please check:
        1. Your API key is valid
        2. You have sufficient credits in your OpenAI account
        3. The selected model ({model}) is available for your account
        4. Your request is not exceeding token limits
        """
        return error_message

def format_prompt_for_analytics(data_description: str, data_content: str, analysis_instructions: str) -> str:
    """
    Format a prompt for analytics data analysis.
    
    Args:
        data_description: Description of the data
        data_content: The actual data content
        analysis_instructions: Instructions for the analysis
        
    Returns:
        Formatted prompt
    """
    prompt = f"""
    # Google Analytics Data Analysis
    
    ## Data Description
    {data_description}
    
    ## Data Content
    ```
    {data_content}
    ```
    
    ## Analysis Instructions
    {analysis_instructions}
    
    Please provide a comprehensive analysis of the data above, following the instructions.
    Format your response with clear sections and insights.
    """
    
    return prompt

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get a dictionary of available OpenAI models with their properties.
    
    Returns:
        Dictionary of model information
    """
    return {
        "o3": {
            "name": "o3",
            "description": "Najnowszy i najpotężniejszy model rozumowania z wiodącą wydajnością w kodowaniu, matematyce i nauce",
            "max_context": 128000
        },
        "gpt-4o": {
            "name": "gpt-4o",
            "description": "Najnowszy model GPT-4 z ulepszoną wydajnością i niższym kosztem",
            "max_context": 128000
        },
        "gpt-4-turbo": {
            "name": "gpt-4-turbo",
            "description": "Szybszy model GPT-4 z dużym kontekstem",
            "max_context": 128000
        },
        "gpt-4": {
            "name": "gpt-4",
            "description": "Standardowy model GPT-4",
            "max_context": 8192
        },
        "gpt-3.5-turbo": {
            "name": "gpt-3.5-turbo",
            "description": "Szybki i ekonomiczny model",
            "max_context": 16385
        }
    }

def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text.
    This is a rough estimation based on the assumption that 1 token ≈ 4 characters.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    return len(text) // 4  # Rough estimation: 1 token ≈ 4 characters

def check_token_limit(prompt: str, model: str) -> bool:
    """
    Check if a prompt is within the token limit for a given model.
    
    Args:
        prompt: The prompt to check
        model: The model name
        
    Returns:
        True if within limit, False otherwise
    """
    models = get_available_models()
    if model not in models:
        return False
    
    max_context = models[model]["max_context"]
    estimated_tokens = estimate_token_count(prompt)
    
    # Allow for response tokens by checking against 80% of max context
    return estimated_tokens <= (max_context * 0.8)
