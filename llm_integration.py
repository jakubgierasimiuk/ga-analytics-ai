"""
LLM Integration Module

This module provides integration with various Large Language Models (LLMs)
for generating insights from Google Analytics data.

Features:
- Support for multiple LLM providers (OpenAI, Anthropic, Google Gemini)
- Prompt template management
- Analytics-specific prompt library
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Union, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import LLM providers with error handling
try:
    import openai
except ImportError:
    logger.warning("OpenAI package not found. OpenAI integration will not be available.")
    openai = None

try:
    import anthropic
except ImportError:
    logger.warning("Anthropic package not found. Anthropic integration will not be available.")
    anthropic = None

class PromptTemplate:
    """Class for managing prompt templates with parameter substitution."""
    
    def __init__(self, template: str, parameters: Dict[str, str] = None):
        """
        Initialize a prompt template.
        
        Args:
            template: The prompt template string with {parameter} placeholders
            parameters: Default parameter values
        """
        self.template = template
        self.parameters = parameters or {}
    
    def render(self, **kwargs) -> str:
        """
        Render the prompt template with provided parameters.
        
        Args:
            **kwargs: Parameter values to use for rendering
            
        Returns:
            Rendered prompt string
        """
        # Combine default parameters with provided ones
        params = self.parameters.copy()
        params.update(kwargs)
        
        # Render the template
        try:
            return self.template.format(**params)
        except KeyError as e:
            logger.error(f"Missing parameter in prompt template: {e}")
            # Return template with missing parameters marked
            return self.template.format(**{k: f"MISSING_{k}" for k in params.keys()})

class AnalyticsPromptLibrary:
    """Library of prompt templates for Google Analytics data analysis."""
    
    def get_general_analysis_prompt(self) -> PromptTemplate:
        """Get a prompt template for general analysis of Google Analytics data."""
        template = """
        Analyze the following Google Analytics data and provide insights:
        
        Time Period: {time_period}
        
        Data Summary:
        {data_summary}
        
        Please provide:
        1. Key trends and patterns in the data
        2. Notable metrics and their significance
        3. Actionable recommendations based on the data
        4. Areas that might need further investigation
        
        Format your response in clear sections with headers.
        """
        
        parameters = {
            "time_period": "Last 30 days",
            "data_summary": "Summary will be provided"
        }
        
        return PromptTemplate(template=template.strip(), parameters=parameters)
    
    def get_traffic_analysis_prompt(self) -> PromptTemplate:
        """Get a prompt template for traffic analysis."""
        template = """
        Analyze the following Google Analytics traffic data and provide insights:
        
        Time Period: {time_period}
        
        Traffic Data:
        {data_summary}
        
        Please provide:
        1. Analysis of traffic sources and their performance
        2. Trends in user acquisition channels
        3. Recommendations for improving traffic quality and quantity
        4. Potential issues or opportunities in the traffic patterns
        
        Format your response in clear sections with headers.
        """
        
        parameters = {
            "time_period": "Last 30 days",
            "data_summary": "Traffic data will be provided"
        }
        
        return PromptTemplate(template=template.strip(), parameters=parameters)
    
    def get_conversion_analysis_prompt(self) -> PromptTemplate:
        """Get a prompt template for conversion analysis."""
        template = """
        Analyze the following Google Analytics conversion data and provide insights:
        
        Time Period: {time_period}
        
        Conversion Data:
        {data_summary}
        
        Please provide:
        1. Analysis of conversion rates and patterns
        2. Factors affecting conversion performance
        3. Recommendations for improving conversion rates
        4. Opportunities for optimization in the conversion funnel
        
        Format your response in clear sections with headers.
        """
        
        parameters = {
            "time_period": "Last 30 days",
            "data_summary": "Conversion data will be provided"
        }
        
        return PromptTemplate(template=template.strip(), parameters=parameters)
    
    def get_user_behavior_analysis_prompt(self) -> PromptTemplate:
        """Get a prompt template for user behavior analysis."""
        template = """
        Analyze the following Google Analytics user behavior data and provide insights:
        
        Time Period: {time_period}
        
        User Behavior Data:
        {data_summary}
        
        Please provide:
        1. Analysis of user engagement patterns
        2. Insights on user journey and flow through the site
        3. Recommendations for improving user experience
        4. Opportunities for enhancing user engagement
        
        Format your response in clear sections with headers.
        """
        
        parameters = {
            "time_period": "Last 30 days",
            "data_summary": "User behavior data will be provided"
        }
        
        return PromptTemplate(template=template.strip(), parameters=parameters)
    
    def get_anomaly_detection_prompt(self) -> PromptTemplate:
        """Get a prompt template for anomaly detection."""
        template = """
        Analyze the following Google Analytics data for anomalies:
        
        Time Period: {time_period}
        
        Data:
        {data_summary}
        
        Detected Anomalies:
        {anomalies}
        
        Please provide:
        1. Analysis of each anomaly and its potential causes
        2. Context and significance of the anomalies
        3. Recommendations for addressing concerning anomalies
        4. Suggestions for monitoring these metrics in the future
        
        Format your response in clear sections with headers.
        """
        
        parameters = {
            "time_period": "Last 30 days",
            "data_summary": "Data will be provided",
            "anomalies": "Anomalies will be provided"
        }
        
        return PromptTemplate(template=template.strip(), parameters=parameters)
    
    def get_comparative_analysis_prompt(self) -> PromptTemplate:
        """Get a prompt template for comparative analysis."""
        template = """
        Compare the following Google Analytics data between two time periods:
        
        Current Period: {current_period}
        Previous Period: {previous_period}
        
        Current Period Data:
        {current_data}
        
        Previous Period Data:
        {previous_data}
        
        Please provide:
        1. Key differences between the two periods
        2. Metrics that have improved or declined significantly
        3. Potential reasons for the observed changes
        4. Recommendations based on the comparison
        
        Format your response in clear sections with headers.
        """
        
        parameters = {
            "current_period": "Last 30 days",
            "previous_period": "Previous 30 days",
            "current_data": "Current period data will be provided",
            "previous_data": "Previous period data will be provided"
        }
        
        return PromptTemplate(template=template.strip(), parameters=parameters)

class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, api_key: str, model: str):
        """
        Initialize the LLM provider.
        
        Args:
            api_key: API key for the provider
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate text based on the prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4)
        """
        super().__init__(api_key, model)
        if openai is None:
            raise ImportError("OpenAI package is not installed")
        
        # Configure OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
    )
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            return f"Error generating text: {str(e)}"

class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            model: Model name (default: claude-3-opus-20240229)
        """
        super().__init__(api_key, model)
        if anthropic is None:
            raise ImportError("Anthropic package is not installed")
        
        # Configure Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate text using Anthropic API.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {e}")
            return f"Error generating text: {str(e)}"

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing or when no API is available."""
    
    def __init__(self, api_key: str = "mock_key", model: str = "mock_model"):
        """Initialize the mock provider."""
        super().__init__(api_key, model)
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate mock text.
        
        Args:
            prompt: The prompt (not used)
            max_tokens: Maximum number of tokens (not used)
            
        Returns:
            Mock generated text
        """
        return """
        # Google Analytics Insights
        
        ## Key Trends
        - User traffic has increased by 15% compared to the previous period
        - Mobile users now account for 65% of all sessions
        - Average session duration has decreased slightly
        
        ## Recommendations
        1. Optimize mobile experience further
        2. Investigate the decrease in session duration
        3. Focus on improving conversion rates for top traffic sources
        
        ## Areas for Further Investigation
        - High bounce rate on product pages
        - Conversion funnel drop-offs
        - Performance of recent marketing campaigns
        """

class LLMFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_name: str, api_key: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            provider_name: Name of the provider (openai, anthropic, mock)
            api_key: API key for the provider
            model: Model name to use
            
        Returns:
            LLM provider instance
            
        Raises:
            ValueError: If provider_name is not recognized
        """
        provider_name = provider_name.lower()
        
        if provider_name == "openai":
            if openai is None:
                logger.warning("OpenAI package not installed, using mock provider instead")
                return MockLLMProvider()
            model = model or "gpt-4"
            return OpenAIProvider(api_key=api_key, model=model)
        
        elif provider_name == "anthropic":
            if anthropic is None:
                logger.warning("Anthropic package not installed, using mock provider instead")
                return MockLLMProvider()
            model = model or "claude-3-opus-20240229"
            return AnthropicProvider(api_key=api_key, model=model)
        
        elif provider_name == "mock":
            return MockLLMProvider()
        
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

class AnalyticsInsightGenerator:
    """Class for generating insights from Google Analytics data using LLMs."""
    
    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the insight generator.
        
        Args:
            llm_provider: Name of the LLM provider to use
            api_key: API key for the provider
            model: Model name to use
        """
        try:
            self.provider = LLMFactory.create_provider(
                provider_name=llm_provider,
                api_key=api_key,
                model=model
            )
            logger.info(f"Initialized AnalyticsInsightGenerator with {llm_provider} provider")
        except Exception as e:
            logger.warning(f"Failed to initialize {llm_provider} provider: {e}. Using mock provider instead.")
            self.provider = MockLLMProvider()
    
    def generate_insights(self, data: Dict[str, Any], prompt_template: PromptTemplate, **kwargs) -> str:
        """
        Generate insights from Google Analytics data.
        
        Args:
            data: Processed Google Analytics data
            prompt_template: Prompt template to use
            **kwargs: Additional parameters for the prompt template
            
        Returns:
            Generated insights
        """
        # Prepare data summary
        data_summary = json.dumps(data.get('summary', {}), indent=2)
        
        # Render the prompt
        prompt = prompt_template.render(
            data_summary=data_summary,
            **kwargs
        )
        
        # Generate insights
        logger.info("Generating insights from data")
        insights = self.provider.generate(prompt=prompt, max_tokens=2000)
        
        return insights
    
    def generate_general_analysis(self, data: Dict[str, Any], time_period: str = "Last 30 days") -> str:
        """
        Generate a general analysis of Google Analytics data.
        
        Args:
            data: Processed Google Analytics data
            time_period: Time period of the data
            
        Returns:
            Generated analysis
        """
        prompt_library = AnalyticsPromptLibrary()
        prompt_template = prompt_library.get_general_analysis_prompt()
        
        return self.generate_insights(
            data=data,
            prompt_template=prompt_template,
            time_period=time_period
        )
    
    def generate_traffic_analysis(self, data: Dict[str, Any], time_period: str = "Last 30 days") -> str:
        """
        Generate a traffic analysis from Google Analytics data.
        
        Args:
            data: Processed Google Analytics data
            time_period: Time period of the data
            
        Returns:
            Generated traffic analysis
        """
        prompt_library = AnalyticsPromptLibrary()
        prompt_template = prompt_library.get_traffic_analysis_prompt()
        
        return self.generate_insights(
            data=data,
            prompt_template=prompt_template,
            time_period=time_period
        )
    
    def generate_conversion_analysis(self, data: Dict[str, Any], time_period: str = "Last 30 days") -> str:
        """
        Generate a conversion analysis from Google Analytics data.
        
        Args:
            data: Processed Google Analytics data
            time_period: Time period of the data
            
        Returns:
            Generated conversion analysis
        """
        prompt_library = AnalyticsPromptLibrary()
        prompt_template = prompt_library.get_conversion_analysis_prompt()
        
        return self.generate_insights(
            data=data,
            prompt_template=prompt_template,
            time_period=time_period
        )
    
    def generate_user_behavior_analysis(self, data: Dict[str, Any], time_period: str = "Last 30 days") -> str:
        """
        Generate a user behavior analysis from Google Analytics data.
        
        Args:
            data: Processed Google Analytics data
            time_period: Time period of the data
            
        Returns:
            Generated user behavior analysis
        """
        prompt_library = AnalyticsPromptLibrary()
        prompt_template = prompt_library.get_user_behavior_analysis_prompt()
        
        return self.generate_insights(
            data=data,
            prompt_template=prompt_template,
            time_period=time_period
        )
    
    def generate_anomaly_analysis(self, data: Dict[str, Any], anomalies: List[Dict[str, Any]], time_period: str = "Last 30 days") -> str:
        """
        Generate an analysis of anomalies in Google Analytics data.
        
        Args:
            data: Processed Google Analytics data
            anomalies: List of detected anomalies
            time_period: Time period of the data
            
        Returns:
            Generated anomaly analysis
        """
        prompt_library = AnalyticsPromptLibrary()
        prompt_template = prompt_library.get_anomaly_detection_prompt()
        
        # Format anomalies as string
        anomalies_str = json.dumps(anomalies, indent=2)
        
        return self.generate_insights(
            data=data,
            prompt_template=prompt_template,
            time_period=time_period,
            anomalies=anomalies_str
        )
    
    def generate_comparative_analysis(self, current_data: Dict[str, Any], previous_data: Dict[str, Any], 
                                     current_period: str = "Last 30 days", previous_period: str = "Previous 30 days") -> str:
        """
        Generate a comparative analysis between two periods of Google Analytics data.
        
        Args:
            current_data: Processed Google Analytics data for current period
            previous_data: Processed Google Analytics data for previous period
            current_period: Description of current time period
            previous_period: Description of previous time period
            
        Returns:
            Generated comparative analysis
        """
        prompt_library = AnalyticsPromptLibrary()
        prompt_template = prompt_library.get_comparative_analysis_prompt()
        
        # Format data as strings
        current_data_str = json.dumps(current_data.get('summary', {}), indent=2)
        previous_data_str = json.dumps(previous_data.get('summary', {}), indent=2)
        
        return self.generate_insights(
            data=current_data,  # Just pass current data as the main data
            prompt_template=prompt_template,
            current_period=current_period,
            previous_period=previous_period,
            current_data=current_data_str,
            previous_data=previous_data_str
        )
