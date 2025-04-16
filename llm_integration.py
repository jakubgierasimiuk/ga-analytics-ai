"""
LLM Integration Module

This module provides functionality to connect to various LLM providers
and generate insights from Google Analytics data.

Features:
- Support for multiple LLM providers (OpenAI, Anthropic, Google, etc.)
- Prompt template management
- Context optimization for analytics data
- Response parsing and formatting
- Error handling and retry logic
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import openai
from openai import OpenAI
import anthropic
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM provider.
        
        Args:
            api_key: API key for the provider
        """
        self.api_key = api_key or os.environ.get(self._get_env_var_name())
        if not self.api_key:
            raise ValueError(f"API key not provided and {self._get_env_var_name()} environment variable not set")
    
    def _get_env_var_name(self) -> str:
        """Get the environment variable name for the API key."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_available_models(self) -> List[str]:
        """Get a list of available models from the provider."""
        raise NotImplementedError("Subclasses must implement this method")


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4)
        """
        super().__init__(api_key)
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
    
    def _get_env_var_name(self) -> str:
        return "OPENAI_API_KEY"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError))
    )
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: Optional[int] = None,
                         **kwargs) -> str:
        """
        Generate a response from OpenAI.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt to set context
            temperature: Temperature parameter (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        try:
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            # Call the API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract and return the response text
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {str(e)}")
            raise
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models from OpenAI.
        
        Returns:
            List of model IDs
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Error getting available models from OpenAI: {str(e)}")
            return []


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            model: Model to use (default: claude-3-opus-20240229)
        """
        super().__init__(api_key)
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _get_env_var_name(self) -> str:
        return "ANTHROPIC_API_KEY"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError))
    )
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: int = 4096,
                         **kwargs) -> str:
        """
        Generate a response from Anthropic Claude.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt to set context
            temperature: Temperature parameter (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        try:
            # Call the API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            
            # Extract and return the response text
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Error generating response from Anthropic: {str(e)}")
            raise
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models from Anthropic.
        
        Returns:
            List of model IDs
        """
        # Anthropic doesn't have a models.list endpoint, so we return a static list
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro"):
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: Google API key
            model: Model to use (default: gemini-pro)
        """
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    def _get_env_var_name(self) -> str:
        return "GOOGLE_API_KEY"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException))
    )
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: Optional[int] = None,
                         **kwargs) -> str:
        """
        Generate a response from Google Gemini.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt to set context
            temperature: Temperature parameter (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        try:
            # Prepare the request
            url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
            
            # Prepare the content
            content = []
            
            # Add system prompt if provided
            if system_prompt:
                content.append({
                    "role": "system",
                    "parts": [{"text": system_prompt}]
                })
            
            # Add user prompt
            content.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })
            
            # Prepare the request body
            body = {
                "contents": content,
                "generationConfig": {
                    "temperature": temperature
                }
            }
            
            # Add max_tokens if provided
            if max_tokens:
                body["generationConfig"]["maxOutputTokens"] = max_tokens
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in body["generationConfig"]:
                    body["generationConfig"][key] = value
            
            # Make the request
            response = requests.post(url, json=body)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract and return the response text
            return result["candidates"][0]["content"]["parts"][0]["text"]
            
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {str(e)}")
            raise
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models from Gemini.
        
        Returns:
            List of model IDs
        """
        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            
            result = response.json()
            return [model["name"].split("/")[-1] for model in result["models"]]
            
        except Exception as e:
            logger.error(f"Error getting available models from Gemini: {str(e)}")
            return ["gemini-pro", "gemini-pro-vision"]


class OllamaProvider(LLMProvider):
    """Ollama API provider for local open-source models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama provider.
        
        Args:
            api_key: Not used for Ollama, but kept for interface consistency
            model: Model to use (default: llama3)
            base_url: Base URL for the Ollama API
        """
        self.model = model
        self.base_url = base_url
        # Ollama doesn't use API keys, but we keep the init signature consistent
    
    def _get_env_var_name(self) -> str:
        return "OLLAMA_API_KEY"  # Not used
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException))
    )
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: Optional[int] = None,
                         **kwargs) -> str:
        """
        Generate a response from Ollama.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt to set context
            temperature: Temperature parameter (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        try:
            # Prepare the request
            url = f"{self.base_url}/api/generate"
            
            # Prepare the request body
            body = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
            
            # Add system prompt if provided
            if system_prompt:
                body["system"] = system_prompt
            
            # Add max_tokens if provided
            if max_tokens:
                body["num_predict"] = max_tokens
            
            # Add any additional parameters
            for key, value in kwargs.items():
                body[key] = value
            
            # Make the request
            response = requests.post(url, json=body)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract and return the response text
            return result["response"]
            
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {str(e)}")
            raise
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models from Ollama.
        
        Returns:
            List of model IDs
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url)
            response.raise_for_status()
            
            result = response.json()
            return [model["name"] for model in result["models"]]
            
        except Exception as e:
            logger.error(f"Error getting available models from Ollama: {str(e)}")
            return []


class LLMFactory:
    """Factory class for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_name: str, api_key: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            provider_name: Name of the provider ('openai', 'anthropic', 'gemini', 'ollama')
            api_key: Optional API key
            model: Optional model name
            
        Returns:
            LLMProvider instance
        """
        provider_name = provider_name.lower()
        
        if provider_name == "openai":
            return OpenAIProvider(api_key=api_key, model=model or "gpt-4")
        elif provider_name == "anthropic":
            return AnthropicProvider(api_key=api_key, model=model or "claude-3-opus-20240229")
        elif provider_name == "gemini":
            return GeminiProvider(api_key=api_key, model=model or "gemini-pro")
        elif provider_name == "ollama":
            return OllamaProvider(model=model or "llama3")
        else:
            raise ValueError(f"Unknown provider: {provider_name}")


class PromptTemplate:
    """Class for managing prompt templates."""
    
    def __init__(self, template: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the prompt template.
        
        Args:
            template: Template string with placeholders
            parameters: Optional default parameter values
        """
        self.template = template
        self.parameters = parameters or {}
    
    def render(self, **kwargs) -> str:
        """
        Render the template with the provided parameters.
        
        Args:
            **kwargs: Parameters to use for rendering
            
        Returns:
            Rendered prompt string
        """
        # Combine default parameters with provided ones
        params = {**self.parameters, **kwargs}
        
        # Render the template
        try:
            return self.template.format(**params)
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise ValueError(f"Missing required parameter: {missing_key}")
    
    @classmethod
    def from_file(cls, file_path: str) -> 'PromptTemplate':
        """
        Load a prompt template from a file.
        
        Args:
            file_path: Path to the template file
            
        Returns:
            PromptTemplate instance
        """
        with open(file_path, 'r') as f:
            content = json.load(f)
        
        template = content.get('template', '')
        parameters = content.get('parameters', {})
        
        return cls(template=template, parameters=parameters)
    
    def to_file(self, file_path: str):
        """
        Save the prompt template to a file.
        
        Args:
            file_path: Path to save the template
        """
        content = {
            'template': self.template,
            'parameters': self.parameters
        }
        
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2)


class AnalyticsPromptLibrary:
    """Library of prompt templates for analytics tasks."""
    
    @staticmethod
    def get_general_analysis_prompt() -> PromptTemplate:
        """Get a general analytics analysis prompt template."""
        template = """
You are an expert Google Analytics data analyst. Analyze the following Google Analytics data and provide insights.

Data Summary:
{data_summary}

Sample Data:
{sample_data}

Please provide:
1. A summary of key metrics and their trends
2. Identification of any anomalies or unusual patterns
3. Insights about user behavior
4. Recommendations for improvement
5. Suggestions for further analysis

Focus on {focus_area} if specified.
"""
        parameters = {
            'focus_area': 'overall performance'
        }
        
        return PromptTemplate(template=template, parameters=parameters)
    
    @staticmethod
    def get_traffic_analysis_prompt() -> PromptTemplate:
        """Get a traffic analysis prompt template."""
        template = """
You are an expert in web traffic analysis. Analyze the following Google Analytics traffic data and provide insights.

Data Summary:
{data_summary}

Sample Data:
{sample_data}

Time Period: {time_period}

Please provide:
1. A summary of traffic volume and trends
2. Analysis of traffic sources and channels
3. Identification of top referrers and campaigns
4. Day/time patterns in traffic
5. Device and platform breakdown
6. Recommendations to improve traffic quality and quantity

Present your analysis in a clear, structured format with specific actionable insights.
"""
        parameters = {
            'time_period': 'the last 30 days'
        }
        
        return PromptTemplate(template=template, parameters=parameters)
    
    @staticmethod
    def get_conversion_analysis_prompt() -> PromptTemplate:
        """Get a conversion analysis prompt template."""
        template = """
You are an expert in conversion optimization. Analyze the following Google Analytics conversion data and provide insights.

Data Summary:
{data_summary}

Sample Data:
{sample_data}

Conversion Goals: {conversion_goals}
Time Period: {time_period}

Please provide:
1. Overall conversion rate analysis and trends
2. Breakdown of conversion by source/medium
3. Analysis of the conversion funnel and drop-off points
4. Device and platform impact on conversions
5. Identification of high-performing segments
6. Specific recommendations to improve conversion rates

Focus on actionable insights that can lead to measurable improvements.
"""
        parameters = {
            'conversion_goals': 'all conversion goals',
            'time_period': 'the last 30 days'
        }
        
        return PromptTemplate(template=template, parameters=parameters)
    
    @staticmethod
    def get_user_behavior_analysis_prompt() -> PromptTemplate:
        """Get a user behavior analysis prompt template."""
        template = """
You are an expert in user behavior analysis. Analyze the following Google Analytics user behavior data and provide insights.

Data Summary:
{data_summary}

Sample Data:
{sample_data}

Time Period: {time_period}

Please provide:
1. Analysis of user engagement metrics (session duration, pages per session, etc.)
2. Content consumption patterns
3. User flow and navigation paths
4. Identification of popular content and features
5. Analysis of exit pages and potential issues
6. Recommendations to improve user engagement and retention

Present your analysis in a clear, structured format with specific actionable insights.
"""
        parameters = {
            'time_period': 'the last 30 days'
        }
        
        return PromptTemplate(template=template, parameters=parameters)
    
    @staticmethod
    def get_anomaly_detection_prompt() -> PromptTemplate:
        """Get an anomaly detection prompt template."""
        template = """
You are an expert in data anomaly detection. Analyze the following Google Analytics data and identify any anomalies or unusual patterns.

Data Summary:
{data_summary}

Sample Data:
{sample_data}

Anomaly Data:
{anomaly_data}

Time Period: {time_period}
Metrics of Interest: {metrics_of_interest}

Please provide:
1. Identification of significant anomalies in the data
2. Analysis of potential causes for each anomaly
3. Assessment of the impact of these anomalies
4. Recommendations for addressing or monitoring these anomalies
5. Suggestions for preventing similar anomalies in the future

Focus on anomalies that have business significance rather than minor statistical variations.
"""
        parameters = {
            'time_period': 'the last 30 days',
            'metrics_of_interest': 'all key metrics',
            'anomaly_data': 'No specific anomaly data provided'
        }
        
        return PromptTemplate(template=template, parameters=parameters)
    
    @staticmethod
    def get_comparative_analysis_prompt() -> PromptTemplate:
        """Get a comparative analysis prompt template."""
        template = """
You are an expert in comparative data analysis. Compare the following Google Analytics data sets and provide insights.

Data Set 1 ({period_1}):
{data_summary_1}

Data Set 2 ({period_2}):
{data_summary_2}

Comparison Focus: {comparison_focus}

Please provide:
1. Key metrics comparison between the two periods
2. Significant changes and trends
3. Analysis of factors contributing to the changes
4. Evaluation of performance improvement or decline
5. Recommendations based on the comparative analysis

Present your analysis in a clear, structured format highlighting the most significant changes and their implications.
"""
        parameters = {
            'period_1': 'current period',
            'period_2': 'previous period',
            'comparison_focus': 'overall performance'
        }
        
        return PromptTemplate(template=template, parameters=parameters)


class AnalyticsInsightGenerator:
    """Class for generating insights from Google Analytics data using LLMs."""
    
    def __init__(self, 
                llm_provider: Union[str, LLMProvider],
                api_key: Optional[str] = None,
                model: Optional[str] = None):
        """
        Initialize the insight generator.
        
        Args:
            llm_provider: LLM provider name or instance
            api_key: Optional API key
            model: Optional model name
        """
        if isinstance(llm_provider, str):
            self.provider = LLMFactory.create_provider(
                provider_name=llm_provider,
                api_key=api_key,
                model=model
            )
        else:
            self.provider = llm_provider
        
        self.prompt_library = AnalyticsPromptLibrary()
    
    def _prepare_data_for_prompt(self, data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Prepare data for inclusion in a prompt.
        
        Args:
            data: Data dictionary from GADataProcessor.prepare_data_for_llm()
            
        Returns:
            Tuple of (data_summary, sample_data) as formatted strings
        """
        # Format summary
        summary_lines = ["## Metrics Summary"]
        
        for metric, stats in data['summary']['metrics'].items():
            summary_lines.append(f"- {metric}:")
            summary_lines.append(f"  - Sum: {stats['sum']}")
            summary_lines.append(f"  - Mean: {stats['mean']:.2f}")
            summary_lines.append(f"  - Median: {stats['median']:.2f}")
            summary_lines.append(f"  - Range: {stats['min']} to {stats['max']}")
        
        data_summary = "\n".join(summary_lines)
        
        # Format sample data
        sample_lines = ["## Sample Data (first few rows)"]
        
        # Convert sample data to a formatted table
        if data['sample_data']:
            # Get all keys
            keys = list(data['sample_data'][0].keys())
            
            # Add header
            sample_lines.append(" | ".join(keys))
            sample_lines.append(" | ".join(["-" * len(key) for key in keys]))
            
            # Add rows
            for row in data['sample_data']:
                sample_lines.append(" | ".join([str(row.get(key, "")) for key in keys]))
        
        sample_data = "\n".join(sample_lines)
        
        return data_summary, sample_data
    
    def generate_general_analysis(self, 
                                 data: Dict[str, Any],
                                 focus_area: Optional[str] = None) -> str:
        """
        Generate a general analysis of GA data.
        
        Args:
            data: Data dictionary from GADataProcessor.prepare_data_for_llm()
            focus_area: Optional area to focus the analysis on
            
        Returns:
            Generated analysis text
        """
        # Get the prompt template
        prompt_template = self.prompt_library.get_general_analysis_prompt()
        
        # Prepare data
        data_summary, sample_data = self._prepare_data_for_prompt(data)
        
        # Render the prompt
        params = {
            'data_summary': data_summary,
            'sample_data': sample_data
        }
        
        if focus_area:
            params['focus_area'] = focus_area
        
        prompt = prompt_template.render(**params)
        
        # Generate the response
        system_prompt = "You are an expert data analyst specializing in Google Analytics. Provide clear, actionable insights based on the data."
        
        return self.provider.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3  # Lower temperature for more focused analysis
        )
    
    def generate_traffic_analysis(self,
                                data: Dict[str, Any],
                                time_period: str = "the last 30 days") -> str:
        """
        Generate a traffic analysis from GA data.
        
        Args:
            data: Data dictionary from GADataProcessor.prepare_data_for_llm()
            time_period: Description of the time period
            
        Returns:
            Generated analysis text
        """
        # Get the prompt template
        prompt_template = self.prompt_library.get_traffic_analysis_prompt()
        
        # Prepare data
        data_summary, sample_data = self._prepare_data_for_prompt(data)
        
        # Render the prompt
        prompt = prompt_template.render(
            data_summary=data_summary,
            sample_data=sample_data,
            time_period=time_period
        )
        
        # Generate the response
        system_prompt = "You are an expert in web traffic analysis. Provide clear, actionable insights about traffic patterns and sources."
        
        return self.provider.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )
    
    def generate_conversion_analysis(self,
                                   data: Dict[str, Any],
                                   conversion_goals: str = "all conversion goals",
                                   time_period: str = "the last 30 days") -> str:
        """
        Generate a conversion analysis from GA data.
        
        Args:
            data: Data dictionary from GADataProcessor.prepare_data_for_llm()
            conversion_goals: Description of the conversion goals
            time_period: Description of the time period
            
        Returns:
            Generated analysis text
        """
        # Get the prompt template
        prompt_template = self.prompt_library.get_conversion_analysis_prompt()
        
        # Prepare data
        data_summary, sample_data = self._prepare_data_for_prompt(data)
        
        # Render the prompt
        prompt = prompt_template.render(
            data_summary=data_summary,
            sample_data=sample_data,
            conversion_goals=conversion_goals,
            time_period=time_period
        )
        
        # Generate the response
        system_prompt = "You are an expert in conversion rate optimization. Provide clear, actionable insights to improve conversion rates."
        
        return self.provider.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )
    
    def generate_user_behavior_analysis(self,
                                      data: Dict[str, Any],
                                      time_period: str = "the last 30 days") -> str:
        """
        Generate a user behavior analysis from GA data.
        
        Args:
            data: Data dictionary from GADataProcessor.prepare_data_for_llm()
            time_period: Description of the time period
            
        Returns:
            Generated analysis text
        """
        # Get the prompt template
        prompt_template = self.prompt_library.get_user_behavior_analysis_prompt()
        
        # Prepare data
        data_summary, sample_data = self._prepare_data_for_prompt(data)
        
        # Render the prompt
        prompt = prompt_template.render(
            data_summary=data_summary,
            sample_data=sample_data,
            time_period=time_period
        )
        
        # Generate the response
        system_prompt = "You are an expert in user behavior analysis. Provide clear, actionable insights about how users interact with the website."
        
        return self.provider.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )
    
    def generate_anomaly_analysis(self,
                                data: Dict[str, Any],
                                anomaly_data: Optional[str] = None,
                                metrics_of_interest: str = "all key metrics",
                                time_period: str = "the last 30 days") -> str:
        """
        Generate an anomaly analysis from GA data.
        
        Args:
            data: Data dictionary from GADataProcessor.prepare_data_for_llm()
            anomaly_data: Optional string describing detected anomalies
            metrics_of_interest: Description of the metrics to focus on
            time_period: Description of the time period
            
        Returns:
            Generated analysis text
        """
        # Get the prompt template
        prompt_template = self.prompt_library.get_anomaly_detection_prompt()
        
        # Prepare data
        data_summary, sample_data = self._prepare_data_for_prompt(data)
        
        # Render the prompt
        params = {
            'data_summary': data_summary,
            'sample_data': sample_data,
            'metrics_of_interest': metrics_of_interest,
            'time_period': time_period
        }
        
        if anomaly_data:
            params['anomaly_data'] = anomaly_data
        
        prompt = prompt_template.render(**params)
        
        # Generate the response
        system_prompt = "You are an expert in data anomaly detection. Identify significant anomalies and provide clear explanations and recommendations."
        
        return self.provider.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )
    
    def generate_comparative_analysis(self,
                                    data_1: Dict[str, Any],
                                    data_2: Dict[str, Any],
                                    period_1: str = "current period",
                                    period_2: str = "previous period",
                                    comparison_focus: str = "overall performance") -> str:
        """
        Generate a comparative analysis between two GA data sets.
        
        Args:
            data_1: First data dictionary from GADataProcessor.prepare_data_for_llm()
            data_2: Second data dictionary from GADataProcessor.prepare_data_for_llm()
            period_1: Description of the first period
            period_2: Description of the second period
            comparison_focus: Focus area for the comparison
            
        Returns:
            Generated analysis text
        """
        # Get the prompt template
        prompt_template = self.prompt_library.get_comparative_analysis_prompt()
        
        # Prepare data
        data_summary_1, _ = self._prepare_data_for_prompt(data_1)
        data_summary_2, _ = self._prepare_data_for_prompt(data_2)
        
        # Render the prompt
        prompt = prompt_template.render(
            data_summary_1=data_summary_1,
            data_summary_2=data_summary_2,
            period_1=period_1,
            period_2=period_2,
            comparison_focus=comparison_focus
        )
        
        # Generate the response
        system_prompt = "You are an expert in comparative data analysis. Provide clear insights about the differences between the two periods and their implications."
        
        return self.provider.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )
    
    def generate_custom_analysis(self,
                               data: Dict[str, Any],
                               custom_prompt: str,
                               system_prompt: Optional[str] = None) -> str:
        """
        Generate a custom analysis using a user-provided prompt.
        
        Args:
            data: Data dictionary from GADataProcessor.prepare_data_for_llm()
            custom_prompt: Custom prompt template
            system_prompt: Optional system prompt
            
        Returns:
            Generated analysis text
        """
        # Prepare data
        data_summary, sample_data = self._prepare_data_for_prompt(data)
        
        # Create a temporary prompt template
        prompt_template = PromptTemplate(template=custom_prompt)
        
        # Render the prompt
        try:
            prompt = prompt_template.render(
                data_summary=data_summary,
                sample_data=sample_data
            )
        except ValueError:
            # If the custom prompt doesn't use our standard parameters,
            # just use it directly with the data appended
            prompt = f"{custom_prompt}\n\n{data_summary}\n\n{sample_data}"
        
        # Generate the response
        default_system_prompt = "You are an expert data analyst. Provide clear, actionable insights based on the data."
        
        return self.provider.generate_response(
            prompt=prompt,
            system_prompt=system_prompt or default_system_prompt,
            temperature=0.5  # Slightly higher temperature for custom prompts
        )


# Example usage functions

def example_openai_analysis():
    """Example of generating insights using OpenAI."""
    # Sample data (in a real application, this would come from GADataProcessor)
    sample_data = {
        'metadata': {
            'dimensions': ['date', 'deviceCategory'],
            'metrics': ['activeUsers', 'sessions', 'screenPageViews'],
            'row_count': 90,
            'sample_size': 10
        },
        'summary': {
            'row_count': 90,
            'metrics': {
                'activeUsers': {
                    'sum': 12500,
                    'mean': 138.89,
                    'median': 135.0,
                    'min': 80,
                    'max': 210
                },
                'sessions': {
                    'sum': 18750,
                    'mean': 208.33,
                    'median': 205.0,
                    'min': 120,
                    'max': 320
                },
                'screenPageViews': {
                    'sum': 45000,
                    'mean': 500.0,
                    'median': 490.0,
                    'min': 300,
                    'max': 750
                }
            }
        },
        'sample_data': [
            {'date': '2023-01-01', 'deviceCategory': 'desktop', 'activeUsers': 100, 'sessions': 150, 'screenPageViews': 400},
            {'date': '2023-01-01', 'deviceCategory': 'mobile', 'activeUsers': 80, 'sessions': 120, 'screenPageViews': 300},
            {'date': '2023-01-01', 'deviceCategory': 'tablet', 'activeUsers': 20, 'sessions': 30, 'screenPageViews': 80},
            {'date': '2023-01-02', 'deviceCategory': 'desktop', 'activeUsers': 110, 'sessions': 165, 'screenPageViews': 420},
            {'date': '2023-01-02', 'deviceCategory': 'mobile', 'activeUsers': 85, 'sessions': 130, 'screenPageViews': 320},
            {'date': '2023-01-02', 'deviceCategory': 'tablet', 'activeUsers': 25, 'sessions': 35, 'screenPageViews': 90}
        ]
    }
    
    # Create the insight generator
    generator = AnalyticsInsightGenerator(
        llm_provider='openai',
        api_key='your_api_key_here',  # Replace with your API key
        model='gpt-4'
    )
    
    # Generate general analysis
    analysis = generator.generate_general_analysis(
        data=sample_data,
        focus_area='mobile performance'
    )
    
    print("OpenAI Analysis:")
    print(analysis)
    
    return analysis

def example_anthropic_analysis():
    """Example of generating insights using Anthropic Claude."""
    # Sample data (same as above)
    sample_data = {
        'metadata': {
            'dimensions': ['date', 'deviceCategory'],
            'metrics': ['activeUsers', 'sessions', 'screenPageViews'],
            'row_count': 90,
            'sample_size': 10
        },
        'summary': {
            'row_count': 90,
            'metrics': {
                'activeUsers': {
                    'sum': 12500,
                    'mean': 138.89,
                    'median': 135.0,
                    'min': 80,
                    'max': 210
                },
                'sessions': {
                    'sum': 18750,
                    'mean': 208.33,
                    'median': 205.0,
                    'min': 120,
                    'max': 320
                },
                'screenPageViews': {
                    'sum': 45000,
                    'mean': 500.0,
                    'median': 490.0,
                    'min': 300,
                    'max': 750
                }
            }
        },
        'sample_data': [
            {'date': '2023-01-01', 'deviceCategory': 'desktop', 'activeUsers': 100, 'sessions': 150, 'screenPageViews': 400},
            {'date': '2023-01-01', 'deviceCategory': 'mobile', 'activeUsers': 80, 'sessions': 120, 'screenPageViews': 300},
            {'date': '2023-01-01', 'deviceCategory': 'tablet', 'activeUsers': 20, 'sessions': 30, 'screenPageViews': 80},
            {'date': '2023-01-02', 'deviceCategory': 'desktop', 'activeUsers': 110, 'sessions': 165, 'screenPageViews': 420},
            {'date': '2023-01-02', 'deviceCategory': 'mobile', 'activeUsers': 85, 'sessions': 130, 'screenPageViews': 320},
            {'date': '2023-01-02', 'deviceCategory': 'tablet', 'activeUsers': 25, 'sessions': 35, 'screenPageViews': 90}
        ]
    }
    
    # Create the insight generator
    generator = AnalyticsInsightGenerator(
        llm_provider='anthropic',
        api_key='your_api_key_here',  # Replace with your API key
        model='claude-3-opus-20240229'
    )
    
    # Generate traffic analysis
    analysis = generator.generate_traffic_analysis(
        data=sample_data,
        time_period='January 1-2, 2023'
    )
    
    print("Anthropic Analysis:")
    print(analysis)
    
    return analysis

def example_custom_prompt():
    """Example of using a custom prompt."""
    # Sample data (same as above)
    sample_data = {
        'metadata': {
            'dimensions': ['date', 'deviceCategory'],
            'metrics': ['activeUsers', 'sessions', 'screenPageViews'],
            'row_count': 90,
            'sample_size': 10
        },
        'summary': {
            'row_count': 90,
            'metrics': {
                'activeUsers': {
                    'sum': 12500,
                    'mean': 138.89,
                    'median': 135.0,
                    'min': 80,
                    'max': 210
                },
                'sessions': {
                    'sum': 18750,
                    'mean': 208.33,
                    'median': 205.0,
                    'min': 120,
                    'max': 320
                },
                'screenPageViews': {
                    'sum': 45000,
                    'mean': 500.0,
                    'median': 490.0,
                    'min': 300,
                    'max': 750
                }
            }
        },
        'sample_data': [
            {'date': '2023-01-01', 'deviceCategory': 'desktop', 'activeUsers': 100, 'sessions': 150, 'screenPageViews': 400},
            {'date': '2023-01-01', 'deviceCategory': 'mobile', 'activeUsers': 80, 'sessions': 120, 'screenPageViews': 300},
            {'date': '2023-01-01', 'deviceCategory': 'tablet', 'activeUsers': 20, 'sessions': 30, 'screenPageViews': 80},
            {'date': '2023-01-02', 'deviceCategory': 'desktop', 'activeUsers': 110, 'sessions': 165, 'screenPageViews': 420},
            {'date': '2023-01-02', 'deviceCategory': 'mobile', 'activeUsers': 85, 'sessions': 130, 'screenPageViews': 320},
            {'date': '2023-01-02', 'deviceCategory': 'tablet', 'activeUsers': 25, 'sessions': 35, 'screenPageViews': 90}
        ]
    }
    
    # Create the insight generator (using OpenAI in this example)
    generator = AnalyticsInsightGenerator(
        llm_provider='openai',
        api_key='your_api_key_here',  # Replace with your API key
        model='gpt-4'
    )
    
    # Custom prompt
    custom_prompt = """
You are the CMO of a company reviewing Google Analytics data. 
Write a brief executive summary of the key findings from this data.
Focus on business impact and strategic recommendations.

Data Summary:
{data_summary}

Sample Data:
{sample_data}

Your executive summary should be concise, strategic, and actionable.
"""
    
    # Generate custom analysis
    analysis = generator.generate_custom_analysis(
        data=sample_data,
        custom_prompt=custom_prompt,
        system_prompt="You are a strategic marketing executive with 20 years of experience."
    )
    
    print("Custom Prompt Analysis:")
    print(analysis)
    
    return analysis

def main():
    """Main function to demonstrate the LLM integration."""
    # This is just an example - in a real application, you would integrate this
    # with your Flask/FastAPI backend and database
    
    print("LLM Integration for Google Analytics Example")
    print("-------------------------------------------")
    
    # Uncomment and modify these lines to run the examples
    # analysis1 = example_openai_analysis()
    # analysis2 = example_anthropic_analysis()
    # analysis3 = example_custom_prompt()
    
    print("\nNote: This is a demonstration module. In a real application,")
    print("you would need to provide valid API keys and integrate with GA data.")

if __name__ == "__main__":
    main()
