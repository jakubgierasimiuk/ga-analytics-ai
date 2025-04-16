"""
Streamlit Cloud Deployment Configuration

This file contains the necessary configuration for deploying the
Google Analytics AI Analyzer application to Streamlit Cloud.
"""

import streamlit as st
import os
import sys
import json
import datetime
import pandas as pd
from pathlib import Path
import base64
import tempfile
import uuid

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
try:
    from ga_integration import GoogleAnalyticsConnector, GADataProcessor
    from llm_integration import LLMFactory, AnalyticsInsightGenerator, PromptTemplate, AnalyticsPromptLibrary
    from analysis_pipeline import AnalysisPipeline, ReportGenerator, AnalysisWorkflow
except ImportError:
    st.error("Failed to import required modules. Please check that all files are present.")

# Import the main application
try:
    from streamlit_app import main
except ImportError:
    st.error("Failed to import main application. Please check that streamlit_app.py is present.")
    
# Create necessary directories
for directory in ['data', 'cache', 'reports', 'prompts', 'config']:
    os.makedirs(directory, exist_ok=True)

# Run the main application
if __name__ == "__main__":
    main()
