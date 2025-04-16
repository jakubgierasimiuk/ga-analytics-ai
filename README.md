# GA Analytics AI

An AI-powered Google Analytics data analysis application that automatically generates insights using LLM technology.

## Deployment Instructions

Follow these steps to deploy the application to Streamlit Cloud:

1. **Extract the contents** of this zip file to a local folder
2. **Create a GitHub repository** and upload all the files
3. **Rename `streamlit_cloud_app.py` to `app.py`** before uploading
4. **Connect your Streamlit Cloud account** to your GitHub repository
5. **Deploy the application** using `app.py` as the main file

For detailed instructions, refer to the `streamlit_deployment_guide.md` file.

## Configuration

After deployment, you'll need to:

1. Configure Google Analytics API access
2. Add your OpenAI API key in the application settings

## Files Included

- `ga_integration.py`: Google Analytics integration module
- `llm_integration.py`: LLM integration module
- `analysis_pipeline.py`: Data analysis pipeline
- `streamlit_app.py`: Main application file
- `app.py` (renamed from `streamlit_cloud_app.py`): Streamlit Cloud entry point
- `requirements.txt`: Required Python packages
- `.streamlit/config.toml`: Streamlit configuration
- `streamlit_deployment_guide.md`: Detailed deployment instructions

## Support

If you encounter any issues during deployment, refer to the troubleshooting section in the deployment guide.
