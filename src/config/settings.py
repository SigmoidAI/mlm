"""
Configuration settings for the Multi-Agent System
"""

import os

# Azure OpenAI Configuration
#used by valerian for test

# AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
# AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
# AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
# AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "GPT4o")

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Baseline_Benchmarks")

# System Configuration
DEFAULT_SYSTEM_PROMPT = "You are a precise technical assistant. Break down your answer into clear arguments with reasoning."
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
