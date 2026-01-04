"""
Configuration Loader
Loads configuration from YAML file and environment variables
"""
import yaml
import os
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    config_file = Path(__file__).parent.parent.parent / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if present
    if os.getenv("HOPSWORKS_API_KEY"):
        config.setdefault("hopsworks", {})["api_key"] = os.getenv("HOPSWORKS_API_KEY")
    if os.getenv("HOPSWORKS_PROJECT_NAME"):
        config.setdefault("hopsworks", {})["project_name"] = os.getenv("HOPSWORKS_PROJECT_NAME")
    
    return config

