"""
Hopsworks Client Utilities
Handles interactions with Hopsworks Feature Store and Model Registry
"""
import hopsworks
import pandas as pd
from typing import Optional, Dict
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


def _load_hopsworks_config():
    """Load Hopsworks config from config.yaml as fallback"""
    try:
        from config.config_loader import load_config
        config = load_config()
        hopsworks_config = config.get("hopsworks", {})
        return {
            "project_name": hopsworks_config.get("project_name"),
            "host": hopsworks_config.get("host")
        }
    except Exception:
        return {"project_name": None, "host": None}


class HopsworksClient:
    """Client for interacting with Hopsworks Feature Store and Model Registry"""
    
    def __init__(self, api_key: Optional[str] = None, project_name: Optional[str] = None, host: Optional[str] = None):
        # Try environment variables first, then config.yaml
        config_fallback = _load_hopsworks_config()
        
        self.api_key = api_key or os.getenv("HOPSWORKS_API_KEY")
        self.project_name = project_name or os.getenv("HOPSWORKS_PROJECT_NAME") or config_fallback.get("project_name")
        self.host = host or os.getenv("HOPSWORKS_HOST") or config_fallback.get("host")
        
        if not self.api_key or not self.project_name:
            error_msg = (
                "Hopsworks API key and project name must be provided.\n"
                f"  - API Key found: {bool(self.api_key)}\n"
                f"  - Project Name found: {bool(self.project_name)}\n"
                f"  - Looking for .env file at: {env_path}\n"
                f"  - .env file exists: {env_path.exists()}\n"
                "\n"
                "Options to provide credentials:\n"
                "1. Set environment variables:\n"
                "   HOPSWORKS_API_KEY=your_api_key_here\n"
                "   HOPSWORKS_PROJECT_NAME=your_project_name (optional if in config.yaml)\n"
                "   HOPSWORKS_HOST=c.app.hopsworks.ai (optional)\n"
                "\n"
                "2. Add to config.yaml under 'hopsworks' section:\n"
                "   hopsworks:\n"
                "     project_name: your_project_name\n"
                "     host: c.app.hopsworks.ai\n"
                "\n"
                "3. Create .env file in project root with:\n"
                "   HOPSWORKS_API_KEY=your_api_key_here\n"
                "   HOPSWORKS_PROJECT_NAME=your_project_name\n"
                "   HOPSWORKS_HOST=c.app.hopsworks.ai (optional)"
            )
            raise ValueError(error_msg)
        
        # Build login parameters
        login_params = {
            "api_key_value": self.api_key,
            "project": self.project_name
        }
        
        if self.host:
            login_params["host"] = self.host
        
        self.project = hopsworks.login(**login_params)
        self.fs = self.project.get_feature_store()
        self.mr = self.project.get_model_registry()
    
    def get_or_create_feature_group(
        self, 
        name: str, 
        version: int = 1,
        description: str = "",
        primary_key: list = None
    ):
        """Get existing feature group or create a new one"""
        try:
            fg = self.fs.get_feature_group(name=name, version=version)
            
            if fg is None:
                raise ValueError(f"get_feature_group returned None for {name} v{version}")
            
            logger.info(f"Retrieved existing feature group: {name} v{version}")
            return fg
        except Exception as e:
            logger.info(f"Feature group not found, creating new one: {e}")
            try:
                pk_list = primary_key or ["game_id"]
                if isinstance(pk_list, list) and len(pk_list) > 0:
                    pk_list = [str(pk) if not isinstance(pk, str) else pk for pk in pk_list]
                
                fg = self.fs.create_feature_group(
                    name=name,
                    version=version,
                    description=description,
                    primary_key=pk_list,
                    online_enabled=True,
                )
                
                if fg is None:
                    raise ValueError(f"create_feature_group returned None for {name} v{version}")
                
                return fg
            except Exception as e2:
                logger.error(f"Failed to create feature group: {e2}")
                raise
    
    def insert_features(self, feature_group, df: pd.DataFrame):
        """Insert features into feature group"""
        # Ensure game_id is int64 for primary key compatibility
        if 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype('int64')
        
        # Use insert with mode='upsert' to update existing records
        feature_group.insert(df, write_options={"start_offline_backfill": False})
        logger.info(f"Inserted/updated {len(df)} rows into feature group")
    
    def get_features(self, feature_group, filters: Optional[Dict] = None) -> pd.DataFrame:
        """Retrieve features from feature group"""
        # Try online feature store first (faster and more reliable)
        # Then fallback to offline methods if needed
        read_methods = [
            ("online feature store", lambda: feature_group.read(online=True)),
            ("direct read", lambda: feature_group.read()),
            ("read with data_format", lambda: feature_group.read(data_format="pandas")),
            ("query read", lambda: feature_group.select_all().read()),
        ]
        
        for method_name, read_func in read_methods:
            try:
                logger.info(f"Trying {method_name}...")
                result = read_func()
                
                # Apply filters in Python if needed
                if filters:
                    for key, value in filters.items():
                        result = result[result[key] == value]
                
                logger.info(f"Successfully read {len(result)} rows using {method_name}")
                return result
            except Exception as e:
                logger.debug(f"{method_name} failed: {e}")
                continue
        
        # All methods failed - this is a schema/version compatibility issue
        logger.error("="*80)
        logger.error("CANNOT READ FROM FEATURE GROUP - Schema/Version Mismatch")
        logger.error("="*80)
        logger.error("This is likely due to Hopsworks client 4.6.0 vs backend 4.2.2 mismatch.")
        logger.error("")
        logger.error("SOLUTION (choose one):")
        logger.error("")
        logger.error("Option 1: Delete and recreate feature group (RECOMMENDED)")
        logger.error("  1. Go to Hopsworks UI -> Feature Store")
        logger.error(f"  2. Delete feature group: {feature_group.name} v{feature_group.version}")
        logger.error("  3. Re-run feature pipeline to recreate it")
        logger.error("")
        logger.error("Option 2: Downgrade Hopsworks client")
        logger.error("  pip install 'hopsworks==4.2.*'")
        logger.error("")
        logger.error("="*80)
        
        raise Exception(
            f"All read methods failed for feature group {feature_group.name} v{feature_group.version}. "
            "This is a schema/version compatibility issue. Please delete the feature group in "
            "Hopsworks UI and re-run the feature pipeline to recreate it with the correct schema."
        )
    
    def register_model(
        self,
        model,
        model_name: str,
        description: str = "",
        metrics: Optional[Dict] = None
    ):
        """Register a trained model in the Model Registry"""
        model_dir = f"models/{model_name}"
        
        # Save model locally first
        import joblib
        import os
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/model.pkl"
        joblib.dump(model, model_path)
        
        # Create model registry entry
        mr_model = self.mr.python.create_model(
            name=model_name,
            description=description,
            metrics=metrics or {}
        )
        
        # Upload model
        mr_model.save(model_path)
        
        logger.info(f"Registered model: {model_name}")
        return mr_model
    
    def get_latest_model(self, model_name: str):
        """Get the latest version of a model from Model Registry"""
        try:
            model = self.mr.get_model(model_name, version=None)
            return model
        except Exception as e:
            logger.error(f"Error retrieving model {model_name}: {e}")
            raise
    
    def get_model(self, model_name: str, version: int = None):
        """Get a specific version of a model from Model Registry, or latest if version is None"""
        try:
            if version is not None:
                logger.info(f"Retrieving model {model_name} version {version}")
                model = self.mr.get_model(model_name, version=version)
            else:
                logger.info(f"Retrieving latest version of model {model_name}")
                model = self.mr.get_model(model_name, version=None)
            return model
        except Exception as e:
            logger.error(f"Error retrieving model {model_name} version {version}: {e}")
            raise
    
    def download_model(self, model_registry_entry, download_path: str = None, overwrite: bool = True) -> str:
        """Download model from Model Registry and return local path"""
        import os
        import shutil
        if download_path is None:
            download_path = f"models/{model_registry_entry.name}"
        
        if overwrite and os.path.exists(download_path):
            logger.info(f"Removing existing model directory: {download_path}")
            shutil.rmtree(download_path)
        
        os.makedirs(download_path, exist_ok=True)
        
        try:
            model_registry_entry.download(download_path, overwrite=overwrite)
        except TypeError:
            model_registry_entry.download(download_path)
        
        # Find the model file (could be .pkl, .joblib, etc.)
        model_file = None
        for file in os.listdir(download_path):
            if file.endswith(('.pkl', '.joblib', '.h5')):
                model_file = os.path.join(download_path, file)
                break
        
        if model_file is None:
            raise FileNotFoundError(f"No model file found in {download_path}")
        
        logger.info(f"Model downloaded to: {model_file}")
        return model_file

