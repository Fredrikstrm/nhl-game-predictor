"""
Test script to verify connections to NHL API and Hopsworks
"""
import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent.parent))

from utils.nhl_api_client import NHLAPIClient
from utils.hopsworks_client import HopsworksClient
from config.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_nhl_api():
    """Test NHL API connection"""
    logger.info("Testing NHL API connection...")
    try:
        config = load_config()
        client = NHLAPIClient(
            base_url=config['nhl_api']['base_url'],
            timeout=config['nhl_api']['timeout']
        )
        
        # Test getting teams
        teams = client.get_teams()
        logger.info(f"✓ NHL API connection successful! Found {len(teams)} teams")
        return True
    except Exception as e:
        logger.error(f"✗ NHL API connection failed: {e}")
        return False


def test_hopsworks():
    """Test Hopsworks connection"""
    logger.info("Testing Hopsworks connection...")
    try:
        client = HopsworksClient()
        logger.info(f"✓ Hopsworks connection successful! Project: {client.project_name}")
        return True
    except Exception as e:
        logger.error(f"✗ Hopsworks connection failed: {e}")
        logger.error("Make sure you have set HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME in .env")
        return False


def main():
    """Run all connection tests"""
    logger.info("=" * 60)
    logger.info("Connection Tests")
    logger.info("=" * 60)
    
    nhl_ok = test_nhl_api()
    hopsworks_ok = test_hopsworks()
    
    logger.info("=" * 60)
    if nhl_ok and hopsworks_ok:
        logger.info("✓ All connections successful!")
        return 0
    else:
        logger.error("✗ Some connections failed. Please check your configuration.")
        return 1


if __name__ == "__main__":
    exit(main())

