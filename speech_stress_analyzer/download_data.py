# speech_stress_analyzer/download_data.py

import dvc.api
import os
from pathlib import Path
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)

def download_data(cfg: DictConfig = None):
    """
    Download data from DVC remote storage.
    
    Args:
        cfg: Hydra configuration (optional)
    """
    try:
        logger.info("Starting data download from DVC remote...")
        
        project_root = Path.cwd()
        raw_data_dir = project_root / "raw_data"
        
        raw_data_dir.mkdir(exist_ok=True)
        
        audio_file = raw_data_dir / "audio.mp3"
        text_file = raw_data_dir / "text.txt"
        
        if not audio_file.exists() or not text_file.exists():
            logger.info("Pulling data from DVC remote...")
            os.system("poetry run dvc pull")
            logger.info("Data successfully downloaded from DVC remote")
        else:
            logger.info("Data files already exist locally")
            
        if audio_file.exists() and text_file.exists():
            logger.info(f"Audio file: {audio_file} ({audio_file.stat().st_size / 1024 / 1024:.2f} MB)")
            logger.info(f"Text file: {text_file} ({text_file.stat().st_size} bytes)")
        else:
            raise FileNotFoundError("Failed to download required data files")
            
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise

def download_sample_data_from_web():
    """
    Alternative function to download sample data from web sources
    if DVC remote is not available.
    """
    logger.info("DVC remote not configured. Please set up your data manually:")
    logger.info("1. Place audio file at raw_data/audio.mp3")
    logger.info("2. Place transcript at raw_data/text.txt")
    logger.info("3. Run 'poetry run dvc add raw_data/audio.mp3 raw_data/text.txt'")
    logger.info("4. Run 'poetry run dvc push' to upload to remote")

if __name__ == "__main__":
    download_data() 