"""
Download and extract the Household Electric Power Consumption dataset.
"""

import os
import urllib.request
import zipfile
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset URL
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"

def create_dirs(raw_dir, processed_dir):
    """Create necessary directories if they don't exist."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    logger.info(f"Created directories: {raw_dir}, {processed_dir}")

def download_dataset(url, raw_dir):
    """Download the dataset from the given URL."""
    zip_path = os.path.join(raw_dir, "household_power_consumption.zip")
    
    if os.path.exists(zip_path):
        logger.info(f"Dataset already downloaded at {zip_path}")
        return zip_path
    
    logger.info(f"Downloading dataset from {url}")
    urllib.request.urlretrieve(url, zip_path)
    logger.info(f"Dataset downloaded to {zip_path}")
    
    return zip_path

def extract_dataset(zip_path, raw_dir):
    """Extract the dataset zip file."""
    logger.info(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
    logger.info(f"Dataset extracted to {raw_dir}")

def main(raw_dir, processed_dir):
    """Main function to download and extract the dataset."""
    # Create directories
    create_dirs(raw_dir, processed_dir)
    
    # Download dataset
    zip_path = download_dataset(DATASET_URL, raw_dir)
    
    # Extract dataset
    extract_dataset(zip_path, raw_dir)
    
    logger.info("Download and extraction complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download energy consumption dataset.")
    parser.add_argument(
        "--raw_dir", 
        type=str, 
        default="../../data/raw",
        help="Directory for storing raw data"
    )
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        default="../../data/processed",
        help="Directory for storing processed data"
    )
    
    args = parser.parse_args()
    main(args.raw_dir, args.processed_dir) 