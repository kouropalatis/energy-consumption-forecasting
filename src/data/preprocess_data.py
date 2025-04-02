"""
Preprocess the Household Electric Power Consumption dataset for time series analysis.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(raw_dir):
    """Load the raw dataset."""
    file_path = os.path.join(raw_dir, "household_power_consumption.txt")
    logger.info(f"Loading data from {file_path}")
    
    # Specify column names and parse dates
    col_names = [
        'Date', 'Time', 'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    
    # Load the data with proper parsing
    df = pd.read_csv(
        file_path,
        sep=';',
        header=0,
        names=col_names,
        parse_dates=[['Date', 'Time']],
        na_values=['?'],
        low_memory=False
    )
    
    # Rename the datetime column
    df.rename(columns={'Date_Time': 'timestamp'}, inplace=True)
    
    logger.info(f"Data loaded. Shape: {df.shape}")
    return df

def clean_data(df):
    """Clean the dataset by handling missing values and outliers."""
    logger.info("Starting data cleaning")
    
    # Initial count of missing values
    missing_before = df.isna().sum().sum()
    logger.info(f"Missing values before cleaning: {missing_before}")
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Handle missing values: interpolate for short gaps, forward fill for longer ones
    df = df.interpolate(method='time', limit=24)  # For gaps shorter than 24 hours
    df = df.fillna(method='ffill', limit=48)     # For any remaining gaps, forward fill
    
    # Drop any remaining rows with NaN values
    df = df.dropna()
    
    # Handle outliers: cap extreme values at 3 standard deviations
    for col in df.columns:
        if df[col].dtype != 'object':
            mean, std = df[col].mean(), df[col].std()
            df[col] = df[col].clip(lower=mean - 3*std, upper=mean + 3*std)
    
    # Final count of missing values
    missing_after = df.isna().sum().sum()
    logger.info(f"Missing values after cleaning: {missing_after}")
    logger.info(f"Data cleaned. Shape: {df.shape}")
    
    return df

def create_features(df):
    """Create time-based features for time series analysis."""
    logger.info("Creating time-based features")
    
    # Extract time components
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week
    
    # Create cyclical features for hour of day
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Create cyclical features for day of week
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    # Create cyclical features for month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Flag for weekends
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Calculate rolling statistics (7-day window)
    df['rolling_mean_7d'] = df['Global_active_power'].rolling(window=24*7).mean()
    df['rolling_std_7d'] = df['Global_active_power'].rolling(window=24*7).std()
    
    # Calculate lagged values (24 hours and 7 days)
    df['lag_24h'] = df['Global_active_power'].shift(24)
    df['lag_7d'] = df['Global_active_power'].shift(24*7)
    
    # Drop rows with NaN due to lagging/rolling
    df = df.dropna()
    
    logger.info(f"Features created. Shape: {df.shape}")
    return df

def resample_data(df, freq):
    """Resample the data to the specified frequency."""
    logger.info(f"Resampling data to {freq} frequency")
    
    # Define aggregation methods for each column
    agg_dict = {
        'Global_active_power': 'mean',
        'Global_reactive_power': 'mean',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'mean',
        'Sub_metering_2': 'mean',
        'Sub_metering_3': 'mean',
    }
    
    # Resample using the specified frequency
    resampled_df = df.resample(freq).agg(agg_dict)
    
    logger.info(f"Data resampled. Shape: {resampled_df.shape}")
    return resampled_df

def save_processed_data(df, resampled_df, processed_dir):
    """Save the processed datasets."""
    # Create the output directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save the full preprocessed dataset
    full_output_path = os.path.join(processed_dir, "household_power_consumption_processed.csv")
    df.to_csv(full_output_path)
    logger.info(f"Full processed data saved to {full_output_path}")
    
    # Save the resampled dataset
    resampled_output_path = os.path.join(processed_dir, "household_power_consumption_hourly.csv")
    resampled_df.to_csv(resampled_output_path)
    logger.info(f"Hourly resampled data saved to {resampled_output_path}")

def main(raw_dir, processed_dir):
    """Main function to preprocess the dataset."""
    # Load the data
    df = load_data(raw_dir)
    
    # Clean the data
    df = clean_data(df)
    
    # Create time-based features
    df = create_features(df)
    
    # Resample to hourly data
    resampled_df = resample_data(df, 'H')
    
    # Save the processed data
    save_processed_data(df, resampled_df, processed_dir)
    
    logger.info("Data preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess energy consumption dataset.")
    parser.add_argument(
        "--raw_dir", 
        type=str, 
        default="../../data/raw",
        help="Directory containing raw data"
    )
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        default="../../data/processed",
        help="Directory for storing processed data"
    )
    
    args = parser.parse_args()
    main(args.raw_dir, args.processed_dir) 