#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exploratory Data Analysis: Household Electric Power Consumption

This script explores the Household Electric Power Consumption dataset from the UCI Machine Learning Repository.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import os

# Set plotting style
plt.style.use('seaborn-v0_8')  # Using a style that's available in Python 3.13
sns.set_palette('deep')

# Create output directory for plots
os.makedirs('../output/plots', exist_ok=True)

def load_data():
    """Load and prepare the dataset."""
    print("Loading dataset...")
    df = pd.read_csv('../data/raw/household_power_consumption.txt', 
                     sep=';', 
                     parse_dates={'datetime': ['Date', 'Time']},
                     dayfirst=True,
                     na_values=['?'])
    
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def assess_data_quality(df):
    """Assess the quality of the dataset."""
    print("\n=== Data Quality Assessment ===")
    
    # Check missing values
    print("Missing values:")
    print(df.isnull().sum())
    
    # Display basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

def create_time_features(df):
    """Create time-based features from the datetime column."""
    print("\n=== Creating Time Features ===")
    
    # Create time-based features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    return df

def analyze_time_patterns(df):
    """Analyze time-based patterns in the data."""
    print("\n=== Time Series Analysis ===")
    
    # Plot daily power consumption pattern
    plt.figure(figsize=(12, 6))
    df.groupby('hour')['Global_active_power'].mean().plot()
    plt.title('Average Power Consumption by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.grid(True)
    plt.savefig('../output/plots/daily_pattern.png')
    plt.close()
    
    # Plot weekly power consumption pattern
    plt.figure(figsize=(12, 6))
    df.groupby('day_of_week')['Global_active_power'].mean().plot(kind='bar')
    plt.title('Average Power Consumption by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.grid(True)
    plt.savefig('../output/plots/weekly_pattern.png')
    plt.close()
    
    # Plot monthly power consumption pattern
    plt.figure(figsize=(12, 6))
    df.groupby('month')['Global_active_power'].mean().plot(kind='bar')
    plt.title('Average Power Consumption by Month')
    plt.xlabel('Month')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.grid(True)
    plt.savefig('../output/plots/monthly_pattern.png')
    plt.close()
    
    return df

def analyze_feature_relationships(df):
    """Analyze relationships between features."""
    print("\n=== Feature Relationships ===")
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), 
                annot=True, 
                cmap='coolwarm',
                center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('../output/plots/correlation_matrix.png')
    plt.close()
    
    # Distribution plots for key features
    key_features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    
    for feature in key_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature].dropna(), kde=True)
        plt.title(f'Distribution of {feature}')
        plt.savefig(f'../output/plots/{feature}_distribution.png')
        plt.close()
    
    return df

def main():
    """Main function to run the exploratory data analysis."""
    print("Starting Exploratory Data Analysis...")
    
    # Load data
    df = load_data()
    
    # Assess data quality
    df = assess_data_quality(df)
    
    # Create time features
    df = create_time_features(df)
    
    # Analyze time patterns
    df = analyze_time_patterns(df)
    
    # Analyze feature relationships
    df = analyze_feature_relationships(df)
    
    print("\nExploratory Data Analysis complete!")
    print("Plots have been saved to the '../output/plots' directory.")

if __name__ == "__main__":
    main() 