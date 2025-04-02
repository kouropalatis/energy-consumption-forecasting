import json
import os

# Define the notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Exploratory Data Analysis: Household Electric Power Consumption\n",
                "\n",
                "This notebook explores the Household Electric Power Consumption dataset from the UCI Machine Learning Repository.\n",
                "\n",
                "## Dataset Description\n",
                "- Measurements of electric power consumption in one household\n",
                "- Data collected every minute over almost 4 years\n",
                "- 7 different quantities measured\n",
                "\n",
                "## Analysis Goals\n",
                "1. Understand the data structure and quality\n",
                "2. Identify patterns and trends in power consumption\n",
                "3. Detect anomalies and outliers\n",
                "4. Prepare insights for feature engineering"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Import required libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import plotly.express as px\n",
                "from datetime import datetime\n",
                "\n",
                "# Set plotting style\n",
                "plt.style.use('seaborn')\n",
                "sns.set_palette('deep')\n",
                "%matplotlib inline"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Data Loading and Initial Inspection"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Read the dataset\n",
                "df = pd.read_csv('../data/raw/household_power_consumption.txt', \n",
                "                 sep=';', \n",
                "                 parse_dates={'datetime': ['Date', 'Time']},\n",
                "                 dayfirst=True,\n",
                "                 na_values=['?'])\n",
                "\n",
                "# Display basic information\n",
                "print(\"Dataset Shape:\", df.shape)\n",
                "print(\"\\nFirst few rows:\")\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Data Quality Assessment\n",
                "- Check for missing values\n",
                "- Identify outliers\n",
                "- Verify data types"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Check missing values\n",
                "print(\"Missing values:\")\n",
                "df.isnull().sum()\n",
                "\n",
                "# Display basic statistics\n",
                "print(\"\\nBasic statistics:\")\n",
                "df.describe()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Time Series Analysis\n",
                "- Daily patterns\n",
                "- Weekly patterns\n",
                "- Seasonal trends"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Create time-based features\n",
                "df['hour'] = df['datetime'].dt.hour\n",
                "df['day_of_week'] = df['datetime'].dt.dayofweek\n",
                "df['month'] = df['datetime'].dt.month\n",
                "\n",
                "# Plot daily power consumption pattern\n",
                "plt.figure(figsize=(12, 6))\n",
                "df.groupby('hour')['Global_active_power'].mean().plot()\n",
                "plt.title('Average Power Consumption by Hour')\n",
                "plt.xlabel('Hour of Day')\n",
                "plt.ylabel('Global Active Power (kilowatts)')\n",
                "plt.grid(True)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Feature Relationships\n",
                "- Correlation analysis\n",
                "- Distribution plots\n",
                "- Scatter plots"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Correlation heatmap\n",
                "plt.figure(figsize=(10, 8))\n",
                "sns.heatmap(df.select_dtypes(include=[np.number]).corr(), \n",
                "            annot=True, \n",
                "            cmap='coolwarm',\n",
                "            center=0)\n",
                "plt.title('Correlation Matrix')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Insights and Next Steps\n",
                "- Summary of findings\n",
                "- Data preprocessing requirements\n",
                "- Feature engineering suggestions"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.2"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Ensure the notebooks directory exists
os.makedirs('../notebooks', exist_ok=True)

# Write the notebook to a file
with open('../notebooks/1_exploratory_data_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook created successfully!") 