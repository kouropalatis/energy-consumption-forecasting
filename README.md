# Energy Consumption Forecasting

## Overview
This project develops time series forecasting models to predict energy consumption patterns, helping utilities and consumers optimize energy usage, reduce costs, and minimize environmental impact.

## Features
- Data preprocessing and cleaning of energy consumption data
- Exploratory data analysis with interactive visualizations
- Time series decomposition and anomaly detection
- Multiple forecasting models (ARIMA, Prophet, LSTM)
- Model evaluation and comparison
- Interactive dashboard for exploring predictions

## Project Structure
```
energy-consumption-forecasting/
│
├── data/                  # Data directory
│   ├── raw/               # Raw, immutable data
│   └── processed/         # Cleaned, processed data
│
├── notebooks/             # Jupyter notebooks for exploration and analysis
│
├── src/                   # Source code
│   ├── data/              # Data processing scripts
│   ├── models/            # Model implementations
│   └── visualization/     # Visualization code
│
├── docs/                  # Documentation
│
├── requirements.txt       # Project dependencies
└── README.md              # Project overview
```

## Installation
```bash
# Clone the repository
git clone https://github.com/kouropalatis/energy-consumption-forecasting.git
cd energy-consumption-forecasting

# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. Run the data preprocessing scripts to prepare the dataset
2. Execute the Jupyter notebooks to explore the data and examine model results
3. Use the prediction module to generate forecasts
4. View results in the interactive dashboard

## Models
- **ARIMA**: Classical time series forecasting approach
- **Prophet**: Facebook's time series forecasting tool
- **LSTM**: Deep learning approach for sequence prediction

## Data Sources
- Household Electric Power Consumption Dataset (UCI Machine Learning Repository)
- Additional sources to be added

## Results
(Will include visualizations and performance metrics once analysis is complete)

## Future Work
- Incorporate weather data as exogenous variables
- Implement multivariate forecasting
- Develop real-time prediction capability
- Expand to multiple household comparison

## License
MIT

## Contact
Konstantinos Lykostratis - lykostratisk@gmail.com 