# Stock Market Index Prediction System

## 🚀 Quick Start

**👥 Mới sử dụng? Xem hướng dẫn chi tiết:** [GETTING_STARTED.md](GETTING_STARTED.md)

**📊 Chạy ngay:** `streamlit run src/stock_predictor/app.py`

---

## English

A comprehensive machine learning system for predicting stock market movements using technical indicators and traditional ML algorithms. This project implements a complete pipeline from data preprocessing to feature engineering, based on proven financial analysis techniques.

### Key Features

- **Data Preprocessing**: Automatic cleaning, normalization, and return calculation
- **Technical Indicators**:
  - Simple (SMA) and Exponential Moving Averages (EMA)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - ATR (Average True Range)
  - OBV (On Balance Volume)
- **Feature Engineering**: Comprehensive technical analysis using `ta` library
- **Machine Learning Models**: Traditional ML algorithms (Random Forest, XGBoost, SVM)
- **Interactive Demo**: Streamlit web app with sample data and CSV upload
- **Visualization**: Advanced plotting with Plotly for technical analysis

### Quick Start

#### Installation

```bash
# Navigate to project directory
cd python_project_template

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Running Demo

Run the demo script to test the pipeline:

```bash
python demo_reference_implementation.py
```

#### Web Interface

Launch the Streamlit dashboard:

```bash
streamlit run src/stock_predictor/app.py
```

The web app includes:
- Sample data demo
- Technical indicator visualizations
- CSV file upload for custom data analysis
- Feature engineering showcase

#### Basic Usage

```python
from stock_predictor.data.preprocessor import DataPreprocessor
from stock_predictor.data.features import FeatureEngineer

# Initialize components
preprocessor = DataPreprocessor()
feature_engineer = FeatureEngineer()

# Process data
processed_data = preprocessor.prepare_data(raw_data)
features = feature_engineer.create_features(processed_data)
```

## Project Structure

```
src/stock_predictor/
├── __init__.py
├── main.py                 # Main entry point
├── cli.py                  # Command line interface
├── app.py                  # Streamlit web application
├── data/
│   ├── __init__.py
│   ├── collector.py        # Data collection
│   ├── preprocessor.py     # Data preprocessing (Updated)
│   └── features.py         # Feature engineering (Updated)
├── models/
│   ├── __init__.py
│   ├── base_model.py       # Base class for models
│   ├── traditional.py      # Random Forest, XGBoost, etc.
│   ├── deep_learning.py    # LSTM, Neural Networks (Disabled)
│   ├── ensemble.py         # Fusion techniques
│   └── arima.py           # ARIMA time series
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Plotting and charts
└── utils/
    ├── __init__.py
    ├── config.py           # Configuration
    └── helpers.py          # Utility functions
```

## Technical Implementation

### Data Preprocessing Pipeline (Based on Reference Implementation)

1. **Data Loading**: Load CSV files with stock data (code, year, month, day, OHLCV)
2. **Date Processing**: Convert year/month/day to datetime format
3. **Data Cleaning**: Handle missing values and outliers
4. **Return Calculation**: Calculate daily returns and classify targets
5. **Data Merging**: Combine multiple datasets if needed

### Feature Engineering (Using TA Library)

1. **Technical Indicators**:
   - **SMA**: Simple Moving Average (5, 10, 20 periods)
   - **EMA**: Exponential Moving Average (12, 26 periods)
   - **MACD**: Moving Average Convergence Divergence
   - **RSI**: Relative Strength Index (14 periods)
   - **Bollinger Bands**: Upper, Middle, Lower bands
   - **ATR**: Average True Range (14 periods)
   - **OBV**: On Balance Volume

2. **Price Features**:
   - High-Low spread
   - Open-Close spread
   - Volume-weighted prices

### Machine Learning Models

1. **Random Forest**: Ensemble decision trees with feature importance
2. **XGBoost**: Gradient boosting with regularization
3. **SVM**: Support Vector Machine for classification/regression
4. **Traditional Time Series**: ARIMA models

### Evaluation Metrics

- **Accuracy**: Classification accuracy for direction prediction
- **Precision/Recall**: For trend classification
- **MAE/RMSE**: For price prediction
- **Directional Accuracy**: Correct prediction of price movement direction

## Files Updated in Reference Implementation

1. **`data/preprocessor.py`**: Complete rewrite based on reference notebook
2. **`data/features.py`**: Integrated TA library technical indicators
3. **`models/deep_learning.py`**: Disabled TensorFlow/Keras (Python 3.13 compatibility)
4. **`app.py`**: New Streamlit interface with sample data and visualization
5. **`demo_reference_implementation.py`**: Standalone demo script

## Dependencies

All required packages are listed in `requirements.txt`:

- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `ta>=0.10.0` (Technical Analysis Library)
- `plotly>=5.0.0` (Interactive Charts)
- `streamlit>=1.25.0` (Web Interface)
- `matplotlib>=3.4.0`
- `seaborn>=0.11.0`

## Known Issues

- **TensorFlow/Keras**: Currently disabled due to Python 3.13 compatibility issues
- **Deep Learning Models**: Will be re-enabled when TensorFlow supports Python 3.13

## Author

Dương Thị Ngọc Dung - 24210015
Trần Đình Bảo - 24210010
Nguyễn Tiến Lộc - 24210044
Lê Hữu Phước - 24210066
Nguyễn Thị Thu Thanh - 24210082

## License

MIT License
