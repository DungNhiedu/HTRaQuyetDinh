# Stock Market Index Prediction System

## English

A comprehensive machine learning system for predicting stock market movements using technical indicators and traditional ML algorithms. This project implements a complete pipeline from data preprocessing to feature engineering, based on proven financial analysis techniques.

### Features

- **Data Preprocessing**: Automated data cleaning, normalization, and return calculation
- **Technical Indicators**: 
  - Simple Moving Average (SMA) and Exponential Moving Average (EMA)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - ATR (Average True Range)
  - OBV (On Balance Volume)
- **Feature Engineering**: Comprehensive technical analysis using the `ta` library
- **Machine Learning Models**: Traditional ML algorithms (Random Forest, XGBoost, SVM)
- **Interactive Demo**: Streamlit web application with sample data and CSV upload
- **Visualization**: Advanced charting with Plotly for technical analysis

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

#### Demo Usage

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
- Sample data demonstration
- Technical indicators visualization
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

---

## Tiếng Việt

Hệ thống học máy toàn diện để dự báo chuyển động thị trường chứng khoán sử dụng các chỉ báo kỹ thuật và thuật toán ML truyền thống. Dự án triển khai pipeline hoàn chỉnh từ tiền xử lý dữ liệu đến kỹ thuật tạo đặc trưng, dựa trên các kỹ thuật phân tích tài chính đã được chứng minh.

### Tính năng chính

- **Tiền xử lý Dữ liệu**: Tự động làm sạch, chuẩn hóa và tính toán tỷ suất sinh lời
- **Chỉ báo Kỹ thuật**:
  - Đường trung bình động đơn giản (SMA) và hàm mũ (EMA)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - ATR (Average True Range)
  - OBV (On Balance Volume)
- **Kỹ thuật Tạo Đặc trưng**: Phân tích kỹ thuật toàn diện sử dụng thư viện `ta`
- **Mô hình Học máy**: Các thuật toán ML truyền thống (Random Forest, XGBoost, SVM)
- **Demo Tương tác**: Ứng dụng web Streamlit với dữ liệu mẫu và tải lên CSV
- **Trực quan hóa**: Biểu đồ nâng cao với Plotly cho phân tích kỹ thuật

### Bắt đầu nhanh

#### Cài đặt

```bash
# Điều hướng đến thư mục dự án
cd python_project_template

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

#### Sử dụng Demo

Chạy script demo để kiểm tra pipeline:

```bash
python demo_reference_implementation.py
```

#### Giao diện Web

Khởi chạy dashboard Streamlit:

```bash
streamlit run src/stock_predictor/app.py
```

Ứng dụng web bao gồm:
- Demo dữ liệu mẫu
- Trực quan hóa chỉ báo kỹ thuật
- Tải lên file CSV để phân tích dữ liệu tùy chỉnh
- Showcase kỹ thuật tạo đặc trưng

#### Sử dụng cơ bản

```python
from stock_predictor.data.preprocessor import DataPreprocessor
from stock_predictor.data.features import FeatureEngineer

# Khởi tạo components
preprocessor = DataPreprocessor()
feature_engineer = FeatureEngineer()

# Xử lý dữ liệu
processed_data = preprocessor.prepare_data(raw_data)
features = feature_engineer.create_features(processed_data)
```

## Project Structure

```
src/stock_predictor/
├── __init__.py
├── main.py                 # Entry point chính
├── cli.py                  # Command line interface
├── app.py                  # Streamlit web application
├── data/
│   ├── __init__.py
│   ├── collector.py        # Thu thập dữ liệu
│   ├── preprocessor.py     # Xử lý dữ liệu (Updated)
│   └── features.py         # Feature engineering (Updated)
├── models/
│   ├── __init__.py
│   ├── base_model.py       # Base class cho models
│   ├── traditional.py      # Random Forest, XGBoost, etc.
│   ├── deep_learning.py    # LSTM, Neural Networks (Disabled)
│   ├── ensemble.py         # Fusion techniques
│   └── arima.py           # ARIMA time series
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Plotting và charts
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

## License

MIT License
