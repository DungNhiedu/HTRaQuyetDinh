# Stock Market Index Prediction using Fusion of Machine Learning Techniques

## Mô tả Dự án

Dự án này triển khai hệ thống dự báo chỉ số thị trường chứng khoán sử dụng kết hợp nhiều kỹ thuật học máy (Machine Learning Fusion). Hệ thống tích hợp các mô hình khác nhau để tạo ra dự báo chính xác hơn thông qua ensemble methods.

## Tính năng chính

### 1. Thu thập và Xử lý Dữ liệu
- Thu thập dữ liệu thị trường chứng khoán real-time từ Yahoo Finance
- Xử lý và làm sạch dữ liệu lịch sử
- Tính toán các chỉ báo kỹ thuật (Technical Indicators)
- Feature engineering cho dữ liệu time series

### 2. Mô hình Machine Learning
- **Random Forest**: Ensemble của decision trees
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Gradient boosting nhanh và hiệu quả
- **LSTM Neural Networks**: Deep learning cho time series
- **Support Vector Regression (SVR)**: SVM cho regression
- **ARIMA**: Time series forecasting truyền thống

### 3. Fusion Techniques
- **Voting Regressor**: Kết hợp predictions từ nhiều models
- **Stacking**: Meta-model học cách combine predictions
- **Weighted Average**: Trọng số động dựa trên performance
- **Bayesian Model Averaging**: Kết hợp theo xác suất

### 4. Giao diện Demo
- **Streamlit Web App**: Interactive dashboard
- **Real-time Data Visualization**: Charts và graphs
- **Model Performance Metrics**: Đánh giá accuracy
- **Prediction Interface**: Dự báo tương tác

## Cài đặt

```bash
# Clone repository
git clone <repository-url>
cd python_project_template

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc sử dụng pip install với pyproject.toml
pip install -e .
```

## Sử dụng

### 1. Chạy Demo Web App
```bash
streamlit run src/stock_predictor/app.py
```

### 2. Sử dụng Command Line Interface
```bash
# Dự báo chỉ số VN-Index
stock-predict predict --symbol ^VNI --days 30

# Training mô hình mới
stock-predict train --symbol ^VNI --periods 1000
```

### 3. Sử dụng Python API
```python
from stock_predictor import StockPredictor

# Khởi tạo predictor
predictor = StockPredictor()

# Load dữ liệu
data = predictor.load_data("^VNI", periods=1000)

# Training models
predictor.train_models(data)

# Dự báo
predictions = predictor.predict(days=30)
```

## Cấu trúc Dự án

```
src/stock_predictor/
├── __init__.py
├── main.py                 # Entry point chính
├── cli.py                  # Command line interface
├── app.py                  # Streamlit web application
├── data/
│   ├── __init__.py
│   ├── collector.py        # Thu thập dữ liệu
│   ├── preprocessor.py     # Xử lý dữ liệu
│   └── features.py         # Feature engineering
├── models/
│   ├── __init__.py
│   ├── base_model.py       # Base class cho models
│   ├── traditional.py      # Random Forest, XGBoost, etc.
│   ├── deep_learning.py    # LSTM, Neural Networks
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

## Algorithms và Techniques

### Machine Learning Models
1. **Random Forest Regressor**: Ensemble của decision trees với random sampling
2. **XGBoost**: Extreme Gradient Boosting với regularization
3. **LightGBM**: Light Gradient Boosting Machine với leaf-wise growth
4. **LSTM**: Long Short-Term Memory networks cho sequential data
5. **SVR**: Support Vector Regression với RBF kernel
6. **ARIMA**: AutoRegressive Integrated Moving Average

### Fusion Techniques
1. **Simple Voting**: Average predictions từ tất cả models
2. **Weighted Voting**: Trọng số dựa trên validation performance
3. **Stacking**: Meta-learner (Linear Regression) học cách combine
4. **Dynamic Weighting**: Trọng số thay đổi theo thời gian

### Feature Engineering
1. **Technical Indicators**: RSI, MACD, Bollinger Bands, MA
2. **Price Features**: Returns, volatility, price ratios
3. **Volume Features**: Volume trends, volume-price relationship
4. **Time Features**: Day of week, month, quarter effects
5. **Lag Features**: Historical price movements

## Metrics Đánh giá

- **MAE (Mean Absolute Error)**: Sai số tuyệt đối trung bình
- **RMSE (Root Mean Square Error)**: Căn bậc hai sai số bình phương
- **MAPE (Mean Absolute Percentage Error)**: Sai số phần trăm tuyệt đối
- **Directional Accuracy**: Độ chính xác dự báo hướng giá
- **Sharpe Ratio**: Đánh giá risk-adjusted returns

## Demo Features

1. **Interactive Dashboard**: Real-time data và predictions
2. **Model Comparison**: So sánh performance các models
3. **Feature Importance**: Visualization của feature importance
4. **Prediction Confidence**: Confidence intervals cho predictions
5. **Historical Backtesting**: Test performance trên dữ liệu lịch sử

## Tác giả

Dương Thị Ngọc Dung - 24210015

## License

MIT License