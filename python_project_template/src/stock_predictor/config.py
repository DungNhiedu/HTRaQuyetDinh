"""
Cấu hình cho Stock Market Prediction System
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class ModelConfig:
    """Cấu hình cho các mô hình machine learning"""
    
    # Random Forest
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
    RF_RANDOM_STATE = 42
    
    # XGBoost
    XGB_N_ESTIMATORS = 100
    XGB_MAX_DEPTH = 6
    XGB_LEARNING_RATE = 0.1
    XGB_RANDOM_STATE = 42
    
    # LightGBM
    LGB_N_ESTIMATORS = 100
    LGB_MAX_DEPTH = 6
    LGB_LEARNING_RATE = 0.1
    LGB_RANDOM_STATE = 42
    
    # LSTM
    LSTM_UNITS = 50
    LSTM_DROPOUT = 0.2
    LSTM_EPOCHS = 100
    LSTM_BATCH_SIZE = 32
    LSTM_SEQUENCE_LENGTH = 60
    
    # SVR
    SVR_C = 1.0
    SVR_GAMMA = 'scale'
    SVR_KERNEL = 'rbf'

@dataclass
class DataConfig:
    """Cấu hình cho thu thập và xử lý dữ liệu"""
    
    # Symbols để dự báo
    DEFAULT_SYMBOLS = ['^VNI', '^GSPC', '^DJI', '^IXIC']
    
    # Thời gian dữ liệu
    DEFAULT_PERIOD = '2y'  # 2 years
    DEFAULT_INTERVAL = '1d'  # daily
    
    # Feature engineering
    TECHNICAL_INDICATORS = [
        'RSI', 'MACD', 'BB_upper', 'BB_lower', 'BB_middle',
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'Volume_SMA'
    ]
    
    # Train/validation/test split
    TRAIN_RATIO = 0.7
    VALIDATION_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Features để sử dụng
    PRICE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
    LAG_PERIODS = [1, 2, 3, 5, 10]

@dataclass  
class EvaluationConfig:
    """Cấu hình cho đánh giá mô hình"""
    
    METRICS = ['MAE', 'RMSE', 'MAPE', 'R2', 'DIRECTIONAL_ACCURACY']
    CROSS_VALIDATION_FOLDS = 5
    
    # Thresholds
    GOOD_MAPE_THRESHOLD = 5.0  # %
    GOOD_DIRECTIONAL_ACCURACY = 0.6  # 60%

@dataclass
class AppConfig:
    """Cấu hình cho ứng dụng"""
    
    # Streamlit
    PAGE_TITLE = "Market Prediction System"
    PAGE_ICON = "📈"
    LAYOUT = "wide"
    
    # Paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    RESULTS_DIR = "results"
    
    # Cache
    CACHE_TTL = 3600  # 1 hour in seconds

# Khởi tạo configs
model_config = ModelConfig()
data_config = DataConfig()
evaluation_config = EvaluationConfig()
app_config = AppConfig()

# Tạo directories nếu chưa có
for directory in [app_config.DATA_DIR, app_config.MODELS_DIR, app_config.RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)