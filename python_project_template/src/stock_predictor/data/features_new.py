"""
Feature engineering module for stock market data.
Based on the reference implementation from copy_of_đồ_án_dss_nhóm_1.py using ta library.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các chỉ báo kỹ thuật vào DataFrame gốc, gồm:
    - SMA: Trung bình động đơn giản
    - EMA: Trung bình động lũy thừa
    - MACD: Đường trung bình động phân kỳ/hội tụ
    - RSI: Chỉ số sức mạnh tương đối
    - Bollinger Bands: Dải biến động giá
    - ATR: Đo độ biến động trung bình thực tế
    - OBV: Khối lượng cân bằng
    
    Args:
        df: DataFrame with columns ['code', 'year', 'month', 'day', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'return', 'target']
        
    Returns:
        DataFrame with technical indicators added
    """
    all_data = []

    for code in df['code'].unique():
        # Sắp xếp theo thời gian: year -> month -> day
        sub_df = df[df['code'] == code].sort_values(by=["year", "month", "day"]).copy()

        # --- 1. SMA (Simple Moving Average)
        # SMA = trung bình cộng giá đóng cửa trong n ngày
        sub_df['ma_5'] = ta.trend.sma_indicator(sub_df['close'], window=5).round(2)
        sub_df['ma_20'] = ta.trend.sma_indicator(sub_df['close'], window=20).round(2)

        # --- 2. EMA (Exponential Moving Average)
        # EMA = trung bình giá đóng cửa có trọng số, gần với hiện tại hơn
        sub_df['ema_12'] = ta.trend.ema_indicator(sub_df['close'], window=12).round(2)
        sub_df['ema_26'] = ta.trend.ema_indicator(sub_df['close'], window=26).round(2)

        # --- 3. MACD (Moving Average Convergence Divergence)
        # MACD = EMA(12) - EMA(26)
        # Signal = EMA của MACD
        sub_df['macd'] = ta.trend.macd(sub_df['close']).round(2)
        sub_df['macd_signal'] = ta.trend.macd_signal(sub_df['close']).round(2)

        # --- 4. RSI (Relative Strength Index)
        # RSI = đo độ mạnh/yếu xu hướng trong 14 ngày
        sub_df['rsi_14'] = ta.momentum.rsi(sub_df['close'], window=14).round(2)

        # --- 5. Bollinger Bands
        # Dải trên/dưới là MA ± 2 * độ lệch chuẩn
        bb = ta.volatility.BollingerBands(sub_df['close'], window=20, window_dev=2)
        sub_df['bb_bbm'] = bb.bollinger_mavg().round(2)   # MA
        sub_df['bb_bbh'] = bb.bollinger_hband().round(2)  # Upper band
        sub_df['bb_bbl'] = bb.bollinger_lband().round(2)  # Lower band

        # --- 6. ATR (Average True Range)
        # ATR = đo độ biến động giá trong 14 ngày
        sub_df['atr_14'] = ta.volatility.average_true_range(
            sub_df['high'], sub_df['low'], sub_df['close'], window=14
        ).round(2)

        # --- 7. OBV (On-Balance Volume)
        # OBV = tích lũy khối lượng nếu giá tăng, trừ nếu giá giảm
        sub_df['obv'] = ta.volume.on_balance_volume(sub_df['close'], sub_df['volume']).round(2)

        # Drop các dòng NA do rolling/window tính toán
        sub_df = sub_df.dropna().reset_index(drop=True)
        all_data.append(sub_df)

    result = pd.concat(all_data).reset_index(drop=True)
    return result


class FeatureEngineer:
    """Feature engineering class for creating technical indicators and derived features."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names = []
        
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators using the ta library.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        logger.info("Creating technical indicators...")
        result = add_technical_indicators(data)
        logger.info(f"Added technical indicators, shape: {result.shape}")
        return result
    
    def create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with price features
        """
        logger.info("Creating price features...")
        df = data.copy()
        
        # Price ratios
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            df['price_range'] = df['high'] - df['low']
            df['price_range_pct'] = df['price_range'] / df['close']
        
        # Price position within day's range
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['close_position'] = df['close_position'].fillna(0.5)  # Fill NaN when high=low
        
        return df
    
    def create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with volume features
        """
        logger.info("Creating volume features...")
        df = data.copy()
        
        if 'volume' in df.columns:
            # Volume moving averages
            df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            
            # Volume ratios
            df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
            df['volume_ratio_20'] = df['volume'] / df['volume_ma_20']
            
            # Volume-price features
            if 'close' in df.columns:
                df['volume_price'] = df['volume'] * df['close']
        
        return df
    
    def create_lag_features(self, data: pd.DataFrame, 
                          columns: List[str] = None,
                          lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """
        Create lagged features.
        
        Args:
            data: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        logger.info("Creating lag features...")
        df = data.copy()
        
        if columns is None:
            columns = ['close', 'volume', 'return']
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, data: pd.DataFrame,
                              columns: List[str] = None,
                              windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Args:
            data: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        logger.info("Creating rolling features...")
        df = data.copy()
        
        if columns is None:
            columns = ['close', 'volume']
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
        
        return df
    
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features using the pipeline.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with all features
        """
        logger.info("Creating all features...")
        
        # Start with technical indicators
        df = self.create_technical_indicators(data)
        
        # Add price features
        df = self.create_price_features(df)
        
        # Add volume features
        df = self.create_volume_features(df)
        
        # Add lag features
        df = self.create_lag_features(df)
        
        # Add rolling features
        df = self.create_rolling_features(df)
        
        # Drop rows with NaN values (from rolling/lag calculations)
        df = df.dropna()
        
        logger.info(f"Final feature matrix shape: {df.shape}")
        return df
