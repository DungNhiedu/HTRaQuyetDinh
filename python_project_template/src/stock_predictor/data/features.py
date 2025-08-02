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
        
        # Check if we have enough data for technical indicators
        # Adjust minimum requirements based on available data
        if len(sub_df) >= 20:
            # Full set of indicators
            min_periods_needed = {
                'ma_5': 5,
                'ma_20': 20,
                'ema_12': 12,
                'ema_26': 26,
                'rsi_14': 14,
                'macd': 26
            }
        elif len(sub_df) >= 14:
            # Reduced set for medium datasets
            min_periods_needed = {
                'ma_5': 5,
                'ma_10': 10,  # Use MA10 instead of MA20
                'ema_12': 12,
                'rsi_14': 14
            }
        elif len(sub_df) >= 5:
            # Minimal set for small datasets
            min_periods_needed = {
                'ma_5': 5
            }
        else:
            # Too small, just add empty columns
            logger.warning(f"Very limited data for {code}: {len(sub_df)} rows")
            sub_df['ma_5'] = np.nan
            sub_df['ma_20'] = np.nan
            sub_df['ema_12'] = np.nan
            sub_df['ema_26'] = np.nan
            sub_df['macd'] = np.nan
            sub_df['macd_signal'] = np.nan
            sub_df['rsi_14'] = np.nan
            sub_df['bb_bbm'] = np.nan
            sub_df['bb_bbh'] = np.nan
            sub_df['bb_bbl'] = np.nan
            sub_df['atr_14'] = np.nan
            sub_df['obv'] = np.nan
            all_data.append(sub_df)
            continue
        
        # --- 1. SMA (Simple Moving Average) - Adaptive to available data
        try:
            if len(sub_df) >= 5:
                sub_df['ma_5'] = ta.trend.sma_indicator(sub_df['close'], window=5).round(2)
            else:
                sub_df['ma_5'] = np.nan
                
            if len(sub_df) >= 20:
                sub_df['ma_20'] = ta.trend.sma_indicator(sub_df['close'], window=20).round(2)
            elif len(sub_df) >= 10:
                # Use MA10 for smaller datasets
                sub_df['ma_20'] = ta.trend.sma_indicator(sub_df['close'], window=10).round(2)
            else:
                sub_df['ma_20'] = np.nan
        except Exception as e:
            logger.error(f"Error calculating SMA for {code}: {e}")
            sub_df['ma_5'] = np.nan
            sub_df['ma_20'] = np.nan

        # --- 2. EMA (Exponential Moving Average) - Adaptive
        try:
            if len(sub_df) >= 12:
                sub_df['ema_12'] = ta.trend.ema_indicator(sub_df['close'], window=12).round(2)
            else:
                sub_df['ema_12'] = np.nan
                
            if len(sub_df) >= 26:
                sub_df['ema_26'] = ta.trend.ema_indicator(sub_df['close'], window=26).round(2)
            else:
                sub_df['ema_26'] = np.nan
        except Exception as e:
            logger.error(f"Error calculating EMA for {code}: {e}")
            sub_df['ema_12'] = np.nan
            sub_df['ema_26'] = np.nan

        # --- 3. MACD (Moving Average Convergence Divergence) - Adaptive
        try:
            if len(sub_df) >= 26:
                sub_df['macd'] = ta.trend.macd(sub_df['close']).round(2)
                sub_df['macd_signal'] = ta.trend.macd_signal(sub_df['close']).round(2)
            else:
                sub_df['macd'] = np.nan
                sub_df['macd_signal'] = np.nan
        except Exception as e:
            logger.error(f"Error calculating MACD for {code}: {e}")
            sub_df['macd'] = np.nan
            sub_df['macd_signal'] = np.nan

        # --- 4. RSI (Relative Strength Index) - Adaptive
        try:
            if len(sub_df) >= 14:
                sub_df['rsi_14'] = ta.momentum.rsi(sub_df['close'], window=14).round(2)
            elif len(sub_df) >= 7:
                # Use shorter period for small datasets
                sub_df['rsi_14'] = ta.momentum.rsi(sub_df['close'], window=7).round(2)
            else:
                sub_df['rsi_14'] = np.nan
        except Exception as e:
            logger.error(f"Error calculating RSI for {code}: {e}")
            sub_df['rsi_14'] = np.nan

        # --- 5. Bollinger Bands - Adaptive
        try:
            if len(sub_df) >= 20:
                bb_window = 20
            elif len(sub_df) >= 10:
                bb_window = 10
            elif len(sub_df) >= 5:
                bb_window = 5
            else:
                bb_window = None
                
            if bb_window:
                bb = ta.volatility.BollingerBands(sub_df['close'], window=bb_window, window_dev=2)
                sub_df['bb_bbm'] = bb.bollinger_mavg().round(2)   # MA
                sub_df['bb_bbh'] = bb.bollinger_hband().round(2)  # Upper band
                sub_df['bb_bbl'] = bb.bollinger_lband().round(2)  # Lower band
            else:
                sub_df['bb_bbm'] = np.nan
                sub_df['bb_bbh'] = np.nan
                sub_df['bb_bbl'] = np.nan
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands for {code}: {e}")
            sub_df['bb_bbm'] = np.nan
            sub_df['bb_bbh'] = np.nan
            sub_df['bb_bbl'] = np.nan

        # --- 6. ATR (Average True Range) - Adaptive
        try:
            if len(sub_df) >= 14:
                atr_window = 14
            elif len(sub_df) >= 7:
                atr_window = 7
            elif len(sub_df) >= 2:
                atr_window = 2
            else:
                atr_window = None
                
            if atr_window and atr_window >= 2:
                sub_df['atr_14'] = ta.volatility.average_true_range(
                    sub_df['high'], sub_df['low'], sub_df['close'], window=atr_window
                ).round(2)
            else:
                sub_df['atr_14'] = np.nan
        except Exception as e:
            logger.error(f"Error calculating ATR for {code}: {e}")
            sub_df['atr_14'] = np.nan

        # --- 7. OBV (On-Balance Volume) - Always available
        try:
            if 'volume' in sub_df.columns and not sub_df['volume'].isna().all():
                sub_df['obv'] = ta.volume.on_balance_volume(sub_df['close'], sub_df['volume']).round(2)
            else:
                sub_df['obv'] = np.nan
        except Exception as e:
            logger.error(f"Error calculating OBV for {code}: {e}")
            sub_df['obv'] = np.nan

        all_data.append(sub_df)

    # Combine all data (don't drop NaN to preserve all rows)
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
