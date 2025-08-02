"""
Helper functions cho stock market prediction
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
import logging
import os
import json
import pickle
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Path to log file (optional)
    """
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_data_format(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate DataFrame format
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if DataFrame is not empty
        if data.empty:
            logger.error("DataFrame is empty")
            return False
            
        # Check required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return False
            
        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex")
            
        # Check for NaN values
        if data.isnull().any().any():
            logger.warning("Data contains NaN values")
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        return False

def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series
    
    Args:
        prices: Price series
        method: 'simple' or 'log'
        
    Returns:
        Returns series
    """
    try:
        if method == 'simple':
            returns = prices.pct_change()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return returns.dropna()
        
    except Exception as e:
        logger.error(f"Error calculating returns: {str(e)}")
        return pd.Series(dtype=float)

def create_lag_features(
    data: pd.DataFrame, 
    columns: List[str], 
    lags: List[int]
) -> pd.DataFrame:
    """
    Create lag features for specified columns
    
    Args:
        data: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features
    """
    try:
        df = data.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
        return df
        
    except Exception as e:
        logger.error(f"Error creating lag features: {str(e)}")
        return data

def split_train_test_by_date(
    data: pd.DataFrame, 
    split_date: Union[str, datetime]
) -> tuple:
    """
    Split data by date
    
    Args:
        data: DataFrame with DatetimeIndex
        split_date: Date to split on
        
    Returns:
        Tuple (train_data, test_data)
    """
    try:
        if isinstance(split_date, str):
            split_date = pd.to_datetime(split_date)
            
        train_data = data[data.index < split_date]
        test_data = data[data.index >= split_date]
        
        logger.info(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")
        
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error splitting data by date: {str(e)}")
        return data, pd.DataFrame()

def save_object(obj: Any, filepath: str) -> None:
    """
    Save object to file using pickle
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
            
        logger.info(f"Object saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving object: {str(e)}")
        raise

def load_object(filepath: str) -> Any:
    """
    Load object from pickle file
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
            
        logger.info(f"Object loaded from {filepath}")
        return obj
        
    except Exception as e:
        logger.error(f"Error loading object: {str(e)}")
        raise

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dict to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        logger.info(f"JSON saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving JSON: {str(e)}")
        raise

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        logger.info(f"JSON loaded from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading JSON: {str(e)}")
        raise

def format_number(number: float, decimals: int = 2) -> str:
    """
    Format number for display
    
    Args:
        number: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    if pd.isna(number) or np.isinf(number):
        return "N/A"
        
    if abs(number) >= 1e6:
        return f"{number/1e6:.{decimals}f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.{decimals}f}K"
    else:
        return f"{number:.{decimals}f}"

def calculate_sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    try:
        excess_returns = returns - (risk_free_rate / periods_per_year)
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
        
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {str(e)}")
        return 0.0

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        cumulative_returns: Cumulative returns series
        
    Returns:
        Maximum drawdown
    """
    try:
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
        
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {str(e)}")
        return 0.0

def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in data
    
    Args:
        data: Data series
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    try:
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > threshold
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return outliers
        
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        return pd.Series([False] * len(data), index=data.index)

def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    Get list of trading days between start and end date
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of trading days
    """
    try:
        # Simple implementation - exclude weekends
        current = start_date
        trading_days = []
        
        while current <= end_date:
            if current.weekday() < 5:  # Monday = 0, Sunday = 6
                trading_days.append(current)
            current += timedelta(days=1)
            
        return trading_days
        
    except Exception as e:
        logger.error(f"Error getting trading days: {str(e)}")
        return []

def resample_data(
    data: pd.DataFrame, 
    frequency: str = 'D',
    agg_methods: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Resample data to different frequency
    
    Args:
        data: Input DataFrame
        frequency: Target frequency ('D', 'W', 'M', etc.)
        agg_methods: Dict mapping column names to aggregation methods
        
    Returns:
        Resampled DataFrame
    """
    try:
        if agg_methods is None:
            # Default aggregation methods
            agg_methods = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            
        # Apply aggregation methods to available columns
        agg_dict = {col: method for col, method in agg_methods.items() if col in data.columns}
        
        # For other columns, use mean
        for col in data.columns:
            if col not in agg_dict:
                agg_dict[col] = 'mean'
                
        resampled = data.resample(frequency).agg(agg_dict)
        
        return resampled.dropna()
        
    except Exception as e:
        logger.error(f"Error resampling data: {str(e)}")
        return data

def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Check data quality and return report
    
    Args:
        data: DataFrame to check
        
    Returns:
        Dict with quality metrics
    """
    try:
        report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'duplicated_rows': data.duplicated().sum(),
            'data_types': data.dtypes.astype(str).to_dict()
        }
        
        # Check for constant columns
        constant_columns = [col for col in data.columns if data[col].nunique() <= 1]
        report['constant_columns'] = constant_columns
        
        # Check date range
        if isinstance(data.index, pd.DatetimeIndex):
            report['date_range'] = {
                'start': str(data.index.min()),
                'end': str(data.index.max()),
                'days': (data.index.max() - data.index.min()).days
            }
            
        return report
        
    except Exception as e:
        logger.error(f"Error checking data quality: {str(e)}")
        return {'error': str(e)}
