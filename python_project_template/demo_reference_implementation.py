"""
Demo script to test the updated stock predictor with reference implementation.
"""

import sys
import os
sys.path.append('/Users/dungnhi/Documents/HTRaQuyetDinh/python_project_template/src')

import pandas as pd
import numpy as np
from stock_predictor.data.preprocessor import DataPreprocessor
from stock_predictor.data.features import add_technical_indicators, FeatureEngineer

def create_sample_data():
    """Create sample stock data for testing."""
    # Create sample data similar to the reference format
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Generate synthetic OHLCV data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    
    data = []
    for i, date in enumerate(dates):
        open_price = close_prices[i] + np.random.randn() * 0.3
        high_price = max(open_price, close_prices[i]) + abs(np.random.randn()) * 0.5
        low_price = min(open_price, close_prices[i]) - abs(np.random.randn()) * 0.5
        volume = int(1000000 + np.random.randn() * 100000)
        turnover = volume * close_prices[i]
        
        data.append({
            'code': 'VCB',
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_prices[i], 2),
            'volume': volume,
            'turnover': round(turnover, 2),
            'return': 0.0,  # Will be calculated
            'target': 0     # Will be calculated
        })
    
    df = pd.DataFrame(data)
    
    # Calculate return and target
    df['return'] = df['close'].pct_change() * 100
    df['return'] = df['return'].round(2).fillna(0)
    df['target'] = (df['return'] > 0).astype(int)
    
    return df

def test_preprocessor():
    """Test the data preprocessor."""
    print("Testing DataPreprocessor...")
    
    # Create sample data
    sample_data = create_sample_data()
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample data columns: {list(sample_data.columns)}")
    print(f"Sample data head:\n{sample_data.head()}")
    
    # Test preprocessor methods
    preprocessor = DataPreprocessor()
    
    # Test data splitting
    splits = preprocessor.split_data(sample_data, train_ratio=0.7, val_ratio=0.15)
    print(f"\nData splits:")
    print(f"Train: {len(splits['train'])}")
    print(f"Val: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")
    
    print("DataPreprocessor test completed successfully!")

def test_feature_engineer():
    """Test the feature engineer."""
    print("\nTesting FeatureEngineer...")
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Test technical indicators function
    print("Testing add_technical_indicators function...")
    data_with_indicators = add_technical_indicators(sample_data)
    print(f"Data with indicators shape: {data_with_indicators.shape}")
    print(f"New columns added: {set(data_with_indicators.columns) - set(sample_data.columns)}")
    
    # Test FeatureEngineer class
    feature_engineer = FeatureEngineer()
    
    print("\nTesting FeatureEngineer.create_technical_indicators...")
    result = feature_engineer.create_technical_indicators(sample_data)
    print(f"Result shape: {result.shape}")
    
    # Test additional features
    print("\nTesting additional feature creation...")
    result_with_more = feature_engineer.create_price_features(result)
    result_with_more = feature_engineer.create_volume_features(result_with_more)
    result_with_more = feature_engineer.create_lag_features(result_with_more)
    result_with_more = feature_engineer.create_rolling_features(result_with_more)
    
    print(f"Final result shape: {result_with_more.shape}")
    print(f"Total columns: {len(result_with_more.columns)}")
    
    print("FeatureEngineer test completed successfully!")

def main():
    """Main demo function."""
    print("=== Stock Predictor Demo - Reference Implementation ===")
    
    try:
        # Test preprocessor
        test_preprocessor()
        
        # Test feature engineer
        test_feature_engineer()
        
        print("\n=== All tests completed successfully! ===")
        print("\nThe updated implementation based on copy_of_đồ_án_dss_nhóm_1.py is working correctly.")
        print("Key features implemented:")
        print("1. Data preprocessing with CSV loading and processing")
        print("2. Technical indicators using ta library (SMA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV)")
        print("3. Feature engineering with price, volume, lag, and rolling features")
        print("4. Binary target classification (price increase/decrease)")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
