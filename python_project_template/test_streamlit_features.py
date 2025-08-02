#!/usr/bin/env python3
"""
Test script to verify Streamlit app features and identify any improvements needed
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our modules
from src.stock_predictor.data.preprocessor import DataPreprocessor
from src.stock_predictor.data.features import add_technical_indicators, FeatureEngineer

def test_vn30_data_loading():
    """Test VN30 data loading from the actual file"""
    print("🧪 Testing VN30 data loading...")
    
    vn30_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/Dữ liệu Lịch sử VN 30.csv"
    
    try:
        preprocessor = DataPreprocessor()
        vn30_data = preprocessor._read_csv_flexible(vn30_file_path)
        
        if vn30_data is None:
            print("❌ Could not read VN30 CSV file")
            return False
        
        print(f"✅ VN30 data loaded successfully. Shape: {vn30_data.shape}")
        
        # Process the VN30 data
        vn30_data = preprocessor._normalize_data_format(vn30_data, "VN30")
        
        if vn30_data is None or vn30_data.empty:
            print("❌ Failed to normalize VN30 data format")
            return False
        
        print(f"✅ VN30 data normalized. Shape: {vn30_data.shape}")
        
        # Calculate returns and targets
        vn30_data = preprocessor._calculate_returns_and_targets(vn30_data)
        print(f"✅ Returns and targets calculated. Final shape: {vn30_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading VN30 data: {str(e)}")
        return False

def test_csv_upload_simulation():
    """Test CSV upload functionality with different formats"""
    print("\n🧪 Testing CSV upload simulation...")
    
    # Test VN30 format file
    vn30_demo_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/VN30_demo.csv"
    
    if os.path.exists(vn30_demo_path):
        try:
            preprocessor = DataPreprocessor()
            uploaded_data = preprocessor.process_uploaded_data(vn30_demo_path)
            
            if uploaded_data is not None and not uploaded_data.empty:
                print(f"✅ VN30 demo CSV processed successfully. Shape: {uploaded_data.shape}")
                
                # Add technical indicators
                enriched_data = add_technical_indicators(uploaded_data)
                print(f"✅ Technical indicators added. Shape: {enriched_data.shape}")
                
                return True
            else:
                print("❌ VN30 demo CSV processing failed")
                return False
                
        except Exception as e:
            print(f"❌ Error processing VN30 demo CSV: {str(e)}")
            return False
    else:
        print(f"❌ VN30 demo file not found at {vn30_demo_path}")
        return False

def test_technical_indicators():
    """Test technical indicators calculation"""
    print("\n🧪 Testing technical indicators...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    try:
        enriched_data = add_technical_indicators(data)
        print(f"✅ Technical indicators calculated. Shape: {enriched_data.shape}")
        
        # Check for expected columns
        expected_indicators = ['sma_20', 'ema_20', 'rsi', 'macd', 'bb_upper', 'bb_lower']
        found_indicators = [col for col in expected_indicators if col in enriched_data.columns]
        
        print(f"✅ Found indicators: {found_indicators}")
        return True
        
    except Exception as e:
        print(f"❌ Error calculating technical indicators: {str(e)}")
        return False

def test_ai_prediction_data_preparation():
    """Test data preparation for AI prediction"""
    print("\n🧪 Testing AI prediction data preparation...")
    
    # Create sample data
    dates = pd.date_range('2015-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'date': dates,
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
        'return': np.random.randn(1000) * 2,
        'target': np.random.choice([0, 1], 1000)
    })
    
    try:
        # Calculate time duration
        time_duration = (data['date'].max() - data['date'].min()).days / 365.25
        
        # Prepare data summary
        data_summary = {
            'total_days': len(data),
            'time_duration': f"{time_duration:.1f} years ({data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')})",
            'current_price': data['close'].iloc[-1],
            'latest_change': data['return'].iloc[-1],
            'up_days_ratio': 100 * (data['target'] == 1).sum() / len(data),
            'highest_price': data['close'].max(),
            'lowest_price': data['close'].min(),
            'avg_volatility': abs(data['return']).mean()
        }
        
        print("✅ Data summary prepared for AI prediction:")
        for key, value in data_summary.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error preparing AI prediction data: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Streamlit App Features")
    print("="*50)
    
    tests = [
        test_vn30_data_loading,
        test_csv_upload_simulation, 
        test_technical_indicators,
        test_ai_prediction_data_preparation
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "="*50)
    print("📊 Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The Streamlit app should work correctly.")
    else:
        print("⚠️ Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main()
