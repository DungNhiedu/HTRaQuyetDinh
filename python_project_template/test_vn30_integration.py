#!/usr/bin/env python3
"""
Test script to verify VN30 data integration in Sample Data Demo.
This script tests that the VN30 data is loaded, processed, and displayed correctly.
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from src.stock_predictor.data.preprocessor import DataPreprocessor
from src.stock_predictor.data.features import add_technical_indicators, FeatureEngineer

def test_vn30_data_loading():
    """Test loading and processing VN30 data."""
    print("🧪 Testing VN30 Data Loading and Processing...")
    
    vn30_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/Dữ liệu Lịch sử VN 30.csv"
    
    try:
        # Load and process VN30 data using DataPreprocessor
        print("📁 Loading VN30 data...")
        preprocessor = DataPreprocessor()
        vn30_data = preprocessor._read_csv_flexible(vn30_file_path)
        
        if vn30_data is None:
            raise Exception("Could not read VN30 CSV file")
            
        print(f"✅ Raw data loaded: {vn30_data.shape}")
        print(f"   Columns: {list(vn30_data.columns)}")
        
        # Process the VN30 data directly
        print("🔧 Normalizing data format...")
        vn30_data = preprocessor._normalize_data_format(vn30_data, "VN30")
        
        if vn30_data is None or vn30_data.empty:
            raise Exception("Failed to normalize VN30 data format")
            
        print(f"✅ Data format normalized: {vn30_data.shape}")
        print(f"   Normalized columns: {list(vn30_data.columns)}")
        
        # Calculate returns and targets
        print("📈 Calculating returns and targets...")
        vn30_data = preprocessor._calculate_returns_and_targets(vn30_data)
        print(f"✅ Returns and targets calculated: {vn30_data.shape}")
        
        # Sample data info
        print(f"\n📊 VN30 Data Summary:")
        print(f"   Shape: {vn30_data.shape}")
        print(f"   Date range: {vn30_data['date'].min()} to {vn30_data['date'].max()}")
        print(f"   Latest price: {vn30_data['close'].iloc[-1]:.2f}")
        if 'return' in vn30_data.columns:
            print(f"   Latest return: {vn30_data['return'].iloc[-1]:.2f}%")
        if 'target' in vn30_data.columns:
            up_days = (vn30_data['target'] == 1).sum()
            print(f"   Up days: {100 * up_days / len(vn30_data):.1f}%")
        
        # Add technical indicators
        print("\n🔧 Adding technical indicators...")
        data_with_features = add_technical_indicators(vn30_data)
        print(f"✅ Technical indicators added: {data_with_features.shape}")
        
        # Feature engineering
        print("🧠 Advanced feature engineering...")
        feature_engineer = FeatureEngineer()
        enriched_data = feature_engineer.create_price_features(data_with_features)
        enriched_data = feature_engineer.create_volume_features(enriched_data)
        enriched_data = feature_engineer.create_lag_features(enriched_data)
        enriched_data = feature_engineer.create_rolling_features(enriched_data)
        print(f"✅ Feature engineering complete: {enriched_data.shape}")
        
        print(f"\n📋 Final Dataset Preview:")
        print(enriched_data.head())
        
        print(f"\n🎯 Feature Summary:")
        print(f"   Original features: {len(vn30_data.columns)}")
        print(f"   With technical indicators: {len(data_with_features.columns)}")
        print(f"   Final features: {len(enriched_data.columns)}")
        
        # Check for any issues
        if enriched_data.isnull().sum().sum() > 0:
            print(f"⚠️  Warning: {enriched_data.isnull().sum().sum()} null values in final dataset")
        
        print("\n✅ VN30 data integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ VN30 data integration test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_to_synthetic():
    """Test fallback to synthetic data when VN30 data is not available."""
    print("\n🧪 Testing fallback to synthetic data...")
    
    # This would normally be done in the app when VN30 file is not found
    try:
        from src.stock_predictor.app import create_sample_data
        sample_data = create_sample_data()
        print(f"✅ Synthetic data fallback works: {sample_data.shape}")
        return True
    except Exception as e:
        print(f"❌ Synthetic data fallback failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting VN30 Integration Tests...")
    print("=" * 60)
    
    # Test VN30 data loading
    test1_passed = test_vn30_data_loading()
    
    # Test fallback mechanism
    test2_passed = test_fallback_to_synthetic()
    
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   VN30 Data Loading: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Synthetic Fallback: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests PASSED! VN30 integration is working correctly.")
    else:
        print("\n⚠️  Some tests FAILED. Please check the errors above.")
        sys.exit(1)
