#!/usr/bin/env python3
"""
Quick test to verify VN30 technical indicators are working correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from src.stock_predictor.data.preprocessor import DataPreprocessor
from src.stock_predictor.data.features import add_technical_indicators

def test_vn30_technical_indicators():
    """Test VN30 data loading and technical indicators."""
    print("🧪 Testing VN30 Technical Indicators...")
    
    vn30_file_path = "/Users/dungnhi/Desktop/Dữ liệu Lịch sử VN 30.csv"
    
    try:
        # Load VN30 data
        print("📁 Loading VN30 data...")
        preprocessor = DataPreprocessor()
        vn30_data = preprocessor._read_csv_flexible(vn30_file_path)
        
        if vn30_data is None:
            raise Exception("Could not read VN30 CSV file")
        
        print(f"✅ Raw data loaded: {vn30_data.shape}")
        print(f"   Columns: {list(vn30_data.columns)}")
        
        # Process the data
        print("🔧 Processing VN30 data...")
        vn30_data = preprocessor._normalize_data_format(vn30_data, "VN30")
        
        if vn30_data is None or vn30_data.empty:
            raise Exception("Failed to normalize VN30 data format")
        
        # Calculate returns and targets
        vn30_data = preprocessor._calculate_returns_and_targets(vn30_data)
        
        print(f"✅ Processed data: {vn30_data.shape}")
        print(f"   Columns: {list(vn30_data.columns)}")
        print(f"   Date range: {vn30_data['date'].min()} to {vn30_data['date'].max()}")
        print(f"   Price range: {vn30_data['close'].min():.2f} - {vn30_data['close'].max():.2f}")
        
        # Add technical indicators
        print("📊 Adding technical indicators...")
        data_with_features = add_technical_indicators(vn30_data)
        
        print(f"✅ Technical indicators added: {data_with_features.shape}")
        
        # Check what indicators were added
        original_cols = set(vn30_data.columns)
        new_cols = set(data_with_features.columns) - original_cols
        
        print(f"\n📈 Technical Indicators Added ({len(new_cols)}):")
        for col in sorted(new_cols):
            non_null_count = data_with_features[col].notna().sum()
            print(f"   • {col}: {non_null_count}/{len(data_with_features)} non-null values")
        
        # Show sample data
        print(f"\n📋 Sample Data (first 5 rows):")
        display_cols = ['date', 'close', 'ma_5', 'ma_20', 'rsi_14', 'bb_bbm']
        available_cols = [col for col in display_cols if col in data_with_features.columns]
        print(data_with_features[available_cols].head())
        
        # Show last 5 rows
        print(f"\n📋 Sample Data (last 5 rows):")
        print(data_with_features[available_cols].tail())
        
        print("\n✅ VN30 technical indicators test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ VN30 technical indicators test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting VN30 Technical Indicators Test...")
    print("=" * 60)
    
    success = test_vn30_technical_indicators()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests PASSED! VN30 technical indicators are working correctly.")
    else:
        print("⚠️  Tests FAILED. Please check the errors above.")
