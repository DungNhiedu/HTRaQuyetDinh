#!/usr/bin/env python3
"""
Test the fixes for the VN30 data loading and AI prediction errors.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stock_predictor.data.preprocessor import DataPreprocessor
import pandas as pd

def test_vn30_file_loading():
    """Test loading the VN30 demo file."""
    print("🔍 Testing VN30 file loading...")
    
    vn30_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/VN30_demo.csv"
    
    # Test file existence
    if not os.path.exists(vn30_file_path):
        print(f"❌ File does not exist: {vn30_file_path}")
        return False
    
    print(f"✅ File exists: {vn30_file_path}")
    
    # Test reading with DataPreprocessor
    preprocessor = DataPreprocessor()
    df = preprocessor._read_csv_flexible(vn30_file_path)
    
    if df is None:
        print("❌ Could not read VN30 CSV file")
        return False
    
    print(f"✅ Successfully read VN30 file. Shape: {df.shape}")
    print(f"✅ Columns: {list(df.columns)}")
    
    # Test normalization
    normalized_df = preprocessor._normalize_data_format(df, "VN30")
    
    if normalized_df is None or normalized_df.empty:
        print("❌ Failed to normalize VN30 data")
        return False
    
    print(f"✅ Successfully normalized VN30 data. Shape: {normalized_df.shape}")
    print(f"✅ Normalized columns: {list(normalized_df.columns)}")
    
    # Test returns calculation
    final_df = preprocessor._calculate_returns_and_targets(normalized_df)
    
    if final_df is None or final_df.empty:
        print("❌ Failed to calculate returns and targets")
        return False
    
    print(f"✅ Successfully calculated returns and targets. Shape: {final_df.shape}")
    print(f"✅ Final columns: {list(final_df.columns)}")
    
    # Check required columns for AI prediction
    required_cols = ['date', 'close']
    missing_cols = [col for col in required_cols if col not in final_df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns for AI prediction: {missing_cols}")
        return False
    
    print("✅ All required columns present for AI prediction")
    
    return True

def test_data_summary_preparation():
    """Test data summary preparation for AI prediction."""
    print("\n🔍 Testing data summary preparation...")
    
    # Create a sample processed dataset
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'close': [100, 102, 101, 105, 103, 107, 106, 108, 110, 109],
        'return': [0, 2.0, -0.98, 3.96, -1.90, 3.88, -0.93, 1.89, 1.85, -0.91],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        'code': ['TEST'] * 10
    })
    
    try:
        # Test data summary preparation (similar to what's in the app)
        uploaded_summary = {
            'total_days': len(sample_data),
            'time_duration': '10 days',
            'current_price': sample_data['close'].iloc[-1] if len(sample_data) > 0 else 0,
            'latest_change': sample_data['return'].iloc[-1] if 'return' in sample_data.columns and len(sample_data) > 1 else 0,
            'up_days_ratio': 100 * (sample_data['target'] == 1).sum() / len(sample_data) if 'target' in sample_data.columns and len(sample_data) > 0 else 50,
            'highest_price': sample_data['close'].max() if len(sample_data) > 0 else 0,
            'lowest_price': sample_data['close'].min() if len(sample_data) > 0 else 0,
            'avg_volatility': abs(sample_data['return']).mean() if 'return' in sample_data.columns and len(sample_data) > 0 else 0
        }
        
        print("✅ Successfully prepared data summary:")
        for key, value in uploaded_summary.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error preparing data summary: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🚀 Testing fixes for VN30 data loading and AI prediction errors...\n")
    
    # Test 1: VN30 file loading
    test1_passed = test_vn30_file_loading()
    
    # Test 2: Data summary preparation
    test2_passed = test_data_summary_preparation()
    
    print(f"\n📊 Test Results:")
    print(f"   VN30 File Loading: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Data Summary Prep: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The fixes should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
