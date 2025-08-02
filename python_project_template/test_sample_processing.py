#!/usr/bin/env python3
"""
Test script for processing sample data files with flexible format support.
Tests the updated DataPreprocessor with real data files.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stock_predictor.data.preprocessor import DataPreprocessor
from stock_predictor.data.features import add_technical_indicators, FeatureEngineer

def test_sample_files():
    """Test processing of sample data files."""
    print("=== Testing Sample Data Processing ===")
    
    # Create test directory structure
    test_dir = "/tmp/sample_data_test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Sample USD_VND data (similar to your file format)
    usd_vnd_data = """Date;close;Open;High;Low;volumn;% turnover
31/07/2025;26207.5;26225.0;26230.0;26205.0;;0.02%
30/07/2025;26202.5;26225.5;26266.5;26190.0;;-0.09%
29/07/2025;26225.0;26162.5;26260.0;26162.5;;0.10%
28/07/2025;26200.0;26145.0;26221.0;26127.5;;0.21%
25/07/2025;26145.0;26132.5;26158.0;26112.5;;0.04%
24/07/2025;26135.0;26157.5;26163.0;26116.5;;-0.06%
23/07/2025;26150.0;26147.0;26166.5;26120.0;;0.02%
22/07/2025;26145.0;26145.0;26162.5;26128.5;;-0.04%
21/07/2025;26155.0;26155.0;26168.5;26139.0;;-0.02%
18/07/2025;26160.0;26155.0;26173.0;26138.0;;0.00%"""
    
    # Sample Gold data (comma-separated format)
    gold_data = """Date,Close,Open,High,Low,Volume
31/07/2025,2400.50,2395.00,2405.00,2390.00,150000
30/07/2025,2395.00,2400.00,2410.00,2385.00,160000
29/07/2025,2400.00,2385.00,2420.00,2380.00,180000
28/07/2025,2385.00,2390.00,2390.00,2375.00,140000
25/07/2025,2390.00,2380.00,2395.00,2375.00,155000"""
    
    # Write sample files
    with open(os.path.join(test_dir, "USD_VND_sample.csv"), "w", encoding="utf-8") as f:
        f.write(usd_vnd_data)
    
    with open(os.path.join(test_dir, "GOLD_sample.csv"), "w", encoding="utf-8") as f:
        f.write(gold_data)
    
    print(f"Created test files in: {test_dir}")
    
    # Test DataPreprocessor
    try:
        print("\nTesting DataPreprocessor...")
        preprocessor = DataPreprocessor()
        
        # Process all files in test directory
        merged_data = preprocessor.load_and_process_all(test_dir)
        
        print(f"✅ Successfully processed data!")
        print(f"   Shape: {merged_data.shape}")
        print(f"   Columns: {list(merged_data.columns)}")
        print(f"   Stock codes: {merged_data['code'].unique()}")
        
        # Show sample data
        print("\nSample processed data:")
        print(merged_data.head())
        
        # Test feature engineering
        print("\nTesting Feature Engineering...")
        feature_engineer = FeatureEngineer()
        
        # Add technical indicators
        data_with_indicators = add_technical_indicators(merged_data)
        print(f"✅ Added technical indicators! Shape: {data_with_indicators.shape}")
        
        # Create additional features
        enriched_data = feature_engineer.create_features(data_with_indicators)
        print(f"✅ Created additional features! Final shape: {enriched_data.shape}")
        
        # Show feature columns
        new_features = set(enriched_data.columns) - set(merged_data.columns)
        print(f"\n📊 New features added ({len(new_features)}):")
        for i, feature in enumerate(sorted(new_features)):
            if i < 10:  # Show first 10 features
                print(f"   • {feature}")
            elif i == 10:
                print(f"   • ... and {len(new_features) - 10} more features")
                break
        
        # Data quality check
        print(f"\n📈 Data Quality:")
        print(f"   • Total rows: {len(enriched_data)}")
        print(f"   • Date range: {merged_data['date'].min()} to {merged_data['date'].max()}")
        print(f"   • Missing values: {enriched_data.isnull().sum().sum()}")
        print(f"   • Positive returns: {(merged_data['return'] > 0).sum()}")
        print(f"   • Negative returns: {(merged_data['return'] < 0).sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_real_file_processing():
    """Test processing with user's actual files if available."""
    print("\n=== Testing Real File Processing ===")
    
    # Paths to user's files (adjust as needed)
    desktop_path = "/Users/dungnhi/Desktop"
    usd_file = os.path.join(desktop_path, "Dữ liệu Lịch sử USD_VND.csv")
    gold_file = os.path.join(desktop_path, "dữ liệu lịch sử giá vàng.csv")
    
    # Create temporary directory and copy files
    test_dir = "/tmp/real_data_test"
    os.makedirs(test_dir, exist_ok=True)
    
    files_to_test = []
    
    # Check if files exist and copy them
    if os.path.exists(usd_file):
        import shutil
        dest_file = os.path.join(test_dir, "USD_VND_real.csv")
        shutil.copy2(usd_file, dest_file)
        files_to_test.append("USD_VND")
        print(f"✅ Found USD_VND file: {usd_file}")
    
    if os.path.exists(gold_file):
        import shutil
        dest_file = os.path.join(test_dir, "GOLD_real.csv")
        shutil.copy2(gold_file, dest_file)
        files_to_test.append("GOLD")
        print(f"✅ Found Gold file: {gold_file}")
    
    if not files_to_test:
        print("⚠️  No real data files found on Desktop. Skipping real file test.")
        return True
    
    try:
        print(f"\nProcessing {len(files_to_test)} real files...")
        preprocessor = DataPreprocessor()
        
        # Process real files
        merged_data = preprocessor.load_and_process_all(test_dir)
        
        print(f"✅ Successfully processed real data!")
        print(f"   Shape: {merged_data.shape}")
        print(f"   Stock codes: {merged_data['code'].unique()}")
        print(f"   Date range: {merged_data['date'].min()} to {merged_data['date'].max()}")
        
        # Show sample
        print("\nReal data sample:")
        print(merged_data.head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing real files: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Enhanced Data Processing Tests")
    print("=" * 50)
    
    # Test 1: Sample files
    success1 = test_sample_files()
    
    # Test 2: Real files if available
    success2 = test_real_file_processing()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! The system can now process your data files.")
        print("\n📝 Usage Instructions:")
        print("1. Upload your CSV files through the Streamlit app")
        print("2. The system will automatically detect the format")
        print("3. Data will be processed and analyzed with technical indicators")
        print("4. Use 'Upload CSV Files' option in the web interface")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
