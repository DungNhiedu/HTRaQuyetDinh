#!/usr/bin/env python3
"""
Simple test for real data processing with improved error handling.
"""

import sys
import os
import pandas as pd
import shutil

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stock_predictor.data.preprocessor import DataPreprocessor

def test_real_data():
    """Test with real data files."""
    print("=== Testing Real Data Processing ===")
    
    # Paths to real files
    desktop_path = "/Users/dungnhi/Desktop"
    usd_file = os.path.join(desktop_path, "Dữ liệu Lịch sử USD_VND.csv")
    gold_file = os.path.join(desktop_path, "dữ liệu lịch sử giá vàng.csv")
    
    # Create test directory
    test_dir = "/tmp/real_data_test_simple"
    os.makedirs(test_dir, exist_ok=True)
    
    files_found = []
    
    # Check and copy files
    if os.path.exists(usd_file):
        shutil.copy2(usd_file, os.path.join(test_dir, "USD_VND.csv"))
        files_found.append("USD_VND")
        print(f"✅ Found USD file: {usd_file}")
    
    if os.path.exists(gold_file):
        shutil.copy2(gold_file, os.path.join(test_dir, "GOLD.csv"))
        files_found.append("GOLD")
        print(f"✅ Found Gold file: {gold_file}")
    
    if not files_found:
        print("❌ No files found on Desktop")
        return
    
    # Test preprocessing
    try:
        print(f"\nProcessing {len(files_found)} files...")
        preprocessor = DataPreprocessor()
        
        # Process files
        result = preprocessor.load_and_process_all(test_dir)
        
        if result.empty:
            print("❌ No data was processed")
            return
        
        print(f"✅ Successfully processed data!")
        print(f"   Shape: {result.shape}")
        print(f"   Columns: {list(result.columns)}")
        
        if 'code' in result.columns:
            print(f"   Stock codes: {result['code'].unique()}")
        
        if 'date' in result.columns:
            print(f"   Date range: {result['date'].min()} to {result['date'].max()}")
        
        # Show sample data
        print(f"\nSample data:")
        print(result.head())
        
        # Test with smaller sample for technical indicators
        print(f"\n=== Testing Technical Indicators ===")
        
        # Take a larger sample from the data
        if len(result) > 50:
            sample_data = result.tail(50).copy()  # Take last 50 rows
        else:
            sample_data = result.copy()
        
        print(f"Testing with {len(sample_data)} rows...")
        
        from stock_predictor.data.features import add_technical_indicators
        enhanced_data = add_technical_indicators(sample_data)
        
        print(f"✅ Technical indicators added!")
        print(f"   Final shape: {enhanced_data.shape}")
        
        # Show new columns
        original_cols = set(sample_data.columns)
        new_cols = set(enhanced_data.columns) - original_cols
        
        print(f"   New indicators: {list(new_cols)}")
        
        print(f"\n🎉 Processing successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_data()
