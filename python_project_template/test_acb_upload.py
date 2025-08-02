#!/usr/bin/env python3
"""
Test the ACB_10Y.csv file with the updated upload system.
"""

import sys
import os
import tempfile
import shutil
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)

from stock_predictor.data.preprocessor import DataPreprocessor

def test_acb_file():
    """Test the ACB_10Y.csv file processing."""
    print("🔍 Testing ACB_10Y.csv file processing...")
    
    # Create temp directory like the app does
    temp_dir = "/tmp/stock_data"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Copy ACB file to temp directory
        source_file = "/Users/dungnhi/Documents/HTRaQuyetDinh/ACB_10Y.csv"
        target_file = os.path.join(temp_dir, "ACB_10Y.csv")
        
        with open(source_file, 'rb') as src:
            with open(target_file, 'wb') as dst:
                dst.write(src.read())
        
        print(f"✅ Copied ACB file to: {target_file}")
        
        # Process with DataPreprocessor
        preprocessor = DataPreprocessor()
        merged_data = preprocessor.load_and_process_all(temp_dir)
        
        if merged_data.empty:
            print("❌ No data could be processed from ACB file")
            return False
        
        print(f"✅ ACB data processed successfully! Shape: {merged_data.shape}")
        print(f"✅ Columns: {list(merged_data.columns)}")
        
        # Check required columns
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in merged_data.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        print("✅ All required columns present")
        
        # Test data summary preparation
        if 'return' in merged_data.columns and len(merged_data) > 1:
            latest_return = merged_data['return'].iloc[-1] if not merged_data['return'].iloc[-1] != merged_data['return'].iloc[-1] else 0
        else:
            if len(merged_data) >= 2:
                latest_close = merged_data['close'].iloc[-1]
                prev_close = merged_data['close'].iloc[-2]
                latest_return = ((latest_close - prev_close) / prev_close) * 100
            else:
                latest_return = 0
        
        print(f"✅ Latest return calculated: {latest_return}")
        
        # Show sample data
        print(f"✅ Sample data:")
        print(f"   First date: {merged_data['date'].min()}")
        print(f"   Last date: {merged_data['date'].max()}")
        print(f"   Price range: {merged_data['close'].min():.2f} - {merged_data['close'].max():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing ACB file: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"✅ Cleaned up temp directory")

def main():
    """Main test function."""
    print("🚀 Testing ACB_10Y.csv upload processing...\n")
    
    success = test_acb_file()
    
    if success:
        print("\n🎉 ACB file processing works! The upload button should work now.")
    else:
        print("\n⚠️ ACB file processing failed. Check the errors above.")

if __name__ == "__main__":
    main()
