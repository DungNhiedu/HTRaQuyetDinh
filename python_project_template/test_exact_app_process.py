#!/usr/bin/env python3
"""
Test the exact same upload process as the Streamlit app does.
"""

import sys
import os
import tempfile
import shutil
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

from stock_predictor.data.preprocessor import DataPreprocessor

def test_exact_app_process():
    """Test the exact same process as in the Streamlit app."""
    print("🔍 Testing exact app upload process...")
    
    # Create temp directory exactly like the app does
    temp_dir = "/tmp/stock_data"
    os.makedirs(temp_dir, exist_ok=True)
    print(f"✅ Created temp directory: {temp_dir}")
    
    try:
        # Copy VN30 demo file exactly like the app does
        source_file = "/Users/dungnhi/Documents/HTRaQuyetDinh/VN30_demo.csv"
        target_file = os.path.join(temp_dir, "VN30_demo.csv")
        
        # Simulate file upload by copying the file
        with open(source_file, 'rb') as src:
            with open(target_file, 'wb') as dst:
                dst.write(src.read())
        
        print(f"✅ Simulated file upload to: {target_file}")
        
        # Process exactly like the app does
        print("\n🔄 Processing with DataPreprocessor.load_and_process_all()...")
        preprocessor = DataPreprocessor()
        merged_data = preprocessor.load_and_process_all(temp_dir)
        
        if merged_data.empty:
            print("❌ No data could be processed from the uploaded files.")
            return False
        
        print(f"✅ Successfully processed data! Shape: {merged_data.shape}")
        print(f"✅ Columns: {list(merged_data.columns)}")
        
        # Test required columns validation like the app does
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in merged_data.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print("Your CSV file must contain at least a 'close' price column.")
            return False
        
        print("✅ All required columns present")
        return True
        
    except Exception as e:
        print(f"❌ Error in exact app process: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"✅ Cleaned up temp directory")

def main():
    """Main test function."""
    print("🚀 Testing exact Streamlit app upload process...\n")
    
    success = test_exact_app_process()
    
    if success:
        print("\n🎉 Exact app process works! The error might be elsewhere.")
    else:
        print("\n⚠️ Found the problem in the exact app process!")

if __name__ == "__main__":
    main()
