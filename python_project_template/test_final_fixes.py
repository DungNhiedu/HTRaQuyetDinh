#!/usr/bin/env python3
"""
Final test to confirm both errors are fixed.
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

def test_sample_data_loading():
    """Test the sample data loading fix."""
    print("🔍 Testing Sample Data Loading Fix...")
    
    # Test VN30 file path
    vn30_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/VN30_demo.csv"
    
    if not os.path.exists(vn30_file_path):
        print(f"❌ VN30 file does not exist: {vn30_file_path}")
        return False
    
    try:
        preprocessor = DataPreprocessor()
        vn30_data = preprocessor._read_csv_flexible(vn30_file_path)
        
        if vn30_data is None:
            print("❌ Could not read VN30 CSV file")
            return False
        
        print(f"✅ VN30 file read successfully. Shape: {vn30_data.shape}")
        
        # Test normalization
        vn30_data = preprocessor._normalize_data_format(vn30_data, "VN30")
        
        if vn30_data is None or vn30_data.empty:
            print("❌ Failed to normalize VN30 data format")
            return False
        
        print(f"✅ VN30 data normalized successfully. Shape: {vn30_data.shape}")
        
        # Test returns calculation
        vn30_data = preprocessor._calculate_returns_and_targets(vn30_data)
        print(f"✅ VN30 returns calculated successfully. Final shape: {vn30_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in sample data loading: {str(e)}")
        return False

def test_upload_processing_with_cleanup():
    """Test upload processing with proper temp directory cleanup."""
    print("\n🔍 Testing Upload Processing with Cleanup...")
    
    # Clean temp directory like the app does
    temp_dir = "/tmp/stock_data"
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"✅ Cleaned and created temp directory: {temp_dir}")
    
    try:
        # Copy only VN30 demo file
        source_file = "/Users/dungnhi/Documents/HTRaQuyetDinh/VN30_demo.csv"
        target_file = os.path.join(temp_dir, "VN30_demo.csv")
        
        with open(source_file, 'rb') as src:
            with open(target_file, 'wb') as dst:
                dst.write(src.read())
        
        print(f"✅ Uploaded single file: VN30_demo.csv")
        
        # Process with DataPreprocessor
        preprocessor = DataPreprocessor()
        merged_data = preprocessor.load_and_process_all(temp_dir)
        
        if merged_data.empty:
            print("❌ No data could be processed")
            return False
        
        print(f"✅ Data processed successfully. Shape: {merged_data.shape}")
        print(f"✅ Columns: {list(merged_data.columns)}")
        
        # Test required columns
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in merged_data.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        print("✅ All required columns present")
        
        # Test AI prediction data preparation
        if 'return' in merged_data.columns and len(merged_data) > 1:
            latest_return = merged_data['return'].iloc[-1] if not merged_data['return'].iloc[-1] != merged_data['return'].iloc[-1] else 0
        else:
            if len(merged_data) >= 2:
                latest_close = merged_data['close'].iloc[-1]
                prev_close = merged_data['close'].iloc[-2]
                latest_return = ((latest_close - prev_close) / prev_close) * 100
            else:
                latest_return = 0
        
        # Create data summary
        uploaded_summary = {
            'total_days': len(merged_data),
            'current_price': merged_data['close'].iloc[-1] if len(merged_data) > 0 else 0,
            'latest_change': latest_return,
            'up_days_ratio': 100 * (merged_data['target'] == 1).sum() / len(merged_data) if 'target' in merged_data.columns and len(merged_data) > 0 else 50,
            'highest_price': merged_data['close'].max() if len(merged_data) > 0 else 0,
            'lowest_price': merged_data['close'].min() if len(merged_data) > 0 else 0,
            'avg_volatility': abs(merged_data['return']).mean() if 'return' in merged_data.columns and len(merged_data) > 0 else 0
        }
        
        print("✅ AI prediction data summary prepared:")
        for key, value in uploaded_summary.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in upload processing: {str(e)}")
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
    print("🚀 Final Test: Both Errors Fixed\n")
    
    # Test 1: Sample data loading
    test1_passed = test_sample_data_loading()
    
    # Test 2: Upload processing
    test2_passed = test_upload_processing_with_cleanup()
    
    print(f"\n📊 Final Test Results:")
    print(f"   Sample Data Loading: {'✅ FIXED' if test1_passed else '❌ STILL BROKEN'}")
    print(f"   Upload CSV Processing: {'✅ FIXED' if test2_passed else '❌ STILL BROKEN'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 BOTH ERRORS HAVE BEEN SUCCESSFULLY FIXED!")
        print("\n✅ The app should now work correctly for:")
        print("   - Sample Data Analysis (no more 'Could not read VN30 CSV file' error)")
        print("   - Upload CSV Files + AI Prediction (no more 'Error processing files' error)")
    else:
        print("\n⚠️ Some issues remain. Please check the failing tests above.")

if __name__ == "__main__":
    main()
