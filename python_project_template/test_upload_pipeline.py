#!/usr/bin/env python3
"""
Test the upload CSV functionality with the VN30 demo file.
"""

import sys
import os
import tempfile
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stock_predictor.data.preprocessor import DataPreprocessor

def test_upload_processing():
    """Test the full upload processing pipeline."""
    print("🔍 Testing upload CSV processing...")
    
    # Create a temporary directory (simulating upload)
    temp_dir = tempfile.mkdtemp()
    print(f"✅ Created temp directory: {temp_dir}")
    
    try:
        # Copy VN30 demo file to temp directory (simulating upload)
        source_file = "/Users/dungnhi/Documents/HTRaQuyetDinh/VN30_demo.csv"
        target_file = os.path.join(temp_dir, "VN30_demo.csv")
        
        shutil.copy2(source_file, target_file)
        print(f"✅ Copied file to temp directory")
        
        # Process using DataPreprocessor (same as in the app)
        preprocessor = DataPreprocessor()
        merged_data = preprocessor.load_and_process_all(temp_dir)
        
        if merged_data.empty:
            print("❌ No data could be processed from the uploaded files")
            return False
        
        print(f"✅ Successfully processed uploaded data! Shape: {merged_data.shape}")
        
        # Test session state simulation
        session_state = {
            'uploaded_data': merged_data,
            'uploaded_time_duration': '10.0 years (2015-01-01 to 2025-01-01)'
        }
        
        print("✅ Session state simulation created")
        
        # Test AI prediction data preparation (same as in app)
        if 'uploaded_data' not in session_state or session_state['uploaded_data'].empty:
            print("❌ No uploaded data found in session state")
            return False
        
        processed_data = session_state['uploaded_data']
        
        # Validate required columns
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in processed_data.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        print("✅ Required columns validation passed")
        
        # Calculate latest return safely (same logic as in app)
        if 'return' in processed_data.columns and len(processed_data) > 1:
            latest_return = processed_data['return'].iloc[-1] if not processed_data['return'].iloc[-1] != processed_data['return'].iloc[-1] else 0  # Check for NaN
        else:
            # Calculate manually if return column doesn't exist
            if len(processed_data) >= 2:
                latest_close = processed_data['close'].iloc[-1]
                prev_close = processed_data['close'].iloc[-2]
                latest_return = ((latest_close - prev_close) / prev_close) * 100
            else:
                latest_return = 0
        
        print(f"✅ Latest return calculated: {latest_return}")
        
        # Get time duration
        uploaded_time_duration = session_state.get('uploaded_time_duration', 'Unknown duration')
        
        # Prepare data summary (same as in app)
        uploaded_summary = {
            'total_days': len(processed_data),
            'time_duration': uploaded_time_duration,
            'current_price': processed_data['close'].iloc[-1] if len(processed_data) > 0 else 0,
            'latest_change': latest_return,
            'up_days_ratio': 100 * (processed_data['target'] == 1).sum() / len(processed_data) if 'target' in processed_data.columns and len(processed_data) > 0 else 50,
            'highest_price': processed_data['close'].max() if len(processed_data) > 0 else 0,
            'lowest_price': processed_data['close'].min() if len(processed_data) > 0 else 0,
            'avg_volatility': abs(processed_data['return']).mean() if 'return' in processed_data.columns and len(processed_data) > 0 else 0
        }
        
        print("✅ AI prediction data summary prepared successfully:")
        for key, value in uploaded_summary.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in upload processing: {str(e)}")
        return False
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"✅ Cleaned up temp directory")

def main():
    """Main test function."""
    print("🚀 Testing upload CSV processing pipeline...\n")
    
    test_passed = test_upload_processing()
    
    print(f"\n📊 Upload Processing Test: {'✅ PASSED' if test_passed else '❌ FAILED'}")
    
    if test_passed:
        print("\n🎉 Upload CSV processing should work correctly now!")
        print("The 'Error processing files: ['date', 'close']' issue should be fixed.")
    else:
        print("\n⚠️ Upload processing test failed. Please check the error above.")

if __name__ == "__main__":
    main()
