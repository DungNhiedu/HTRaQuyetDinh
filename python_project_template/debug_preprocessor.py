#!/usr/bin/env python3
"""
Debug script to test DataPreprocessor directly
"""

import sys
import os
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_preprocessor():
    """Test DataPreprocessor with the test file."""
    
    print("🚀 Testing DataPreprocessor directly...")
    
    try:
        # Import from the correct path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'stock_predictor'))
        from data.preprocessor import DataPreprocessor
        
        # Test file path
        test_file = os.path.join(os.path.dirname(__file__), 'test_upload.csv')
        
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            return False
        
        print(f"✅ Test file found: {test_file}")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Test reading CSV
        print("\n📖 Testing CSV reading...")
        data = preprocessor._read_csv_flexible(test_file)
        
        if data is not None and not data.empty:
            print("✅ CSV reading successful")
            print(f"   Shape: {data.shape}")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Sample data:\n{data.head()}")
        else:
            print("❌ CSV reading failed")
            return False
        
        # Test normalization
        print("\n🔄 Testing data normalization...")
        normalized = preprocessor._normalize_data_format(data, "TEST")
        
        if normalized is not None and not normalized.empty:
            print("✅ Data normalization successful")
            print(f"   Normalized shape: {normalized.shape}")
            print(f"   Normalized columns: {list(normalized.columns)}")
            print(f"   Sample normalized data:\n{normalized.head()}")
        else:
            print("❌ Data normalization failed")
            return False
        
        # Test returns calculation
        print("\n📊 Testing returns calculation...")
        final_data = preprocessor._calculate_returns_and_targets(normalized)
        
        if final_data is not None and not final_data.empty:
            print("✅ Returns calculation successful")
            print(f"   Final shape: {final_data.shape}")
            print(f"   Final columns: {list(final_data.columns)}")
            print(f"   Sample final data:\n{final_data.head()}")
            
            # Check required columns
            required = ['close', 'return', 'target']
            missing = [col for col in required if col not in final_data.columns]
            if not missing:
                print("✅ All required columns present")
            else:
                print(f"❌ Missing columns: {missing}")
                return False
        else:
            print("❌ Returns calculation failed")
            return False
        
        print("\n🎉 DataPreprocessor test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Check if data/preprocessor.py exists and is accessible")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_file_structure():
    """Check file structure."""
    
    print("\n🔍 Checking file structure...")
    
    required_files = [
        'src/stock_predictor/data/preprocessor.py',
        'src/stock_predictor/data/features.py',
        'src/stock_predictor/app.py',
        'test_upload.csv'
    ]
    
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 DATAPROCESSOR DEBUG TEST")
    print("=" * 60)
    
    check_file_structure()
    test_preprocessor()
    
    print("=" * 60)
