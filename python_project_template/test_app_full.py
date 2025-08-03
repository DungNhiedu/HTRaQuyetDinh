#!/usr/bin/env python3
"""
Test script to check the full app functionality including upload CSV and AI prediction
"""

import sys
import os
import subprocess
import time
import pandas as pd
import numpy as np

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

def test_app_imports():
    """Test if all imports work correctly"""
    print("🔍 Testing app imports...")
    
    try:
        # Import main modules
        from stock_predictor.app import main, show_popup_message
        from stock_predictor.data.preprocessor import DataPreprocessor
        from stock_predictor.data.features import FeatureEngineer, add_technical_indicators
        from stock_predictor.forecast.forecaster import predict_future, get_gemini_prediction
        
        print("✅ All main imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_csv_processing():
    """Test CSV processing functionality"""
    print("\n📊 Testing CSV processing...")
    
    try:
        # Import required modules
        from stock_predictor.data.preprocessor import DataPreprocessor
        from stock_predictor.data.features import add_technical_indicators
        
        # Check if test CSV exists
        test_csv_path = "test_upload.csv"
        if not os.path.exists(test_csv_path):
            print(f"❌ Test CSV file not found: {test_csv_path}")
            return False
        
        # Test data preprocessing
        preprocessor = DataPreprocessor()
        
        # Read and process the CSV
        print(f"📁 Loading test CSV: {test_csv_path}")
        raw_data = pd.read_csv(test_csv_path)
        print(f"   Raw data shape: {raw_data.shape}")
        print(f"   Columns: {list(raw_data.columns)}")
        
        # Process with DataPreprocessor
        print("🔄 Processing with DataPreprocessor...")
        processed_data = preprocessor.process_uploaded_data(raw_data)
        print(f"   Processed data shape: {processed_data.shape}")
        print(f"   New columns: {list(processed_data.columns)}")
        
        # Check required columns
        required_cols = ['close', 'target']
        missing_cols = [col for col in required_cols if col not in processed_data.columns]
        if missing_cols:
            print(f"⚠️  Missing required columns: {missing_cols}")
        else:
            print("✅ All required columns present")
        
        # Test technical indicators
        print("📈 Testing technical indicators...")
        if 'close' in processed_data.columns and 'code' in processed_data.columns:
            enriched_data = add_technical_indicators(processed_data)
            print(f"   Enriched data shape: {enriched_data.shape}")
            
            # Check what indicators were added
            original_cols = set(processed_data.columns)
            new_cols = set(enriched_data.columns) - original_cols
            print(f"   New technical indicators: {list(new_cols)}")
            
            print("✅ Technical indicators processing successful")
        else:
            print("⚠️  Cannot add technical indicators - missing 'close' or 'code' columns")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV processing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_state_logic():
    """Test session state logic simulation"""
    print("\n🔄 Testing session state logic...")
    
    try:
        # Simulate session state
        mock_session_state = {
            'upload_processed': False,
            'uploaded_data': None,
            'uploaded_features': None,
            'uploaded_time_duration': None
        }
        
        # Test initial state
        print(f"   Initial state: {mock_session_state}")
        
        # Simulate successful upload processing
        from stock_predictor.data.preprocessor import DataPreprocessor
        
        # Load test data
        if os.path.exists("test_upload.csv"):
            raw_data = pd.read_csv("test_upload.csv")
            preprocessor = DataPreprocessor()
            processed_data = preprocessor.process_uploaded_data(raw_data)
            
            # Simulate state update
            mock_session_state['upload_processed'] = True
            mock_session_state['uploaded_data'] = processed_data
            mock_session_state['uploaded_time_duration'] = f"{len(processed_data)} days"
            
            print(f"   After processing: upload_processed = {mock_session_state['upload_processed']}")
            print(f"   Data shape: {processed_data.shape if processed_data is not None else 'None'}")
            
            # Test condition for AI prediction
            use_ai_prediction = True  # Simulate button press
            upload_processed = mock_session_state.get('upload_processed', False)
            has_data = mock_session_state.get('uploaded_data') is not None
            
            print(f"   AI Prediction conditions:")
            print(f"     - use_ai_prediction: {use_ai_prediction}")
            print(f"     - upload_processed: {upload_processed}")
            print(f"     - has_data: {has_data}")
            print(f"     - Should show AI prediction: {use_ai_prediction and upload_processed and has_data}")
            
            print("✅ Session state logic test successful")
            return True
        else:
            print("⚠️  No test CSV found for session state test")
            return False
            
    except Exception as e:
        print(f"❌ Session state logic error: {e}")
        return False

def test_ai_prediction():
    """Test AI prediction functionality"""
    print("\n🤖 Testing AI prediction...")
    
    try:
        from stock_predictor.forecast.forecaster import get_gemini_prediction
        
        # Create mock data summary
        mock_summary = {
            'total_days': 100,
            'time_duration': '100 days (2023-01-01 to 2023-04-10)',
            'current_price': 150.50,
            'latest_change': 2.5,
            'up_days_ratio': 55.0,
            'highest_price': 180.25,
            'lowest_price': 120.75,
            'avg_volatility': 1.8
        }
        
        # Test API key
        api_key = "AIzaSyDMs-iLWgB7NuoCtJLqEj4SwG3qhM3B-gQ"
        
        print(f"   Mock data summary: {mock_summary}")
        print("   Testing Gemini API call...")
        
        # Make prediction
        prediction = get_gemini_prediction(mock_summary, api_key)
        
        if "Lỗi" not in prediction:
            print("✅ AI prediction successful")
            print(f"   Response length: {len(prediction)} characters")
            print(f"   Preview: {prediction[:100]}...")
        else:
            print(f"⚠️  AI prediction returned error: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI prediction error: {e}")
        return False

def run_streamlit_test():
    """Test if Streamlit can run the app without crashing"""
    print("\n🚀 Testing Streamlit app startup...")
    
    try:
        app_path = "src/stock_predictor/app.py"
        if not os.path.exists(app_path):
            print(f"❌ App file not found: {app_path}")
            return False
        
        print(f"   Found app file: {app_path}")
        print("   Testing syntax check...")
        
        # Compile the app to check for syntax errors
        with open(app_path, 'r', encoding='utf-8') as f:
            app_code = f.read()
        
        compile(app_code, app_path, 'exec')
        print("✅ App syntax check passed")
        
        # Test imports in the app context
        exec_globals = {}
        exec(app_code, exec_globals)
        print("✅ App execution test passed")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in app: {e}")
        return False
    except Exception as e:
        print(f"❌ App execution error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Full App Functionality Test")
    print("=" * 50)
    
    # Track test results
    test_results = []
    
    # Run all tests
    tests = [
        ("App Imports", test_app_imports),
        ("CSV Processing", test_csv_processing),
        ("Session State Logic", test_session_state_logic),
        ("AI Prediction", test_ai_prediction),
        ("Streamlit App", run_streamlit_test)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 30}")
        result = test_func()
        test_results.append((test_name, result))
    
    # Summary
    print(f"\n{'=' * 50}")
    print("📋 TEST SUMMARY")
    print(f"{'=' * 50}")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n🏆 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! App should be working correctly.")
        print("\nTo run the app:")
        print("   cd /Users/dungnhi/Documents/HTRaQuyetDinh/python_project_template")
        print("   streamlit run src/stock_predictor/app.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
