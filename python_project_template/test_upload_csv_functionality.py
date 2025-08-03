#!/usr/bin/env python3
"""
Test script to verify Upload CSV functionality
"""

import sys
import os
import pandas as pd
import tempfile

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_upload_csv_logic():
    """Test the upload CSV processing logic."""
    
    print("🚀 Testing Upload CSV Logic...")
    
    # Create a test CSV file
    test_data = {
        'Date': ['01/01/2023', '02/01/2023', '03/01/2023', '04/01/2023', '05/01/2023'],
        'Close': ['23000', '23100', '22950', '23200', '23150'],
        'Open': ['22950', '23000', '23100', '22900', '23180'],
        'High': ['23050', '23150', '23100', '23250', '23200'],
        'Low': ['22900', '22980', '22900', '22850', '23100'],
        'Volume': ['1000000', '1200000', '950000', '1100000', '1050000']
    }
    
    df = pd.DataFrame(test_data)
    
    # Save to temporary CSV file (VN30 format with semicolon separator)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write VN30 format
        f.write("Date;Close;Open;High;Low;Volume\n")
        for _, row in df.iterrows():
            f.write(f"{row['Date']};{row['Close']};{row['Open']};{row['High']};{row['Low']};{row['Volume']}\n")
        test_file_path = f.name
    
    print(f"✅ Created test CSV file: {test_file_path}")
    
    # Test DataPreprocessor
    try:
        from data.preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Test reading the CSV
        data = preprocessor._read_csv_flexible(test_file_path)
        if data is not None and not data.empty:
            print("✅ CSV file read successfully")
            print(f"   Shape: {data.shape}")
            print(f"   Columns: {list(data.columns)}")
        else:
            print("❌ Failed to read CSV file")
            return False
        
        # Test normalization
        normalized_data = preprocessor._normalize_data_format(data, "TEST")
        if normalized_data is not None and not normalized_data.empty:
            print("✅ Data normalization successful")
            print(f"   Normalized shape: {normalized_data.shape}")
            print(f"   Normalized columns: {list(normalized_data.columns)}")
        else:
            print("❌ Data normalization failed")
            return False
        
        # Test returns and targets calculation
        final_data = preprocessor._calculate_returns_and_targets(normalized_data)
        if final_data is not None and not final_data.empty:
            print("✅ Returns and targets calculation successful")
            print(f"   Final shape: {final_data.shape}")
            print(f"   Final columns: {list(final_data.columns)}")
            
            # Check for required columns
            required_cols = ['close', 'return', 'target']
            missing_cols = [col for col in required_cols if col not in final_data.columns]
            if not missing_cols:
                print("✅ All required columns present")
            else:
                print(f"❌ Missing required columns: {missing_cols}")
                return False
        else:
            print("❌ Returns and targets calculation failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return False
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.unlink(test_file_path)
            print(f"🗑️ Cleaned up test file: {test_file_path}")
    
    print("✅ Upload CSV logic test completed successfully!")
    return True

def test_streamlit_app_structure():
    """Test if the Streamlit app has correct structure."""
    
    print("\n🚀 Testing Streamlit App Structure...")
    
    app_file = os.path.join('src', 'stock_predictor', 'app.py')
    
    if not os.path.exists(app_file):
        print(f"❌ App file not found: {app_file}")
        return False
    
    with open(app_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for key elements in Upload CSV section
    upload_checks = [
        'demo_option == "Tải Lên File CSV"',  # Check for Vietnamese option
        'st.file_uploader(',  # File uploader
        'process_uploaded_data_button',  # Process button
        'DataPreprocessor()',  # DataPreprocessor usage
        'add_technical_indicators',  # Technical indicators
        'use_ai_prediction and st.session_state.get(\'upload_processed\'',  # AI prediction logic
        'show_popup_message',  # Popup messages
    ]
    
    missing_elements = []
    for check in upload_checks:
        if check not in content:
            missing_elements.append(check)
    
    if missing_elements:
        print("❌ Missing elements in Upload CSV section:")
        for element in missing_elements:
            print(f"   - {element}")
        return False
    
    print("✅ All required elements found in Upload CSV section!")
    
    # Check for potential issues
    potential_issues = []
    
    # Check for duplicate logic
    if content.count('st.button("🧠 Nhận Dự Báo AI') > 0:
        potential_issues.append("Found separate AI prediction button (should use sidebar)")
    
    if content.count('st.session_state.get(\'upload_processed\', False)') > 3:
        potential_issues.append("Multiple session state checks (potential duplication)")
    
    if potential_issues:
        print("⚠️ Potential issues found:")
        for issue in potential_issues:
            print(f"   - {issue}")
    else:
        print("✅ No potential issues found!")
    
    return True

def test_ai_prediction_integration():
    """Test AI prediction integration."""
    
    print("\n🚀 Testing AI Prediction Integration...")
    
    app_file = os.path.join('src', 'stock_predictor', 'app.py')
    
    with open(app_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check AI prediction logic
    ai_checks = [
        'has_uploaded_data = (demo_option == "Tải Lên File CSV"',  # Check for uploaded data
        'use_ai_prediction and st.session_state.get(\'upload_processed\'',  # Conditional AI prediction
        'get_gemini_prediction',  # Gemini API call
        'format_gemini_response',  # Response formatting
    ]
    
    missing_ai_elements = []
    for check in ai_checks:
        if check not in content:
            missing_ai_elements.append(check)
    
    if missing_ai_elements:
        print("❌ Missing AI prediction elements:")
        for element in missing_ai_elements:
            print(f"   - {element}")
        return False
    
    print("✅ AI prediction integration looks good!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 UPLOAD CSV FUNCTIONALITY TEST")
    print("=" * 60)
    
    success = True
    
    # Test upload CSV logic
    if not test_upload_csv_logic():
        success = False
    
    # Test Streamlit app structure
    if not test_streamlit_app_structure():
        success = False
    
    # Test AI prediction integration
    if not test_ai_prediction_integration():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Upload CSV functionality should work correctly.")
        print("\n💡 To test the app:")
        print("   1. Run: streamlit run src/stock_predictor/app.py")
        print("   2. Select 'Tải Lên File CSV' from the sidebar")
        print("   3. Upload a CSV file")
        print("   4. Click 'Xử Lý Dữ Liệu Đã Tải Lên'")
        print("   5. Use 'Nhận Dự Báo Thị Trường AI' from sidebar for AI prediction")
    else:
        print("❌ SOME TESTS FAILED! Please review the Upload CSV functionality.")
    print("=" * 60)
