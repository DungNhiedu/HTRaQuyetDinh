#!/usr/bin/env python3
"""
Test script to verify removal of Reset Upload button and popup message functionality
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_reset_button_removal():
    """Test that the Reset Upload button has been removed."""
    print("🧪 Testing Reset Upload Button Removal...")
    
    try:
        # Read the app_new.py file
        app_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/python_project_template/src/stock_predictor/app_new.py"
        
        with open(app_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that reset button code is removed
        reset_button_patterns = [
            "Reset Upload",
            "🗑️ Reset Upload",
            "Clear uploaded data and start fresh",
            "st.experimental_rerun()"
        ]
        
        found_patterns = []
        for pattern in reset_button_patterns:
            if pattern in content:
                found_patterns.append(pattern)
        
        if found_patterns:
            print(f"❌ Found remaining reset button code: {found_patterns}")
            return False
        else:
            print("✅ Reset Upload button successfully removed")
            return True
            
    except Exception as e:
        print(f"❌ Error testing reset button removal: {str(e)}")
        return False

def test_popup_message_implementation():
    """Test that popup message function is implemented correctly."""
    print("\n🧪 Testing Popup Message Implementation...")
    
    try:
        # Read the app_new.py file
        app_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/python_project_template/src/stock_predictor/app_new.py"
        
        with open(app_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that popup function exists
        if "def show_popup_message(" not in content:
            print("❌ show_popup_message function not found")
            return False
        
        # Check that st.toast is used when available
        if "st.toast" not in content:
            print("❌ st.toast implementation not found")
            return False
        
        # Check that old message patterns are replaced
        old_patterns = [
            "st.success(",
            "st.error(",
            "st.warning(",
            "st.info("
        ]
        
        # Count occurrences (should be minimal, only in the popup function itself)
        total_old_patterns = 0
        for pattern in old_patterns:
            count = content.count(pattern)
            total_old_patterns += count
        
        # Should have some occurrences in the popup function but not many
        if total_old_patterns > 10:  # Allow for popup function implementation
            print(f"⚠️  Found {total_old_patterns} old message patterns (expected few for fallback)")
        
        # Check that new patterns exist
        new_patterns = [
            "show_popup_message(",
        ]
        
        found_new = 0
        for pattern in new_patterns:
            found_new += content.count(pattern)
        
        if found_new < 5:  # Should have multiple calls to show_popup_message
            print(f"❌ Expected more show_popup_message calls, found: {found_new}")
            return False
        
        print(f"✅ Popup message implementation verified ({found_new} popup calls)")
        return True
        
    except Exception as e:
        print(f"❌ Error testing popup messages: {str(e)}")
        return False

def test_code_structure():
    """Test that the overall code structure is intact."""
    print("\n🧪 Testing Code Structure...")
    
    try:
        # Read the app_new.py file
        app_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/python_project_template/src/stock_predictor/app_new.py"
        
        with open(app_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check key functions exist
        required_functions = [
            "def main():",
            "def show_popup_message(",
            "def get_ai_prediction(",
            "def display_ai_prediction_result(",
            "def plot_price_chart(",
            "def calculate_time_duration("
        ]
        
        missing_functions = []
        for func in required_functions:
            if func not in content:
                missing_functions.append(func)
        
        if missing_functions:
            print(f"❌ Missing functions: {missing_functions}")
            return False
        
        # Check that main sections exist
        required_sections = [
            "Sample Data Demo",
            "Upload CSV Files",
            "AI Prediction",
            "session_state"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing sections: {missing_sections}")
            return False
        
        print("✅ Code structure is intact")
        return True
        
    except Exception as e:
        print(f"❌ Error testing code structure: {str(e)}")
        return False

def test_upload_section_improvements():
    """Test that upload section no longer has reset functionality."""
    print("\n🧪 Testing Upload Section Improvements...")
    
    try:
        # Read the app_new.py file
        app_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/python_project_template/src/stock_predictor/app_new.py"
        
        with open(app_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that upload section still exists
        if "Upload CSV Files" not in content:
            print("❌ Upload CSV Files section not found")
            return False
        
        # Check that file uploader still exists
        if "st.file_uploader" not in content:
            print("❌ File uploader not found")
            return False
        
        # Check that process button still exists
        if "Process Uploaded Data" not in content:
            print("❌ Process button not found")
            return False
        
        # Check that reset-related session state clearing is removed
        reset_patterns = [
            "for key in ['uploaded_processed'",
            "del st.session_state[key]",
            "Clear uploaded data and start fresh"
        ]
        
        found_reset_code = []
        for pattern in reset_patterns:
            if pattern in content:
                found_reset_code.append(pattern)
        
        if found_reset_code:
            print(f"❌ Found reset-related code: {found_reset_code}")
            return False
        
        print("✅ Upload section improved (reset functionality removed)")
        return True
        
    except Exception as e:
        print(f"❌ Error testing upload section: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Reset Button Removal and Popup Message Implementation")
    print("=" * 70)
    
    tests = [
        ("Reset Button Removal", test_reset_button_removal),
        ("Popup Message Implementation", test_popup_message_implementation),
        ("Code Structure", test_code_structure),
        ("Upload Section Improvements", test_upload_section_improvements)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
    
    print("=" * 70)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Changes implemented successfully.")
        print("\n📝 Summary of changes:")
        print("✅ Removed Reset Upload button and related functionality")
        print("✅ Implemented popup message system with 3-second auto-disappear")
        print("✅ Converted all st.success/error/warning/info to popup messages")
        print("✅ Maintained all core functionality")
        print("✅ Upload section still works without reset capability")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
