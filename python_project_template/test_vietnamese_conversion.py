#!/usr/bin/env python3
"""
Test script to verify Vietnamese UI conversion in the Streamlit app
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_vietnamese_conversion():
    """Test if the Vietnamese conversion is working correctly."""
    
    print("🚀 Testing Vietnamese UI Conversion...")
    
    # Check if the file contains Vietnamese text
    app_file = os.path.join('src', 'stock_predictor', 'app.py')
    
    if not os.path.exists(app_file):
        print(f"❌ App file not found: {app_file}")
        return False
    
    with open(app_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for key Vietnamese phrases
    vietnamese_checks = [
        "Hệ Thống Dự Báo Thị Trường Chứng Khoán",  # Page title
        "Demo Dữ Liệu Mẫu",  # Sample data demo
        "Tải Lên File CSV",  # Upload CSV files
        "Demo Dự Báo",  # Forecast demo
        "Phân Tích Dữ Liệu Mẫu",  # Sample data analysis
        "Kỹ Thuật Xây Dựng Đặc Trưng",  # Feature engineering
        "Biểu Đồ & Phân Tích",  # Charts & Analysis
        "Demo Huấn Luyện Mô Hình",  # Model training demo
        "Dự Báo Thị Trường Dựa Trên AI",  # AI-based market prediction
        "Phân Tích Kỹ Thuật",  # Technical analysis
        "show_popup_message",  # Popup message function
        "Đang tải dữ liệu VN30...",  # Loading VN30 data
        "Đã tải thành công dữ liệu VN30!",  # Success message
        "Chỉ Số VN30 với Các Chỉ Báo Kỹ Thuật",  # Chart title
        "Chỉ Báo Kỹ Thuật RSI",  # RSI indicator
        "Mua quá mức",  # Overbought
        "Bán quá mức",  # Oversold
        "Nhận Dự Báo Thị Trường AI",  # AI prediction button
        "Chọn Kiểu Demo",  # Choose demo type
        "Tổng Số Ngày",  # Total days
        "Giá Mới Nhất",  # Latest price
        "Thay Đổi Giá %",  # Price change %
        "% Ngày Tăng",  # Up days %
        "Chọn file CSV",  # Choose CSV files
        "Xử Lý Dữ Liệu Đã Tải Lên",  # Process uploaded data
        "Đang xử lý file CSV đã tải lên...",  # Processing CSV files
        "Chạy Demo Huấn Luyện Mô Hình",  # Run model training demo
        "Nhận Dự Báo AI cho Dữ Liệu Của Bạn",  # Get AI prediction for your data
    ]
    
    missing_phrases = []
    for phrase in vietnamese_checks:
        if phrase not in content:
            missing_phrases.append(phrase)
    
    if missing_phrases:
        print("❌ Missing Vietnamese phrases:")
        for phrase in missing_phrases:
            print(f"   - {phrase}")
        return False
    
    print("✅ All Vietnamese phrases found!")
    
    # Check for old English phrases that should be replaced
    english_checks = [
        "Stock Market Prediction Demo",  # Should be replaced
        "Sample Data Demo",  # Should be replaced
        "Upload CSV Files",  # Should be replaced (in selectbox)
        "Choose Demo Type",  # Should be replaced
        "Loading VN30 data...",  # Should be replaced
        "VN30 Index with Technical Indicators",  # Should be replaced
        "RSI Technical Indicator",  # Should be replaced
        "Get AI Market Prediction",  # Should be replaced
        "Run Model Training Demo",  # Should be replaced (button text)
        "Process Uploaded Data",  # Should be replaced
    ]
    
    remaining_english = []
    for phrase in english_checks:
        if phrase in content:
            remaining_english.append(phrase)
    
    if remaining_english:
        print("⚠️ Found remaining English phrases that should be translated:")
        for phrase in remaining_english:
            print(f"   - {phrase}")
    else:
        print("✅ No old English phrases found!")
    
    # Check if show_popup_message function exists
    if "def show_popup_message(" in content:
        print("✅ show_popup_message function found!")
    else:
        print("❌ show_popup_message function not found!")
        return False
    
    # Check if forecast demo is properly integrated
    if "Demo Dự Báo" in content and "from forecast.forecaster import StockForecaster" in content:
        print("✅ Forecast demo integration found!")
    else:
        print("❌ Forecast demo integration not complete!")
        return False
    
    print("\n🎉 Vietnamese UI conversion test completed successfully!")
    return True

def test_forecast_module():
    """Test if the forecast module has Vietnamese text."""
    
    print("\n🚀 Testing Forecast Module Vietnamese conversion...")
    
    forecaster_file = os.path.join('src', 'stock_predictor', 'forecast', 'forecaster.py')
    
    if not os.path.exists(forecaster_file):
        print(f"❌ Forecaster file not found: {forecaster_file}")
        return False
    
    with open(forecaster_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for Vietnamese phrases in forecaster
    vietnamese_forecast_checks = [
        "Giá Lịch sử",  # Historical prices
        "Dự báo",  # Forecast
        "Khoảng Tin cậy",  # Confidence interval
        "Dự báo Giá",  # Price forecast
        "Thời gian",  # Time
        "Giá",  # Price
        "Tăng mạnh",  # Strong increase
        "Tăng nhẹ",  # Light increase
        "Đi ngang",  # Sideways
        "Giảm",  # Decrease
    ]
    
    missing_forecast_phrases = []
    for phrase in vietnamese_forecast_checks:
        if phrase not in content:
            missing_forecast_phrases.append(phrase)
    
    if missing_forecast_phrases:
        print("❌ Missing Vietnamese phrases in forecaster:")
        for phrase in missing_forecast_phrases:
            print(f"   - {phrase}")
        return False
    
    print("✅ All Vietnamese phrases found in forecaster!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 VIETNAMESE UI CONVERSION TEST")
    print("=" * 60)
    
    success = True
    
    # Test main app conversion
    if not test_vietnamese_conversion():
        success = False
    
    # Test forecast module conversion
    if not test_forecast_module():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Vietnamese conversion is complete.")
    else:
        print("❌ SOME TESTS FAILED! Please review the conversion.")
    print("=" * 60)
