"""
Test script for Gemini API integration
"""

import google.generativeai as genai

def test_gemini_connection():
    """Test Gemini API connection with the provided API key"""
    api_key = "AIzaSyDMs-iLWgB7NuoCtJLqEj4SwG3qhM3B-gQ"
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # List available models first
        print("Available models:")
        for model in genai.list_models():
            print(f"  - {model.name}")
        
        # Try with the newer model name
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Simple test prompt
        response = model.generate_content("Chào Gemini, bạn có thể làm gì?")
        
        print("✅ Gemini API connection successful!")
        print("Response:", response.text[:200] + "..." if len(response.text) > 200 else response.text)
        return True
        
    except Exception as e:
        print(f"❌ Gemini API connection failed: {str(e)}")
        return False

def test_stock_analysis():
    """Test stock analysis with Gemini"""
    api_key = "AIzaSyDMs-iLWgB7NuoCtJLqEj4SwG3qhM3B-gQ"
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Stock analysis prompt
        prompt = """
        Phân tích dữ liệu thị trường chứng khoán sau và đưa ra dự báo:

        Thông tin dữ liệu:
        - Tổng số ngày giao dịch: 3650
        - Khoảng thời gian: 10.0 years
        - Giá đóng cửa hiện tại: 120.50
        - Thay đổi giá gần nhất: 0.75%
        - Tỷ lệ ngày tăng giá: 52.1%
        - Giá cao nhất: 145.30
        - Giá thấp nhất: 85.20
        - Biến động trung bình: 1.2%

        Hãy phân tích xu hướng và đưa ra dự báo ngắn gọn (khoảng 3-4 câu).
        """
        
        response = model.generate_content(prompt)
        
        print("✅ Stock analysis test successful!")
        print("Analysis response:", response.text)
        return True
        
    except Exception as e:
        print(f"❌ Stock analysis test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Gemini API integration...")
    print("="*50)
    
    # Test basic connection
    print("1. Testing basic connection...")
    test_gemini_connection()
    
    print("\n" + "="*50)
    
    # Test stock analysis
    print("2. Testing stock analysis...")
    test_stock_analysis()
    
    print("\n" + "="*50)
    print("Testing completed!")
