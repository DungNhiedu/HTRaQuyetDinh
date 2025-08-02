#!/usr/bin/env python3
"""
Test script for Gemini AI integration
Tests the AI prediction functionality with the provided API key
"""

import os
import google.generativeai as genai

def test_gemini_api():
    """Test the Gemini API with the provided key."""
    
    # Configure API key
    api_key = "AIzaSyDMs-iLWgB7NuoCtJLqEj4SwG3qhM3B-gQ"
    genai.configure(api_key=api_key)
    
    try:
        # List available models
        print("Available models:")
        for model in genai.list_models():
            print(f"- {model.name}")
        
        # Try with gemini-1.5-flash (newer model)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Test prompt
        test_prompt = """
        Phân tích dữ liệu thị trường chứng khoán sau và đưa ra dự báo:

        Thông tin dữ liệu:
        - Tổng số ngày giao dịch: 3650
        - Khoảng thời gian: 10.0 years
        - Giá đóng cửa hiện tại: 115.25
        - Thay đổi giá gần nhất: 1.45%
        - Tỷ lệ ngày tăng giá: 52.1%
        - Giá cao nhất: 145.80
        - Giá thấp nhất: 85.30
        - Biến động trung bình: 1.8%

        Hãy phân tích xu hướng và đưa ra dự báo ngắn gọn.
        """
        
        # Generate response
        response = model.generate_content(test_prompt)
        
        print("\n" + "="*50)
        print("GEMINI AI RESPONSE:")
        print("="*50)
        print(response.text)
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"Error testing Gemini API: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Gemini AI integration...")
    success = test_gemini_api()
    if success:
        print("\n✅ Gemini API integration test successful!")
    else:
        print("\n❌ Gemini API integration test failed!")
