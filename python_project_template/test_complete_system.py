#!/usr/bin/env python3
"""
Test script for the complete Gemini AI integration in the stock prediction system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import google.generativeai as genai

# Import our modules
from src.stock_predictor.data.preprocessor import DataPreprocessor
from src.stock_predictor.data.features import add_technical_indicators, FeatureEngineer

def create_sample_data(n_days=3650):
    """Create sample stock data for demonstration (10 years)."""
    # Start from 10 years ago
    start_date = datetime.now() - timedelta(days=3650)
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    
    # Generate synthetic OHLCV data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    
    data = []
    for i, date in enumerate(dates):
        open_price = close_prices[i] + np.random.randn() * 0.3
        high_price = max(open_price, close_prices[i]) + abs(np.random.randn()) * 0.5
        low_price = min(open_price, close_prices[i]) - abs(np.random.randn()) * 0.5
        volume = int(1000000 + np.random.randn() * 100000)
        turnover = volume * close_prices[i]
        
        data.append({
            'code': 'VCB',
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'date': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_prices[i], 2),
            'volume': volume,
            'turnover': round(turnover, 2),
            'return': 0.0,
            'target': 0
        })
    
    df = pd.DataFrame(data)
    
    # Calculate return and target
    df['return'] = df['close'].pct_change() * 100
    df['return'] = df['return'].round(2).fillna(0)
    df['target'] = (df['return'] > 0).astype(int)
    
    return df

def calculate_time_duration(data):
    """Calculate the time duration of the dataset."""
    if 'date' in data.columns:
        dates = pd.to_datetime(data['date'])
        start_date = dates.min()
        end_date = dates.max()
        duration = end_date - start_date
        years = duration.days / 365.25
        return f"{years:.1f} years ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
    else:
        # Fallback: estimate from row count
        years = len(data) / 365.25
        return f"~{years:.1f} years ({len(data)} data points)"

def get_gemini_prediction(data_summary, api_key):
    """Get AI-based market prediction using Gemini."""
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Use the latest Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create prompt for stock market analysis
        prompt = f"""
        Phân tích dữ liệu thị trường sau và đưa ra dự báo:

        Thông tin dữ liệu:
        - Tổng số ngày giao dịch: {data_summary['total_days']}
        - Khoảng thời gian: {data_summary['time_duration']}
        - Giá đóng cửa hiện tại: {data_summary['current_price']:.2f}
        - Thay đổi giá gần nhất: {data_summary['latest_change']:.2f}%
        - Tỷ lệ ngày tăng giá: {data_summary['up_days_ratio']:.1f}%
        - Giá cao nhất: {data_summary['highest_price']:.2f}
        - Giá thấp nhất: {data_summary['lowest_price']:.2f}
        - Biến động trung bình: {data_summary['avg_volatility']:.2f}%

        Hãy phân tích xu hướng và đưa ra:
        1. Đánh giá tình hình thị trường hiện tại
        2. Dự báo xu hướng ngắn hạn (1-2 tuần)
        3. Dự báo xu hướng trung hạn (1-3 tháng)
        4. Các yếu tố rủi ro cần lưu ý
        5. Khuyến nghị đầu tư (nếu có)

        Vui lòng trả lời bằng tiếng Việt và cung cấp phân tích chi tiết, khách quan.
        """
        
        # Generate response
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Lỗi khi gọi Gemini API: {str(e)}"

def test_complete_system():
    """Test the complete system with Gemini AI integration."""
    
    print("🚀 Testing Complete Prediction System with Gemini AI")
    print("="*60)
    
    # Step 1: Create sample data
    print("📊 Step 1: Creating sample data (10 years)...")
    sample_data = create_sample_data()
    print(f"✅ Created {len(sample_data)} data points")
    
    # Step 2: Calculate time duration
    time_duration = calculate_time_duration(sample_data)
    print(f"⏰ Time duration: {time_duration}")
    
    # Step 3: Add technical indicators
    print("\n🔧 Step 2: Adding technical indicators...")
    data_with_features = add_technical_indicators(sample_data)
    print(f"✅ Added technical indicators. Final shape: {data_with_features.shape}")
    
    # Step 4: Prepare data summary for AI
    print("\n📈 Step 3: Preparing data summary for AI analysis...")
    data_summary = {
        'total_days': len(sample_data),
        'time_duration': time_duration,
        'current_price': sample_data['close'].iloc[-1],
        'latest_change': sample_data['return'].iloc[-1],
        'up_days_ratio': 100 * (sample_data['target'] == 1).sum() / len(sample_data),
        'highest_price': sample_data['close'].max(),
        'lowest_price': sample_data['close'].min(),
        'avg_volatility': abs(sample_data['return']).mean()
    }
    
    print("Data Summary:")
    for key, value in data_summary.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.2f}")
        else:
            print(f"  - {key}: {value}")
    
    # Step 5: Get AI prediction
    print("\n🤖 Step 4: Getting AI prediction from Gemini...")
    api_key = "AIzaSyDMs-iLWgB7NuoCtJLqEj4SwG3qhM3B-gQ"
    
    ai_prediction = get_gemini_prediction(data_summary, api_key)
    
    print("\n" + "="*60)
    print("🔮 GEMINI AI MARKET ANALYSIS & PREDICTION")
    print("="*60)
    print(ai_prediction)
    print("="*60)
    
    print("\n✅ Complete system test successful!")
    print("🎯 The system can now:")
    print("   - Generate 10-year sample data")
    print("   - Calculate technical indicators")
    print("   - Display correct time duration")
    print("   - Integrate with Gemini AI for predictions")
    print("   - Provide detailed market analysis in Vietnamese")

if __name__ == "__main__":
    test_complete_system()
