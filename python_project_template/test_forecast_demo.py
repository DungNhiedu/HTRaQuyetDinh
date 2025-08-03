#!/usr/bin/env python3
"""
Test script for the Forecast Demo functionality
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stock_predictor.forecast.forecaster import StockForecaster

def test_forecast_demo():
    """Test the forecast demo functionality."""
    print("=== Testing Forecast Demo ===")
    
    # Initialize forecaster
    forecaster = StockForecaster()
    
    # Test loading forecast data
    print("\n1. Testing data loading...")
    data_loaded = forecaster.load_forecast_data()
    
    if data_loaded:
        print(f"✅ Successfully loaded data for {len(forecaster.available_symbols)} symbols: {forecaster.available_symbols}")
        
        # Test each available symbol
        for symbol in forecaster.available_symbols:
            print(f"\n2. Testing forecast for {symbol}...")
            
            # Get historical data
            historical = forecaster.get_historical_data(symbol, days=30)
            if not historical.empty:
                print(f"   ✅ Historical data: {len(historical)} records")
                print(f"   📅 Date range: {historical['Date'].min()} to {historical['Date'].max()}")
                print(f"   💰 Latest price: {historical['Close'].iloc[-1]:,.0f}")
            
            # Create forecast model
            model_created = forecaster.create_forecast_model(symbol, forecast_days=30)
            if model_created:
                print(f"   ✅ Forecast model created")
                
                # Generate forecast
                forecast = forecaster.generate_forecast(symbol, forecast_days=30)
                if forecast is not None:
                    print(f"   ✅ Forecast generated: {len(forecast)} days")
                    print(f"   📈 Price range: {forecast['Predicted_Price'].min():,.0f} - {forecast['Predicted_Price'].max():,.0f}")
                    
                    # Get forecast summary
                    summary = forecaster.get_forecast_summary(symbol, forecast_days=30)
                    if summary:
                        print(f"   📊 Price change prediction: {summary['price_change_pct']:+.1f}%")
                        print(f"   🎯 Trend: {summary['trend']}")
                    
                    # Test chart creation
                    chart = forecaster.create_forecast_chart(symbol, forecast_days=30, historical_days=60)
                    if chart:
                        print(f"   ✅ Forecast chart created")
                    else:
                        print(f"   ❌ Failed to create forecast chart")
                
                else:
                    print(f"   ❌ Failed to generate forecast")
            else:
                print(f"   ❌ Failed to create forecast model")
                
    else:
        print("❌ Failed to load forecast data")
        print("💡 Make sure these files exist on Desktop:")
        print("   • Dữ liệu Lịch sử USD_VND.csv")
        print("   • dữ liệu lịch sử giá vàng.csv")

def test_file_availability():
    """Test if the required CSV files are available."""
    print("\n=== Testing File Availability ===")
    
    desktop_path = "/Users/dungnhi/Desktop"
    required_files = [
        "Dữ liệu Lịch sử USD_VND.csv",
        "dữ liệu lịch sử giá vàng.csv"
    ]
    
    for filename in required_files:
        file_path = os.path.join(desktop_path, filename)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {filename} - Size: {file_size:,} bytes")
        else:
            print(f"❌ {filename} - Not found")

if __name__ == "__main__":
    test_file_availability()
    test_forecast_demo()
    print("\n🎉 Forecast demo test completed!")
