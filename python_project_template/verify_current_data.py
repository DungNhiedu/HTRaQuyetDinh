#!/usr/bin/env python3
"""
Quick verification of what data is now being used in the Sample Data Demo.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from src.stock_predictor.data.preprocessor import DataPreprocessor
from src.stock_predictor.data.features import add_technical_indicators

def verify_current_data():
    """Verify what data is currently being used."""
    print("🔍 Verifying Current VN30 Data Usage...")
    
    vn30_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/Dữ liệu Lịch sử VN 30.csv"
    
    try:
        # Load and process VN30 data using the same logic as the app
        preprocessor = DataPreprocessor()
        vn30_data = preprocessor._read_csv_flexible(vn30_file_path)
        
        if vn30_data is None:
            raise Exception("Could not read VN30 CSV file")
        
        # Process the VN30 data directly
        vn30_data = preprocessor._normalize_data_format(vn30_data, "VN30")
        
        if vn30_data is None or vn30_data.empty:
            raise Exception("Failed to normalize VN30 data format")
        
        # Calculate returns and targets
        vn30_data = preprocessor._calculate_returns_and_targets(vn30_data)
        
        sample_data = vn30_data.copy()
        
        print(f"✅ Loaded VN30 data! Shape: {sample_data.shape}")
        
        # What would be displayed in the Sample Data Analysis
        print(f"\n📊 Sample Data Analysis (what app shows):")
        print(f"   Data Source: Vietnam VN30 Index Historical Data")
        print(f"   Records: {len(sample_data)}")
        
        # Show sample data (first 10 rows)
        print(f"\n📋 Raw VN30 Data (First 10 rows):")
        display_data = sample_data.head(10).copy()
        if 'date' in display_data.columns:
            display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
        print(display_data[['date', 'close', 'open', 'high', 'low', 'volume']])
        
        # Add technical indicators
        data_with_features = add_technical_indicators(sample_data)
        print(f"\n🔧 Technical indicators added: {data_with_features.shape}")
        
        # Check what indicators are available
        original_cols = set(sample_data.columns)
        new_cols = set(data_with_features.columns) - original_cols
        
        ma_indicators = [col for col in new_cols if 'ma_' in col.lower()]
        rsi_indicators = [col for col in new_cols if 'rsi' in col.lower()]
        bb_indicators = [col for col in new_cols if 'bb_' in col.lower()]
        
        print(f"\n📈 Available for charts:")
        print(f"   MA Indicators: {ma_indicators}")
        print(f"   RSI Indicators: {rsi_indicators}")
        print(f"   Bollinger Bands: {bb_indicators}")
        
        # Show final dataset preview
        print(f"\n📋 Final dataset preview (Real VN30 Data):")
        print(data_with_features.head()[['date', 'close', 'ma_5', 'ma_20', 'rsi_14']].to_string())
        
        print(f"\n✅ Summary:")
        print(f"   • No more hardcoded date ranges")
        print(f"   • No more hardcoded price ranges") 
        print(f"   • No more hardcoded volume averages")
        print(f"   • Using real VN30 data from 2015-2025 with {len(sample_data)} records")
        print(f"   • Technical indicators working with {len(new_cols)} new features")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_current_data()
