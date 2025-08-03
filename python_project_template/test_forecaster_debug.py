#!/usr/bin/env python3
"""
Debug test for forecaster functionality
"""

import sys
import os
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_forecaster_debug():
    """Debug forecaster loading."""
    print("=== Debug Forecaster Loading ===")
    
    # Test direct CSV processing similar to forecaster
    desktop_path = "/Users/dungnhi/Desktop"
    usd_file = os.path.join(desktop_path, "Dữ liệu Lịch sử USD_VND.csv")
    gold_file = os.path.join(desktop_path, "dữ liệu lịch sử giá vàng.csv")
    
    def process_csv_file(file_path, symbol):
        """Process a single CSV file."""
        try:
            print(f"\nProcessing {symbol} from {file_path}")
            
            # Read CSV with semicolon separator
            df = pd.read_csv(file_path, sep=';', encoding='utf-8')
            print(f"  ✅ Read CSV: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            print(f"  Cleaned columns: {list(df.columns)}")
            
            # Process date column
            if 'Date' in df.columns:
                print(f"  Processing Date column...")
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
                df = df.dropna(subset=['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                print(f"  ✅ Date processed: {len(df)} valid dates")
                
            # Process price columns (handle comma decimal separator)
            price_columns = ['close', 'Close', 'Open', 'High', 'Low']
            for col in price_columns:
                if col in df.columns:
                    print(f"  Processing {col} column...")
                    print(f"    Original values: {df[col].head(2).tolist()}")
                    # Remove comma (thousand separator) and convert to float
                    df[col] = df[col].astype(str).str.replace(',', '').replace('nan', '')
                    print(f"    After cleaning: {df[col].head(2).tolist()}")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"    After numeric conversion: {df[col].head(2).tolist()}")
                    valid_prices = df[col].notna().sum()
                    print(f"    Valid prices: {valid_prices}/{len(df)}")
                    if valid_prices > 0:
                        print(f"    Price range: {df[col].min():.2f} - {df[col].max():.2f}")
            
            # Standardize column names
            if 'close' in df.columns:
                df['Close'] = df['close']
                print(f"  ✅ Standardized close -> Close")
            elif 'Close' not in df.columns:
                print(f"  ❌ No Close/close column found")
                return pd.DataFrame()
                
            # Calculate returns
            df['Return'] = df['Close'].pct_change() * 100
            df['Return'] = df['Return'].fillna(0)
            print(f"  ✅ Returns calculated")
            
            # Add symbol
            df['Symbol'] = symbol
            
            # Select relevant columns
            result_columns = ['Date', 'Symbol', 'Close', 'Open', 'High', 'Low', 'Return']
            available_columns = [col for col in result_columns if col in df.columns]
            print(f"  Available result columns: {available_columns}")
            
            df = df[available_columns].copy()
            df = df.dropna(subset=['Date', 'Close'])
            
            print(f"  ✅ Final result: {df.shape}")
            if len(df) > 0:
                print(f"  Sample data:")
                print(df.head(2))
            
            return df
            
        except Exception as e:
            print(f"  ❌ Error processing {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    # Test both files
    available_symbols = []
    
    if os.path.exists(usd_file):
        usd_data = process_csv_file(usd_file, "USD/VND")
        if not usd_data.empty:
            available_symbols.append("USD/VND")
            
    if os.path.exists(gold_file):
        gold_data = process_csv_file(gold_file, "Gold")
        if not gold_data.empty:
            available_symbols.append("Gold")
    
    print(f"\n=== Summary ===")
    print(f"Available symbols: {available_symbols}")
    print(f"Data loading success: {len(available_symbols) > 0}")

if __name__ == "__main__":
    test_forecaster_debug()
