#!/usr/bin/env python3
"""
Simple test for forecast functionality
"""

import pandas as pd
import os

def test_csv_reading():
    """Test reading CSV files directly."""
    print("=== Testing CSV Reading ===")
    
    # Test USD file
    usd_file = "/Users/dungnhi/Desktop/Dữ liệu Lịch sử USD_VND.csv"
    if os.path.exists(usd_file):
        try:
            print(f"Reading {usd_file}...")
            df = pd.read_csv(usd_file, sep=';', encoding='utf-8')
            print(f"✅ Successfully read USD file")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print("   First few rows:")
            print(df.head(3))
            
            # Test processing
            df.columns = df.columns.str.strip()
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
                print(f"   Date conversion: {df['Date'].notna().sum()} valid dates")
            
            if 'close' in df.columns:
                df['close'] = df['close'].astype(str).str.replace(',', '.')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                print(f"   Price conversion: {df['close'].notna().sum()} valid prices")
                print(f"   Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
            
        except Exception as e:
            print(f"❌ Error reading USD file: {e}")
    
    # Test Gold file
    gold_file = "/Users/dungnhi/Desktop/dữ liệu lịch sử giá vàng.csv"
    if os.path.exists(gold_file):
        try:
            print(f"\nReading {gold_file}...")
            df = pd.read_csv(gold_file, sep=';', encoding='utf-8')
            print(f"✅ Successfully read Gold file")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print("   First few rows:")
            print(df.head(3))
            
        except Exception as e:
            print(f"❌ Error reading Gold file: {e}")

if __name__ == "__main__":
    test_csv_reading()
