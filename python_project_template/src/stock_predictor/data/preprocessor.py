"""
Data preprocessing module for stock market prediction.
Based on the reference implementation from copy_of_đồ_án_dss_nhóm_1.py
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing class for stock market data."""
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        self.scalers = {}
        
    def load_and_process_all(self, input_dir: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and process all CSV files from input directory.
        Supports multiple CSV formats including semicolon-separated files.
        
        Args:
            input_dir: Directory containing CSV files
            output_path: Optional path to save merged data
            
        Returns:
            Merged and processed DataFrame
        """
        all_data = []
        
        for filename in os.listdir(input_dir):
            if filename.endswith(".csv"):
                filepath = os.path.join(input_dir, filename)
                
                logger.info(f"Processing file: {filename}")
                
                # Try different separators and encodings
                df = self._read_csv_flexible(filepath)
                
                if df is None or df.empty:
                    logger.warning(f"Could not read or file is empty: {filename}")
                    continue

                # Extract stock code from filename (before first underscore or dot)
                stock_code = filename.split("_")[0].split(".")[0].upper()
                logger.info(f"Extracted stock code: {stock_code} from filename: {filename}")
                
                # Normalize data format
                normalized_df = self._normalize_data_format(df, stock_code)
                
                if normalized_df is None:
                    logger.error(f"Failed to normalize data format for {filename} (stock: {stock_code})")
                    continue
                elif normalized_df.empty:
                    logger.warning(f"Normalized data is empty for {filename} (stock: {stock_code})")
                    continue
                else:
                    logger.info(f"Successfully normalized {filename} (stock: {stock_code})")
                    all_data.append(normalized_df)

        if not all_data:
            logger.error("No valid data found in any files")
            return pd.DataFrame()

        # Gộp tất cả vào một DataFrame
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # Calculate returns and targets after all processing
        merged_df = self._calculate_returns_and_targets(merged_df)

        # Lưu nếu có đường dẫn
        if output_path:
            merged_df.to_csv(output_path, index=False)

        return merged_df
        
    def normalize_data(self, data: pd.DataFrame, 
                      columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize the data using the specified scaler.
        
        Args:
            data: Input DataFrame
            columns: Columns to normalize
            
        Returns:
            Normalized DataFrame
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
            
        result = data.copy()
        
        for col in columns:
            if col not in self.scalers:
                if self.scaler_type == 'standard':
                    self.scalers[col] = StandardScaler()
                else:
                    self.scalers[col] = MinMaxScaler()
                    
                result[col] = self.scalers[col].fit_transform(
                    result[col].values.reshape(-1, 1)
                ).flatten()
            else:
                result[col] = self.scalers[col].transform(
                    result[col].values.reshape(-1, 1)
                ).flatten()
                
        return result
    
    def create_sequences(self, data: pd.DataFrame, 
                        sequence_length: int = 60,
                        target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: Input DataFrame
            sequence_length: Length of input sequences
            target_column: Target column name
            
        Returns:
            X, y arrays for training
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data.iloc[i-sequence_length:i].values)
            y.append(data[target_column].iloc[i])
            
        return np.array(X), np.array(y)
    
    def split_data(self, data: pd.DataFrame, 
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Input DataFrame
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            
        Returns:
            Dictionary with train, val, test DataFrames
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return {
            'train': data.iloc[:train_end],
            'val': data.iloc[train_end:val_end],
            'test': data.iloc[val_end:]
        }
    
    def handle_missing_values(self, data: pd.DataFrame, 
                            method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            data: Input DataFrame
            method: Method to handle missing values
            
        Returns:
            DataFrame with handled missing values
        """
        result = data.copy()
        
        if method == 'forward_fill':
            result = result.fillna(method='ffill')
        elif method == 'backward_fill':
            result = result.fillna(method='bfill')
        elif method == 'interpolate':
            result = result.interpolate()
        elif method == 'drop':
            result = result.dropna()
            
        return result
    
    def add_lag_features(self, data: pd.DataFrame, 
                        columns: List[str],
                        lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Add lag features to the data.
        
        Args:
            data: Input DataFrame
            columns: Columns to create lag features for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        result = data.copy()
        
        for col in columns:
            for lag in lags:
                result[f'{col}_lag_{lag}'] = result[col].shift(lag)
                
        return result
    
    def add_rolling_features(self, data: pd.DataFrame,
                           columns: List[str],
                           windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Add rolling statistics features.
        
        Args:
            data: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        result = data.copy()
        
        for col in columns:
            for window in windows:
                result[f'{col}_rolling_mean_{window}'] = result[col].rolling(window).mean()
                result[f'{col}_rolling_std_{window}'] = result[col].rolling(window).std()
                result[f'{col}_rolling_min_{window}'] = result[col].rolling(window).min()
                result[f'{col}_rolling_max_{window}'] = result[col].rolling(window).max()
                
        return result
    
    def _read_csv_flexible(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Try to read CSV file with different separators and encodings.
        Handles files with header rows and comma decimal separators.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame or None if reading fails
        """
        logger.info(f"Attempting to read CSV file: {filepath}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            logger.error(f"File does not exist: {filepath}")
            return None
        
        # Check file size
        try:
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                logger.error(f"File is empty: {filepath}")
                return None
            logger.info(f"File size: {file_size} bytes")
        except Exception as e:
            logger.error(f"Cannot access file: {filepath}, error: {e}")
            return None
        
        # Common separators and encodings to try
        separators = [';', ',', '\t']
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        
        successful_read = False
        
        for sep in separators:
            for encoding in encodings:
                for skip_rows in [0, 1, 2]:  # Try skipping header rows
                    for decimal in [',', '.']:  # Try different decimal separators
                        try:
                            thousands = '.' if decimal == ',' else ','
                            
                            df = pd.read_csv(
                                filepath, 
                                sep=sep, 
                                encoding=encoding, 
                                skiprows=skip_rows,
                                decimal=decimal,
                                thousands=thousands if decimal != thousands else None
                            )
                            
                            # Check if we have valid data
                            if len(df.columns) > 1 and not df.empty and len(df) > 0:
                                # Check if we have reasonable column names (not just numbers)
                                has_text_columns = any(isinstance(col, str) and any(c.isalpha() for c in str(col)) for col in df.columns)
                                if has_text_columns:
                                    logger.info(f"Successfully read {filepath} with separator='{sep}', encoding='{encoding}', skiprows={skip_rows}, decimal='{decimal}'")
                                    logger.info(f"Columns found: {list(df.columns)}")
                                    logger.info(f"Shape: {df.shape}")
                                    successful_read = True
                                    return df
                        except Exception as e:
                            logger.debug(f"Failed with sep='{sep}', encoding='{encoding}', skiprows={skip_rows}, decimal='{decimal}': {e}")
                            continue
        
        if not successful_read:
            logger.error(f"Could not read {filepath} with any separator/encoding combination")
            
            # Try to provide helpful error information
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    first_few_lines = [f.readline().strip() for _ in range(5)]
                    logger.info(f"First few lines of the file: {first_few_lines}")
            except Exception as e:
                logger.error(f"Cannot even read file as text: {e}")
        
        return None
    
    def _normalize_data_format(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        Normalize different data formats to a standard format.
        
        Args:
            df: Input DataFrame
            stock_code: Stock code extracted from filename
            
        Returns:
            Normalized DataFrame
        """
        # Chuẩn hóa tên cột
        df.columns = [col.strip().lower().replace('%', 'pct').replace(' ', '_') for col in df.columns]
        
        # Mapping different column names to standard names
        column_mapping = {
            'date': 'date',
            'ngay': 'date',
            'ngày': 'date',  # Handle Vietnamese with accent
            'open': 'open',
            'mo_cua': 'open',
            'mở': 'open',   # Vietnamese
            'high': 'high',
            'cao_nhat': 'high',
            'cao': 'high',  # Vietnamese
            'low': 'low',
            'thap_nhat': 'low',
            'thấp': 'low',  # Vietnamese
            'close': 'close',
            'dong_cua': 'close',
            'lan_cuoi': 'close',
            'lần_cuối': 'close',  # Vietnamese with accent
            'volume': 'volume',
            'volumn': 'volume',  # Handle typo in original data
            'kl': 'volume',  # Vietnamese abbreviation
            'khoi_luong': 'volume',
            'turnover': 'turnover',
            'gia_tri': 'turnover',
            'pct_turnover': 'pct_change',
            'pct_thay_đổi': 'pct_change'  # Vietnamese percentage change
        }
        
        # Rename columns based on mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Ensure we have required columns
        required_cols = ['date', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns in {stock_code}: {missing_cols}")
            logger.info(f"Available columns in {stock_code}: {list(df.columns)}")
            # Try to infer missing columns
            if 'close' in df.columns and 'open' not in df.columns:
                df['open'] = df['close'].shift(1).fillna(df['close'])
                logger.info(f"Inferred 'open' column for {stock_code}")
            if 'close' in df.columns and 'high' not in df.columns:
                df['high'] = df['close']
                logger.info(f"Inferred 'high' column for {stock_code}")
            if 'close' in df.columns and 'low' not in df.columns:
                df['low'] = df['close']
                logger.info(f"Inferred 'low' column for {stock_code}")
                
            # After inference, check again
            final_missing = [col for col in required_cols if col not in df.columns]
            if final_missing:
                logger.error(f"Still missing critical columns in {stock_code} after inference: {final_missing}")
                return None  # Return None if critical columns are still missing
        
        # Handle date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
            
            # Extract year, month, day
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
        
        # Add stock code
        df['code'] = stock_code
        
        # Handle missing volume and turnover
        if 'volume' not in df.columns:
            df['volume'] = 1000000  # Default volume
        
        # Convert numeric columns to proper numeric types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                # Handle different number formats
                if df[col].dtype == 'object':
                    def convert_number_flexible(value):
                        if pd.isna(value) or str(value).strip() in ['', 'nan']:
                            return None
                        try:
                            str_val = str(value).strip()
                            
                            # Remove 'M' suffix (millions) and handle it
                            multiplier = 1
                            if str_val.endswith('M'):
                                str_val = str_val[:-1]
                                multiplier = 1000000
                            elif str_val.endswith('K'):
                                str_val = str_val[:-1]
                                multiplier = 1000
                            
                            # Check if it's European format (comma as decimal separator)
                            if ',' in str_val and '.' not in str_val:
                                # European format: 1,234.56 -> replace comma with dot
                                str_val = str_val.replace(',', '.')
                            elif ',' in str_val and '.' in str_val:
                                # Mixed format: check which is thousands vs decimal
                                comma_pos = str_val.rfind(',')
                                dot_pos = str_val.rfind('.')
                                if comma_pos > dot_pos:
                                    # Comma is decimal separator: 1.234,56
                                    str_val = str_val.replace('.', '').replace(',', '.')
                                else:
                                    # Dot is decimal separator: 1,234.56
                                    str_val = str_val.replace(',', '')
                            
                            return float(str_val) * multiplier
                        except:
                            return None
                    
                    df[col] = df[col].apply(convert_number_flexible)
                else:
                    # Already numeric, just ensure it's float
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'turnover' not in df.columns and 'volume' in df.columns and 'close' in df.columns:
            # Make sure both columns are numeric before multiplication
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(1000000)
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['turnover'] = df['volume'] * df['close']
        
        # Remove rows with invalid data
        df = df.dropna(subset=['date', 'close'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def _calculate_returns_and_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns and targets for each stock.
        
        Args:
            df: Input DataFrame with stock data
            
        Returns:
            DataFrame with returns and targets
        """
        result_data = []
        
        # Group by stock code to calculate returns separately
        for code in df['code'].unique():
            stock_data = df[df['code'] == code].copy()
            stock_data = stock_data.sort_values('date').reset_index(drop=True)
            
            # Calculate daily returns (percentage change)
            stock_data['return'] = stock_data['close'].pct_change() * 100
            stock_data['return'] = stock_data['return'].round(2).fillna(0)
            
            # Create binary target (1 if return > 0, 0 otherwise)
            stock_data['target'] = (stock_data['return'] > 0).astype(int)
            
            result_data.append(stock_data)
        
        # Combine all stock data
        final_df = pd.concat(result_data, ignore_index=True)
        
        return final_df
