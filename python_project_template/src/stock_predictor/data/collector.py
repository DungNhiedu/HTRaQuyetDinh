"""
Data Collector cho thu thập dữ liệu thị trường chứng khoán
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataCollector:
    """Class để thu thập dữ liệu thị trường chứng khoán từ Yahoo Finance"""
    
    def __init__(self):
        self.cache = {}
        
    def fetch_stock_data(
        self, 
        symbol: str, 
        period: str = "2y",
        interval: str = "1d",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Thu thập dữ liệu cổ phiếu từ Yahoo Finance
        
        Args:
            symbol: Mã cổ phiếu (VD: '^VNI', 'AAPL')
            period: Thời gian lấy dữ liệu ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Khoảng thời gian ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            force_refresh: Có bắt buộc refresh cache không
            
        Returns:
            DataFrame với dữ liệu OHLCV
        """
        cache_key = f"{symbol}_{period}_{interval}"
        
        if not force_refresh and cache_key in self.cache:
            logger.info(f"Sử dụng cached data cho {symbol}")
            return self.cache[cache_key].copy()
            
        try:
            logger.info(f"Thu thập dữ liệu cho {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"Không thể thu thập dữ liệu cho {symbol}")
                
            # Làm sạch dữ liệu
            data = self._clean_data(data)
            
            # Cache dữ liệu
            self.cache[cache_key] = data.copy()
            
            logger.info(f"Thu thập thành công {len(data)} records cho {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Lỗi khi thu thập dữ liệu cho {symbol}: {str(e)}")
            raise
            
    def fetch_multiple_symbols(
        self, 
        symbols: List[str], 
        period: str = "2y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Thu thập dữ liệu cho nhiều symbols
        
        Args:
            symbols: List các mã cổ phiếu
            period: Thời gian lấy dữ liệu
            interval: Khoảng thời gian
            
        Returns:
            Dict với key là symbol và value là DataFrame
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_stock_data(symbol, period, interval)
                results[symbol] = data
            except Exception as e:
                logger.warning(f"Không thể lấy dữ liệu cho {symbol}: {str(e)}")
                
        return results
        
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Lấy thông tin cơ bản về cổ phiếu
        
        Args:
            symbol: Mã cổ phiếu
            
        Returns:
            Dict chứa thông tin cổ phiếu
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Lọc thông tin quan trọng
            important_info = {
                'symbol': symbol,
                'longName': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'marketCap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
                'country': info.get('country', 'N/A')
            }
            
            return important_info
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin cho {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}
            
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Lấy giá hiện tại của cổ phiếu
        
        Args:
            symbol: Mã cổ phiếu
            
        Returns:
            Giá hiện tại hoặc None nếu có lỗi
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy giá hiện tại cho {symbol}: {str(e)}")
            return None
            
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu thô
        
        Args:
            data: DataFrame thô từ yfinance
            
        Returns:
            DataFrame đã được làm sạch
        """
        # Loại bỏ rows có NaN
        data = data.dropna()
        
        # Đảm bảo index là datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
            
        # Sắp xếp theo thời gian
        data = data.sort_index()
        
        # Loại bỏ duplicates
        data = data[~data.index.duplicated(keep='first')]
        
        # Đảm bảo các cột số không âm (trừ returns)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = data[col].clip(lower=0)
                
        return data
        
    def save_data(self, data: pd.DataFrame, filepath: str) -> None:
        """
        Lưu dữ liệu ra file
        
        Args:
            data: DataFrame để lưu
            filepath: Đường dẫn file
        """
        try:
            if filepath.endswith('.csv'):
                data.to_csv(filepath)
            elif filepath.endswith('.parquet'):
                data.to_parquet(filepath)
            elif filepath.endswith('.xlsx'):
                data.to_excel(filepath)
            else:
                # Default to CSV
                data.to_csv(filepath + '.csv')
                
            logger.info(f"Đã lưu dữ liệu ra {filepath}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu: {str(e)}")
            raise
            
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load dữ liệu từ file
        
        Args:
            filepath: Đường dẫn file
            
        Returns:
            DataFrame
        """
        try:
            if filepath.endswith('.csv'):
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            elif filepath.endswith('.parquet'):
                data = pd.read_parquet(filepath)
            elif filepath.endswith('.xlsx'):
                data = pd.read_excel(filepath, index_col=0, parse_dates=True)
            else:
                raise ValueError("Định dạng file không được hỗ trợ")
                
            logger.info(f"Đã load dữ liệu từ {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Lỗi khi load dữ liệu từ {filepath}: {str(e)}")
            raise
