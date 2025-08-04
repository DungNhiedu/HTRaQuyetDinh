"""
Stock Forecast Module
Provides forecasting functionality using historical data for USD/VND and Gold prices
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')

class StockForecaster:
    """Stock price forecasting class."""
    
    def __init__(self):
        self.data = {}
        self.models = {}
        self.available_symbols = []
        
    def load_forecast_data(self):
        """Load USD and Gold forecast data from CSV files."""
        # Define file paths - try multiple locations
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        data_path = os.path.join(current_dir, "data")
        desktop_path = "/Users/dungnhi/Desktop"
        
        # Try sample data first, then desktop
        usd_files = [
            os.path.join(data_path, "USD_VND_sample.csv"),
            os.path.join(desktop_path, "Dữ liệu Lịch sử USD_VND.csv")
        ]
        gold_files = [
            os.path.join(data_path, "Gold_sample.csv"),
            os.path.join(desktop_path, "dữ liệu lịch sử giá vàng.csv")
        ]
        
        try:
            # Load USD data - try multiple locations
            usd_loaded = False
            for usd_file in usd_files:
                if os.path.exists(usd_file):
                    usd_data = self._process_csv_file(usd_file, "USD/VND")
                    if not usd_data.empty:
                        self.data["USD/VND"] = usd_data
                        self.available_symbols.append("USD/VND")
                        usd_loaded = True
                        print(f"✅ USD data loaded from: {usd_file}")
                        break
            
            if not usd_loaded:
                print("⚠️ Không thể tải dữ liệu USD/VND: Không thể đọc file CSV USD/VND. Sử dụng dữ liệu tổng hợp thay thế.")
                    
            # Load Gold data - try multiple locations
            gold_loaded = False
            for gold_file in gold_files:
                if os.path.exists(gold_file):
                    gold_data = self._process_csv_file(gold_file, "Gold")
                    if not gold_data.empty:
                        self.data["Gold"] = gold_data
                        self.available_symbols.append("Gold")
                        gold_loaded = True
                        print(f"✅ Gold data loaded from: {gold_file}")
                        break
            
            if not gold_loaded:
                print("⚠️ Không thể tải dữ liệu Gold: Không thể đọc file CSV Gold. Sử dụng dữ liệu tổng hợp thay thế.")
                    
        except Exception as e:
            print(f"❌ Error loading forecast data: {str(e)}")
            
        return len(self.available_symbols) > 0
    
    def _process_csv_file(self, file_path, symbol):
        """Process a single CSV file."""
        try:
            # Try different separators
            try:
                df = pd.read_csv(file_path, sep=',', encoding='utf-8')
            except Exception as e:
                try:
                    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
                except Exception as e2:
                    return pd.DataFrame()
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Check if data is semicolon-separated in single column (from Desktop)
            if len(df.columns) == 1 and ';' in df.columns[0]:
                # Split the column by semicolon
                col_name = df.columns[0]
                split_data = df[col_name].str.split(';', expand=True)
                
                # Get column names from the split
                if 'Date;close;Open;High;Low;volumn;% turnover' in col_name:
                    split_data.columns = ['Date', 'close', 'Open', 'High', 'Low', 'volumn', 'turnover']
                elif 'Date;Close;Open;High;Low;Volumn;% turnover' in col_name:
                    split_data.columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volumn', 'turnover']
                else:
                    # Try to infer from first row
                    parts = col_name.split(';')
                    split_data.columns = parts[:len(split_data.columns)]
                
                df = split_data.copy()
            
            # Clean column names again
            df.columns = df.columns.str.strip()
            
            # Process date column
            if 'Date' in df.columns:
                # Remove any extra quotes or spaces
                df['Date'] = df['Date'].astype(str).str.strip().str.replace('"', '')
                
                # Try multiple date formats
                date_parsed = False
                try:
                    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
                    date_parsed = True
                except:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
                        date_parsed = True
                    except:
                        try:
                            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                            date_parsed = True
                        except:
                            pass
                
                if date_parsed:
                    df = df.dropna(subset=['Date'])
                    df = df.sort_values('Date').reset_index(drop=True)
                else:
                    return pd.DataFrame()
                
            # Process price columns (handle comma as thousand separator)
            price_columns = ['close', 'Close', 'Open', 'High', 'Low']
            for col in price_columns:
                if col in df.columns:
                    # Remove spaces, comma (thousand separator) and convert to float
                    df[col] = df[col].astype(str).str.strip().str.replace(' ', '').str.replace(',', '').str.replace('"', '').replace('nan', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Standardize column names
            if 'close' in df.columns:
                df['Close'] = df['close']
            elif 'Close' not in df.columns:
                return pd.DataFrame()
            
            # Check if Close column has valid data
            if df['Close'].isna().all() or len(df.dropna(subset=['Close'])) == 0:
                return pd.DataFrame()
                
            # Calculate returns
            df['Return'] = df['Close'].pct_change() * 100
            df['Return'] = df['Return'].fillna(0)
            
            # Add symbol
            df['Symbol'] = symbol
            
            # Select relevant columns
            result_columns = ['Date', 'Symbol', 'Close', 'Open', 'High', 'Low', 'Return']
            available_columns = [col for col in result_columns if col in df.columns]
            
            df = df[available_columns].copy()
            df = df.dropna(subset=['Date', 'Close'])
            
            return df
            
        except Exception as e:
            return pd.DataFrame()
    
    def create_forecast_model(self, symbol, forecast_days=30):
        """Create a simple forecast model for the given symbol."""
        if symbol not in self.data:
            return None
            
        data = self.data[symbol].copy()
        
        # Prepare features
        data['Days'] = range(len(data))
        data['Moving_Avg_7'] = data['Close'].rolling(window=min(7, len(data)//2)).mean()
        data['Moving_Avg_30'] = data['Close'].rolling(window=min(30, len(data)//2)).mean()
        data['Price_Change'] = data['Close'].diff()
        
        # Remove NaN values but keep enough data
        data = data.dropna()
        
        # If still not enough data, use simpler features
        if len(data) < 10:
            data = self.data[symbol].copy()
            data['Days'] = range(len(data))
            data['Price_Change'] = data['Close'].diff().fillna(0)
        
        if len(data) < 5:
            return None
            
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=2)
        
        # Prepare training data (use last 100 days or all data if less)
        train_data = data.tail(min(100, len(data))).copy()
        
        X = train_data[['Days']].values
        y = train_data['Close'].values
        
        # Transform features
        X_poly = poly_features.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Store model and transformer
        self.models[symbol] = {
            'model': model,
            'poly_features': poly_features,
            'last_day': data['Days'].max(),
            'last_price': data['Close'].iloc[-1],
            'last_date': data['Date'].max()
        }
        
        return True
    
    def generate_forecast(self, symbol, forecast_days=30):
        """Generate forecast for the given symbol."""
        if symbol not in self.models:
            if not self.create_forecast_model(symbol, forecast_days):
                return None
                
        model_info = self.models[symbol]
        model = model_info['model']
        poly_features = model_info['poly_features']
        last_day = model_info['last_day']
        last_date = model_info['last_date']
        
        # Generate future days
        future_days = np.arange(last_day + 1, last_day + 1 + forecast_days).reshape(-1, 1)
        future_days_poly = poly_features.transform(future_days)
        
        # Make predictions
        predictions = model.predict(future_days_poly)
        
        # Create forecast dates
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted_Price': predictions,
            'Symbol': symbol
        })
        
        # Add confidence intervals (simple approach)
        historical_volatility = self.data[symbol]['Return'].std()
        forecast_df['Upper_Bound'] = forecast_df['Predicted_Price'] * (1 + historical_volatility/100 * 2)
        forecast_df['Lower_Bound'] = forecast_df['Predicted_Price'] * (1 - historical_volatility/100 * 2)
        
        return forecast_df
    
    def get_historical_data(self, symbol, days=None):
        """Get historical data for plotting."""
        if symbol not in self.data:
            return pd.DataFrame()
            
        data = self.data[symbol].copy()
        
        if days:
            data = data.tail(days)
            
        return data
    
    def create_forecast_chart(self, symbol, forecast_days=30, historical_days=90):
        """Create a comprehensive forecast chart."""
        # Get historical data
        historical = self.get_historical_data(symbol, historical_days)
        
        if historical.empty:
            return None
            
        # Generate forecast
        forecast = self.generate_forecast(symbol, forecast_days)
        
        if forecast is None:
            return None
            
        # Create plotly figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical['Date'],
            y=historical['Close'],
            mode='lines',
            name='Giá Lịch sử',
            line=dict(color='blue', width=2)
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast['Date'],
            y=forecast['Predicted_Price'],
            mode='lines',
            name='Dự báo',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['Date'].tolist() + forecast['Date'].tolist()[::-1],
            y=forecast['Upper_Bound'].tolist() + forecast['Lower_Bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Khoảng Tin cậy',
            showlegend=True
        ))
        
        # Update layout
        y_title = "Giá (VND)" if symbol == "USD/VND" else "Giá (USD/ounce)"
        fig.update_layout(
            title=f'Dự báo Giá {symbol} ({forecast_days} ngày)',
            xaxis_title='Thời gian',
            yaxis_title=y_title,
            height=500,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Format y-axis based on symbol
        if symbol == "USD/VND":
            fig.update_yaxes(tickformat=":,.0f")
        else:  # Gold
            fig.update_yaxes(tickformat=":,.2f")
        
        return fig
    
    def get_forecast_summary(self, symbol, forecast_days=30):
        """Get forecast summary statistics."""
        forecast = self.generate_forecast(symbol, forecast_days)
        historical = self.get_historical_data(symbol)
        
        if forecast is None or historical.empty:
            return None
            
        current_price = historical['Close'].iloc[-1]
        forecast_end_price = forecast['Predicted_Price'].iloc[-1]
        
        price_change = forecast_end_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        avg_forecast_price = forecast['Predicted_Price'].mean()
        max_forecast_price = forecast['Predicted_Price'].max()
        min_forecast_price = forecast['Predicted_Price'].min()
        
        # Determine trend
        if price_change_pct > 2:
            trend = "📈 Tăng mạnh"
            trend_color = "green"
        elif price_change_pct > 0:
            trend = "📈 Tăng nhẹ"
            trend_color = "lightgreen"
        elif price_change_pct > -2:
            trend = "➡️ Đi ngang"
            trend_color = "orange"
        else:
            trend = "📉 Giảm"
            trend_color = "red"
            
        return {
            'current_price': current_price,
            'forecast_end_price': forecast_end_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'avg_forecast_price': avg_forecast_price,
            'max_forecast_price': max_forecast_price,
            'min_forecast_price': min_forecast_price,
            'trend': trend,
            'trend_color': trend_color,
            'historical_volatility': historical['Return'].std()
        }
