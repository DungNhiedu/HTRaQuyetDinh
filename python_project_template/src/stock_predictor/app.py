"""
Streamlit web application for Stock Market Prediction Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import shutil
import google.generativeai as genai
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessor import DataPreprocessor
from data.features import add_technical_indicators, FeatureEngineer
# from models.traditional import TraditionalModels  # Commented out for demo

st.set_page_config(
    page_title="Stock Market Prediction Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def create_sample_data(n_days=3650):  # 10 years of data
    """Create sample stock data for demonstration (2015-2025)."""
    # Use actual date range from 2015 to 2025
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2025, 7, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Use actual number of days in the range
    n_days = len(dates)
    
    # Generate realistic USD/VND-like data (starting around 22,000)
    np.random.seed(42)
    base_price = 22000
    price_changes = np.random.randn(n_days) * 50  # Daily volatility
    close_prices = base_price + np.cumsum(price_changes)
    
    # Ensure prices stay in realistic range
    close_prices = np.clip(close_prices, 20000, 27000)
    
    data = []
    for i, date in enumerate(dates):
        open_price = close_prices[i] + np.random.randn() * 20
        high_price = max(open_price, close_prices[i]) + abs(np.random.randn()) * 30
        low_price = min(open_price, close_prices[i]) - abs(np.random.randn()) * 30
        volume = int(1000000 + np.random.randn() * 100000)
        turnover = volume * close_prices[i]
        
        data.append({
            'code': 'USD_VND',
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

def plot_price_chart(data):
    """Create price chart with technical indicators using proper date axis."""
    fig = go.Figure()
    
    # Prepare x-axis (dates)
    if 'date' in data.columns:
        x_axis = pd.to_datetime(data['date'])
    else:
        # Create date range from 2015 to 2025 if no date column
        start_date = datetime(2015, 1, 1)
        x_axis = pd.date_range(start=start_date, periods=len(data), freq='D')
    
    # Price line
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=data['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add moving averages - check for different possible column names
    ma_columns = [col for col in data.columns if 'ma_' in col.lower() or 'sma_' in col.lower()]
    
    # Specifically look for MA 5 and MA 20
    ma5_col = None
    ma20_col = None
    
    for col in ma_columns:
        if '5' in col:
            ma5_col = col
        elif '20' in col:
            ma20_col = col
    
    if ma5_col and ma5_col in data.columns and not data[ma5_col].isna().all():
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=data[ma5_col],
            mode='lines',
            name='MA 5',
            line=dict(color='orange', width=1.5)
        ))
    
    if ma20_col and ma20_col in data.columns and not data[ma20_col].isna().all():
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=data[ma20_col],
            mode='lines',
            name='MA 20',
            line=dict(color='red', width=1.5)
        ))
    
    # Add Bollinger Bands if available
    bb_columns = [col for col in data.columns if 'bb_' in col.lower()]
    bb_upper = None
    bb_lower = None
    bb_middle = None
    
    for col in bb_columns:
        if 'upper' in col.lower() or 'bbh' in col.lower():
            bb_upper = col
        elif 'lower' in col.lower() or 'bbl' in col.lower():
            bb_lower = col
        elif 'middle' in col.lower() or 'bbm' in col.lower():
            bb_middle = col
    
    if bb_upper and bb_lower and bb_upper in data.columns and bb_lower in data.columns:
        if not data[bb_upper].isna().all() and not data[bb_lower].isna().all():
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=data[bb_upper],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=data[bb_lower],
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ))
    
    fig.update_layout(
        title='VN30 Index with Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        height=500,
        showlegend=True
    )
    
    return fig

def plot_technical_indicators(data):
    """Plot technical indicators with proper date axis."""
    fig = go.Figure()
    
    # Prepare x-axis (dates)
    if 'date' in data.columns:
        x_axis = pd.to_datetime(data['date'])
    else:
        # Create date range from 2015 to 2025 if no date column
        start_date = datetime(2015, 1, 1)
        x_axis = pd.date_range(start=start_date, periods=len(data), freq='D')
    
    # Look for RSI columns
    rsi_columns = [col for col in data.columns if 'rsi' in col.lower()]
    
    if rsi_columns:
        rsi_col = rsi_columns[0]  # Use the first RSI column found
        if not data[rsi_col].isna().all():
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=data[rsi_col],
                mode='lines',
                name=f'RSI ({rsi_col})',
                line=dict(color='purple', width=2)
            ))
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            
            fig.update_layout(
                title='RSI Technical Indicator',
                xaxis_title='Date',
                yaxis_title='RSI Value',
                height=300,
                yaxis=dict(range=[0, 100])
            )
        else:
            # No valid RSI data
            fig.add_annotation(
                text="No RSI data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title='RSI Technical Indicator (No Data)',
                xaxis_title='Date',
                yaxis_title='RSI Value',
                height=300
            )
    else:
        # No RSI columns found
        fig.add_annotation(
            text="RSI indicator not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title='RSI Technical Indicator (Not Available)',
            xaxis_title='Date',
            yaxis_title='RSI Value',
            height=300
        )
    
    return fig

def generate_future_prediction(data_summary):
    """Generate a future prediction based on historical data trends."""
    current_price = data_summary['current_price']
    avg_volatility = data_summary['avg_volatility']
    up_days_ratio = data_summary['up_days_ratio']
    
    # Simple trend analysis
    if up_days_ratio > 52:
        trend = "tăng trưởng"
        direction = "📈"
        growth_estimate = "2-5%"
    elif up_days_ratio < 48:
        trend = "giảm"
        direction = "📉"
        growth_estimate = "-2 đến -5%"
    else:
        trend = "ổn định"
        direction = "➡️"
        growth_estimate = "-1 đến +1%"
    
    # Generate 10-year projection
    projection_text = f"""
    **Dự báo mở rộng cho 10 năm tiếp theo (2025-2035):**
    
    {direction} **Xu hướng dài hạn:** {trend}
    
    📊 **Ước tính tăng trưởng hàng năm:** {growth_estimate}
    
    🎯 **Kịch bản dự kiến:**
    - **Năm 1-3 (2025-2027):** Tiếp tục xu hướng hiện tại với biến động {avg_volatility:.1f}%
    - **Năm 4-7 (2028-2031):** Điều chỉnh theo chu kỳ kinh tế
    - **Năm 8-10 (2032-2035):** Ổn định ở mức giá mới
    
    💡 **Lưu ý:** Dự báo dài hạn có độ không chắc chắn cao do nhiều yếu tố không dự đoán được.
    """
    
    return projection_text

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
    """Get AI-based market prediction using Gemini Pro."""
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Use the latest Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create prompt for stock market analysis
        prompt = f"""
        Phân tích dữ liệu thị trường chứng khoán sau và đưa ra dự báo:

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

def format_gemini_response(response_text):
    """Format Gemini response for better display."""
    # Split into sections
    sections = response_text.split('\n\n')
    
    formatted_sections = []
    for section in sections:
        if section.strip():
            # Check if it's a header (starts with number)
            if section.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                formatted_sections.append(f"**{section.strip()}**")
            else:
                formatted_sections.append(section.strip())
    
    return '\n\n'.join(formatted_sections)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">📈 Market Prediction Demo</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;">Dự báo chỉ số thị trường sử dụng Machine Learning</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # AI Prediction Button
    st.sidebar.markdown("### 🤖 AI Prediction")
    use_ai_prediction = st.sidebar.button(
        "🧠 Get AI Market Prediction",
        type="primary",
        help="Click to get AI-powered market analysis and predictions",
        use_container_width=True
    )
    
    # Demo options
    demo_option = st.sidebar.selectbox(
        "Choose Demo Type",
        ["Sample Data Demo", "Upload CSV Files"]
    )
    
    if demo_option == "Sample Data Demo":
        st.markdown('<div class="section-header">📊 Sample Data Analysis</div>', unsafe_allow_html=True)
        
        # Load real VN30 data instead of synthetic data
        vn30_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/VN30_demo.csv"
        
        try:
            with st.spinner("Loading VN30 data..."):
                # Load and process VN30 data using DataPreprocessor
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
                
                # Use this as our sample data
                sample_data = vn30_data.copy()
                
            st.success(f"✅ Loaded VN30 data! Shape: {sample_data.shape}")
            
            # Show actual VN30 data analysis
            with st.expander("📊 VN30 Data Analysis", expanded=True):
                st.write("**Data Source:** Vietnam VN30 Index Historical Data")
                
                # Show raw data sample
                st.write("**Raw VN30 Data (First 10 rows):**")
                display_data = sample_data.head(10).copy()
                if 'date' in display_data.columns:
                    display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
                st.dataframe(display_data, use_container_width=True)
            
            st.info("✅ Using real VN30 index data for analysis and prediction")
            
        except Exception as e:
            st.warning(f"Could not load VN30 data: {str(e)}. Using synthetic data instead.")
            # Fallback to synthetic data
            with st.spinner("Generating sample data..."):
                sample_data = create_sample_data()
        
        # Calculate time duration
        time_duration = calculate_time_duration(sample_data)
        
        # Display basic info
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Time Duration", time_duration.split(' (')[0])  # Just the years part
        with col2:
            st.metric("Total Days", len(sample_data))
        with col3:
            st.metric("Latest Price", f"{sample_data['close'].iloc[-1]:.2f}")
        with col4:
            if 'return' in sample_data.columns:
                st.metric("Price Change %", f"{sample_data['return'].iloc[-1]:.2f}%")
            else:
                st.metric("Price Change %", "N/A")
        with col5:
            if 'target' in sample_data.columns:
                up_days = (sample_data['target'] == 1).sum()
                st.metric("Up Days %", f"{100 * up_days / len(sample_data):.1f}%")
            else:
                st.metric("Up Days %", "N/A")
        
        # Add technical indicators
        st.markdown('<div class="section-header">🔧 Feature Engineering</div>', unsafe_allow_html=True)
        
        with st.spinner("Adding technical indicators..."):
            data_with_features = add_technical_indicators(sample_data)
        
        st.success(f"✅ Added technical indicators! Shape: {data_with_features.shape}")
        
        # Debug: Show what indicators were actually added
        with st.expander("🔍 Debug: Technical Indicators Status"):
            st.write("**Available columns after adding technical indicators:**")
            all_cols = list(data_with_features.columns)
            original_cols = list(sample_data.columns)
            new_cols = [col for col in all_cols if col not in original_cols]
            
            st.write(f"Original columns ({len(original_cols)}): {original_cols}")
            st.write(f"New technical indicators ({len(new_cols)}): {new_cols}")
            
            # Check for specific MA indicators
            ma_indicators = [col for col in new_cols if 'ma_' in col.lower()]
            rsi_indicators = [col for col in new_cols if 'rsi' in col.lower()]
            bb_indicators = [col for col in new_cols if 'bb_' in col.lower()]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**MA Indicators:** {ma_indicators}")
            with col2:
                st.write(f"**RSI Indicators:** {rsi_indicators}")
            with col3:
                st.write(f"**Bollinger Bands:** {bb_indicators}")
            
            # Show sample of technical indicator values
            if len(new_cols) > 0:
                st.write("**Sample values (last 5 rows):**")
                sample_tech = data_with_features[new_cols].tail()
                st.dataframe(sample_tech, use_container_width=True)
        
        # Show feature columns
        with st.expander("View Technical Indicators"):
            new_features = set(data_with_features.columns) - set(sample_data.columns)
            st.write("**New technical indicators added:**")
            for feature in sorted(new_features):
                st.write(f"• {feature}")
        
        # Visualizations
        st.markdown('<div class="section-header">📈 Charts & Analysis</div>', unsafe_allow_html=True)
        
        # Price chart
        st.write("**VN30 Index Price Chart with Technical Indicators:**")
        price_fig = plot_price_chart(data_with_features)
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Show chart info
        with st.expander("📊 Chart Information"):
            available_indicators = []
            ma_cols = [col for col in data_with_features.columns if 'ma_' in col.lower() or 'sma_' in col.lower()]
            bb_cols = [col for col in data_with_features.columns if 'bb_' in col.lower()]
            rsi_cols = [col for col in data_with_features.columns if 'rsi' in col.lower()]
            
            if ma_cols:
                available_indicators.append(f"Moving Averages: {ma_cols}")
            if bb_cols:
                available_indicators.append(f"Bollinger Bands: {bb_cols}")
            if rsi_cols:
                available_indicators.append(f"RSI: {rsi_cols}")
            
            if available_indicators:
                st.write("**Technical indicators displayed:**")
                for indicator in available_indicators:
                    st.write(f"• {indicator}")
            else:
                st.warning("No technical indicators found in the data")
        
        # Technical indicators chart
        st.write("**RSI Technical Indicator:**")
        rsi_fig = plot_technical_indicators(data_with_features)
        st.plotly_chart(rsi_fig, use_container_width=True)
        
        # Feature engineering demo
        st.markdown('<div class="section-header">🧠 Advanced Feature Engineering</div>', unsafe_allow_html=True)
        
        feature_engineer = FeatureEngineer()
        
        with st.spinner("Creating additional features..."):
            # Add more features using the real VN30 data
            enriched_data = feature_engineer.create_price_features(data_with_features)
            enriched_data = feature_engineer.create_volume_features(enriched_data)
            enriched_data = feature_engineer.create_lag_features(enriched_data)
            enriched_data = feature_engineer.create_rolling_features(enriched_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Features", len(sample_data.columns))
        with col2:
            st.metric("Final Features", len(enriched_data.columns))
        
        # Show data sample
        st.markdown('<div class="section-header">📋 Data Preview</div>', unsafe_allow_html=True)
        
        st.write("**Final dataset preview (Real VN30 Data):**")
        st.dataframe(enriched_data.head(), use_container_width=True)
        
        # Additional info about the VN30 data
        with st.expander("ℹ️ About VN30 Data"):
            st.write("**Data Source:** Vietnam VN30 Index Historical Data")
            st.write("**Records:**", len(sample_data))
            st.write("**Features after enrichment:**", len(enriched_data.columns))
        
        # Model demonstration
        st.markdown('<div class="section-header">🤖 Model Training Demo</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Run Model Training Demo", type="primary"):
            with st.spinner("Preparing data for modeling..."):
                # Prepare data for modeling
                # Show data info before cleaning
                st.info(f"📊 Data before cleaning: {enriched_data.shape}")
                st.info(f"📊 NaN values: {enriched_data.isnull().sum().sum()}")
                
                # Remove rows with NaN values, but be more flexible
                clean_data = enriched_data.dropna()
                
                st.info(f"📊 Data after cleaning: {clean_data.shape}")
                
                # Lower the threshold for minimum data points
                min_required_samples = min(20, len(enriched_data) // 4)  # More flexible threshold
                
                if len(clean_data) >= min_required_samples:  # Ensure we have enough data
                    # Split features and target
                    feature_cols = [col for col in clean_data.columns if col not in ['code', 'target', 'date', 'year', 'month', 'day']]
                    
                    # Select only numeric columns for modeling
                    numeric_features = []
                    for col in feature_cols:
                        if pd.api.types.is_numeric_dtype(clean_data[col]):
                            numeric_features.append(col)
                    
                    X = clean_data[numeric_features]
                    y = clean_data['target']
                    
                    st.info(f"📊 Selected {len(numeric_features)} numeric features for modeling")
                    
                    # Simple train/test split
                    split_idx = max(1, int(0.8 * len(clean_data)))  # Ensure at least 1 sample for test
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    st.success(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
                    
                    # Show feature importance simulation
                    np.random.seed(42)
                    feature_importance = np.random.rand(min(10, len(numeric_features)))
                    top_features = np.array(numeric_features[:len(feature_importance)])
                    
                    importance_df = pd.DataFrame({
                        'Feature': top_features,
                        'Importance': feature_importance
                    }).sort_values('Importance', ascending=True)
                    
                    fig_importance = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Simulated Feature Importance'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Simulated model results
                    st.write("**Simulated Model Performance:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", "75.2%")
                    with col2:
                        st.metric("Precision", "73.8%")
                    with col3:
                        st.metric("Recall", "76.5%")
                    
                else:
                    st.error(f"Not enough clean data for modeling. Need at least {min_required_samples} samples, but only have {len(clean_data)} after cleaning.")
                    st.error("Try uploading more data or check data quality.")
        
        # AI Prediction Section
        st.markdown('<div class="section-header">🤖 AI-Based Market Prediction</div>', unsafe_allow_html=True)
        
        if use_ai_prediction:
            with st.spinner("Đang phân tích dữ liệu với AI..."):
                # Prepare data summary for AI analysis
                data_summary = {
                    'total_days': len(enriched_data),
                    'time_duration': time_duration,
                    'current_price': sample_data['close'].iloc[-1],
                    'latest_change': sample_data['return'].iloc[-1],
                    'up_days_ratio': 100 * (sample_data['target'] == 1).sum() / len(sample_data),
                    'highest_price': sample_data['close'].max(),
                    'lowest_price': sample_data['close'].min(),
                    'avg_volatility': abs(sample_data['return']).mean()
                }
                api_key = "AIzaSyDMs-iLWgB7NuoCtJLqEj4SwG3qhM3B-gQ"
                ai_prediction = get_gemini_prediction(data_summary, api_key)
                st.markdown("---")
                st.markdown("### 🔮 AI Market Analysis & Prediction")
                if "Lỗi" not in ai_prediction:
                    with st.expander("📊 Xem dự báo AI cho 10 năm tiếp theo", expanded=True):
                        st.success("✅ AI đã phân tích xong!")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Số ngày phân tích", f"{data_summary['total_days']:,}")
                        with col2:
                            st.metric("Giá hiện tại", f"{data_summary['current_price']:.2f}")
                        with col3:
                            st.metric("Thay đổi gần nhất", f"{data_summary['latest_change']:.2f}%")
                        with col4:
                            st.metric("Tỷ lệ ngày tăng", f"{data_summary['up_days_ratio']:.1f}%")
                        st.markdown("---")
                        st.markdown("#### � Dự báo chi tiết:")
                        formatted_response = format_gemini_response(ai_prediction)
                        st.markdown(
                            f"""
                            <div style='background-color: #000000; color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 10px 0;'>
                                {formatted_response}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.markdown("#### 🔮 Dự báo mở rộng (10 năm tiếp theo)")
                        future_prediction = generate_future_prediction(data_summary)
                        st.info(future_prediction)
                        st.markdown("---")
                        st.warning(
                            "⚠️ **LƯU Ý QUAN TRỌNG:** Đây là phân tích AI dựa trên dữ liệu lịch sử. "
                            "Không được coi là lời khuyên đầu tư. Thị trường chứng khoán có rủi ro cao. "
                            "Vui lòng tham khảo ý kiến chuyên gia tài chính trước khi đưa ra quyết định đầu tư."
                        )
                else:
                    st.error(ai_prediction)
        else:
            st.info("� Click vào button 'Get AI Market Prediction' trong sidebar để nhận được phân tích AI về thị trường")
    
    else:  # Upload CSV Files
        st.markdown('<div class="section-header">📁 Upload CSV Files</div>', unsafe_allow_html=True)
        
        st.info("Upload CSV files with stock data. Supports multiple formats including VN30 format.")
        
        with st.expander("📋 Supported CSV Formats", expanded=False):
            st.write("**Format 1 - Standard format:**")
            st.write("Columns: date, open, high, low, close, volume, turnover")
            st.write("Filename: {STOCK_CODE}_{something}.csv")
            
            st.write("\n**Format 2 - VN30 format:**")
            st.write("Columns: Date;Close;Open;High;Low;Volumn;% turnover")
            st.write("Uses semicolon separator and comma as decimal separator")
            
            st.write("\n**Format 3 - International format:**")
            st.write("Standard comma-separated format with dot as decimal separator")
        
        uploaded_files = st.file_uploader(
            "Choose CSV files",
            accept_multiple_files=True,
            type=['csv'],
            help="You can upload multiple CSV files. The system will automatically detect the format."
        )
        
        if uploaded_files:
            st.success(f"✅ Uploaded {len(uploaded_files)} files")
            
            # Show uploaded files info
            with st.expander("📄 Uploaded Files Details"):
                for i, uploaded_file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
            
            # Save uploaded files temporarily and clean up first
            temp_dir = "/tmp/stock_data"
            
            # Clean up any existing files first
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            if st.button("🔄 Process Uploaded Data", type="primary"):
                with st.spinner("Processing uploaded CSV files..."):
                    try:
                        # Step 1: Load and process data
                        st.info("Step 1: Loading and processing CSV files...")
                        preprocessor = DataPreprocessor()
                        merged_data = preprocessor.load_and_process_all(temp_dir)
                        
                        if merged_data.empty:
                            st.error("❌ No data could be processed from the uploaded files. Please check the file format.")
                            return
                        
                        st.success(f"✅ Successfully processed data! Shape: {merged_data.shape}")
                        st.info(f"Available columns: {list(merged_data.columns)}")
                        
                        # Step 2: Calculate time duration
                        st.info("Step 2: Calculating time duration...")
                        uploaded_time_duration = calculate_time_duration(merged_data)
                        st.success(f"✅ Time duration: {uploaded_time_duration}")
                        
                        # Validate required columns
                        required_columns = ['close']
                        missing_columns = [col for col in required_columns if col not in merged_data.columns]
                        
                        if missing_columns:
                            st.error(f"❌ Missing required columns: {missing_columns}")
                            st.error("Your CSV file must contain at least a 'close' price column.")
                            return
                        
                        # Display basic info for uploaded data
                        st.markdown('<div class="section-header">📊 Uploaded Data Analysis</div>', unsafe_allow_html=True)
                        
                        try:
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Time Duration", uploaded_time_duration.split(' (')[0])
                            with col2:
                                st.metric("Total Records", len(merged_data))
                            with col3:
                                if 'close' in merged_data.columns:
                                    st.metric("Latest Price", f"{merged_data['close'].iloc[-1]:.2f}")
                                else:
                                    st.metric("Latest Price", "N/A")
                            with col4:
                                if 'return' in merged_data.columns:
                                    latest_return = merged_data['return'].iloc[-1] if not pd.isna(merged_data['return'].iloc[-1]) else 0
                                    st.metric("Latest Change %", f"{latest_return:.2f}%")
                                else:
                                    st.metric("Latest Change %", "N/A")
                            with col5:
                                if 'target' in merged_data.columns:
                                    up_days = (merged_data['target'] == 1).sum()
                                    st.metric("Up Days %", f"{100 * up_days / len(merged_data):.1f}%")
                                else:
                                    st.metric("Up Days %", "N/A")
                        except Exception as e:
                            st.error(f"⚠️ Error displaying metrics: {str(e)}")
                        
                        # Show data preview
                        try:
                            with st.expander("📊 Your Data Analysis", expanded=True):
                                st.write("**Processed data preview (First 10 rows):**")
                                display_data = merged_data.head(10).copy()
                                if 'date' in display_data.columns:
                                    display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
                                st.dataframe(display_data, use_container_width=True)
                        except Exception as e:
                            st.error(f"⚠️ Error displaying data preview: {str(e)}")
                        
                        # Add technical indicators
                        st.markdown('<div class="section-header">🔧 Technical Analysis</div>', unsafe_allow_html=True)
                        
                        # Validate data has required columns for technical indicators
                        required_ta_columns = ['close', 'code']
                        missing_ta_columns = [col for col in required_ta_columns if col not in merged_data.columns]
                        
                        if missing_ta_columns:
                            st.warning(f"⚠️ Cannot calculate technical indicators. Missing columns: {missing_ta_columns}")
                            st.warning("Using original data without technical indicators.")
                            data_with_features = merged_data.copy()
                        else:
                            try:
                                with st.spinner("Adding technical indicators..."):
                                    data_with_features = add_technical_indicators(merged_data)
                                
                                st.success(f"✅ Added technical indicators! Final shape: {data_with_features.shape}")
                                
                                # Debug: Show what indicators were actually added
                                with st.expander("🔍 Technical Indicators Details"):
                                    original_cols = set(merged_data.columns)
                                    new_cols = set(data_with_features.columns) - original_cols
                                    
                                    if new_cols:
                                        ma_indicators = [col for col in new_cols if 'ma_' in col.lower()]
                                        rsi_indicators = [col for col in new_cols if 'rsi' in col.lower()]
                                        bb_indicators = [col for col in new_cols if 'bb_' in col.lower()]
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.write(f"**MA Indicators:** {ma_indicators}")
                                        with col2:
                                            st.write(f"**RSI Indicators:** {rsi_indicators}")
                                        with col3:
                                            st.write(f"**Bollinger Bands:** {bb_indicators}")
                                    else:
                                        st.warning("No technical indicators were successfully calculated")
                            except Exception as e:
                                st.error(f"⚠️ Error calculating technical indicators: {str(e)}")
                                st.warning("Using original data without technical indicators.")
                                data_with_features = merged_data.copy()
                        
                        # Charts
                        st.markdown('<div class="section-header">📈 Charts & Visualization</div>', unsafe_allow_html=True)
                        
                        # Price chart - with validation
                        try:
                            st.write("**Price Chart with Technical Indicators:**")
                            price_fig = plot_price_chart(data_with_features)
                            st.plotly_chart(price_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"⚠️ Error creating price chart: {str(e)}")
                            st.warning("Skipping price chart display.")
                        
                        # RSI chart - with validation
                        try:
                            st.write("**RSI Technical Indicator:**")
                            rsi_fig = plot_technical_indicators(data_with_features)
                            st.plotly_chart(rsi_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"⚠️ Error creating RSI chart: {str(e)}")
                            st.warning("Skipping RSI chart display.")
                        
                        # Store processed data in session state for AI prediction
                        try:
                            st.session_state['uploaded_data'] = merged_data
                            st.session_state['uploaded_features'] = data_with_features
                            st.session_state['uploaded_time_duration'] = uploaded_time_duration
                            st.success("✅ Data stored in session for AI prediction")
                        except Exception as e:
                            st.error(f"⚠️ Error storing data in session: {str(e)}")
                        
                        # Advanced Feature Engineering
                        st.markdown('<div class="section-header">🧠 Advanced Feature Engineering</div>', unsafe_allow_html=True)
                        
                        try:
                            feature_engineer = FeatureEngineer()
                            
                            with st.spinner("Creating additional features..."):
                                # Add more features using the uploaded data
                                enriched_data = feature_engineer.create_price_features(data_with_features)
                                enriched_data = feature_engineer.create_volume_features(enriched_data)
                                enriched_data = feature_engineer.create_lag_features(enriched_data)
                                enriched_data = feature_engineer.create_rolling_features(enriched_data)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Original Features", len(merged_data.columns))
                            with col2:
                                st.metric("Final Features", len(enriched_data.columns))
                        except Exception as e:
                            st.error(f"⚠️ Error in feature engineering: {str(e)}")
                            st.warning("Using data with technical indicators only.")
                            enriched_data = data_with_features.copy()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Original Features", len(merged_data.columns))
                            with col2:
                                st.metric("Final Features", len(enriched_data.columns))
                        
                        # Show final dataset preview
                        st.markdown('<div class="section-header">📋 Final Dataset Preview</div>', unsafe_allow_html=True)
                        
                        st.write("**Final enriched dataset preview (Your Uploaded Data):**")
                        st.dataframe(enriched_data.head(), use_container_width=True)
                        
                        # Model Training Demo
                        st.markdown('<div class="section-header">🤖 Model Training Demo</div>', unsafe_allow_html=True)
                        
                        if st.button("🚀 Run Model Training Demo", type="primary", key="model_training_uploaded"):
                            with st.spinner("Preparing data for modeling..."):
                                # Prepare data for modeling
                                # Show data info before cleaning
                                st.info(f"📊 Data before cleaning: {enriched_data.shape}")
                                st.info(f"📊 NaN values: {enriched_data.isnull().sum().sum()}")
                                
                                # Remove rows with NaN values, but be more flexible
                                clean_data = enriched_data.dropna()
                                
                                st.info(f"📊 Data after cleaning: {clean_data.shape}")
                                
                                # Lower the threshold for minimum data points
                                min_required_samples = min(10, len(enriched_data) // 4)  # More flexible threshold
                                
                                if len(clean_data) >= min_required_samples:
                                    # Split features and target
                                    feature_cols = [col for col in clean_data.columns if col not in ['code', 'target', 'date', 'year', 'month', 'day']]
                                    
                                    # Select only numeric columns for modeling
                                    numeric_features = []
                                    for col in feature_cols:
                                        if pd.api.types.is_numeric_dtype(clean_data[col]):
                                            numeric_features.append(col)
                                    
                                    X = clean_data[numeric_features]
                                    y = clean_data['target']
                                    
                                    st.info(f"📊 Selected {len(numeric_features)} numeric features for modeling")
                                    
                                    # Simple train/test split
                                    split_idx = max(1, int(0.8 * len(clean_data)))  # Ensure at least 1 sample for test
                                    X_train, X_test = X[:split_idx], X[split_idx:]
                                    y_train, y_test = y[:split_idx], y[split_idx:]
                                    
                                    st.success(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
                                    
                                    # Show feature importance simulation
                                    np.random.seed(42)
                                    feature_importance = np.random.rand(min(10, len(numeric_features)))
                                    top_features = np.array(numeric_features[:len(feature_importance)])
                                    
                                    importance_df = pd.DataFrame({
                                        'Feature': top_features,
                                        'Importance': feature_importance
                                    }).sort_values('Importance', ascending=True)
                                    
                                    fig_importance = px.bar(
                                        importance_df,
                                        x='Importance',
                                        y='Feature',
                                        orientation='h',
                                        title='Simulated Feature Importance for Your Data'
                                    )
                                    st.plotly_chart(fig_importance, use_container_width=True)
                                    
                                    # Simulated model results
                                    st.write("**Simulated Model Performance on Your Data:**")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Accuracy", "78.5%")
                                    with col2:
                                        st.metric("Precision", "76.2%")
                                    with col3:
                                        st.metric("Recall", "79.8%")
                                    
                                else:
                                    st.error(f"Not enough clean data for modeling. Need at least {min_required_samples} samples, but only have {len(clean_data)} after cleaning.")
                                    st.error("Try uploading more data or check data quality.")
                        
                        # AI Prediction for uploaded data
                        st.markdown('<div class="section-header">🤖 AI Analysis for Your Data</div>', unsafe_allow_html=True)
                        
                        # Create a dedicated AI prediction button for uploaded data
                        if st.button("🧠 Get AI Prediction for Your Data", type="primary", key="ai_prediction_uploaded"):
                            with st.spinner("AI is analyzing your market data..."):
                                try:
                                    # Validate that we have the required data
                                    if 'uploaded_data' not in st.session_state or st.session_state['uploaded_data'].empty:
                                        st.error("❌ No uploaded data found. Please process your data first.")
                                        st.stop()
                                    
                                    # Get processed data from session state
                                    processed_data = st.session_state['uploaded_data']
                                    
                                    # Validate required columns
                                    required_columns = ['close']
                                    missing_columns = [col for col in required_columns if col not in processed_data.columns]
                                    
                                    if missing_columns:
                                        st.error(f"❌ Missing required columns: {missing_columns}")
                                        st.error("Please ensure your CSV file contains at least a 'close' price column.")
                                        st.stop()
                                    
                                    # Calculate latest return safely
                                    if 'return' in processed_data.columns and len(processed_data) > 1:
                                        latest_return = processed_data['return'].iloc[-1] if not pd.isna(processed_data['return'].iloc[-1]) else 0
                                    else:
                                        # Calculate manually if return column doesn't exist
                                        if len(processed_data) >= 2:
                                            latest_close = processed_data['close'].iloc[-1]
                                            prev_close = processed_data['close'].iloc[-2]
                                            latest_return = ((latest_close - prev_close) / prev_close) * 100
                                        else:
                                            latest_return = 0
                                    
                                    # Get time duration
                                    uploaded_time_duration = st.session_state.get('uploaded_time_duration', 'Unknown duration')
                                    
                                    # Prepare data summary for uploaded data with safe access
                                    uploaded_summary = {
                                        'total_days': len(processed_data),
                                        'time_duration': uploaded_time_duration,
                                        'current_price': processed_data['close'].iloc[-1] if len(processed_data) > 0 else 0,
                                        'latest_change': latest_return,
                                        'up_days_ratio': 100 * (processed_data['target'] == 1).sum() / len(processed_data) if 'target' in processed_data.columns and len(processed_data) > 0 else 50,
                                        'highest_price': processed_data['close'].max() if len(processed_data) > 0 else 0,
                                        'lowest_price': processed_data['close'].min() if len(processed_data) > 0 else 0,
                                        'avg_volatility': abs(processed_data['return']).mean() if 'return' in processed_data.columns and len(processed_data) > 0 else 0
                                    }
                                    
                                    # Use hardcoded API key
                                    api_key = "AIzaSyDMs-iLWgB7NuoCtJLqEj4SwG3qhM3B-gQ"
                                    
                                    # Get AI prediction for uploaded data
                                    ai_prediction_uploaded = get_gemini_prediction(uploaded_summary, api_key)
                                    
                                    # Display prediction with enhanced UI
                                    st.markdown("---")
                                    st.markdown("### 🔮 AI Analysis of Your Market Data")
                                    
                                    if "Lỗi" not in ai_prediction_uploaded:
                                        # Create an enhanced display for uploaded data analysis
                                        with st.container():
                                            st.success("✅ Your Data Analysis Complete!")
                                            
                                            # Show key metrics in columns
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("Data Points", f"{uploaded_summary['total_days']:,}")
                                            with col2:
                                                if uploaded_summary['current_price'] > 0:
                                                    st.metric("Current Price", f"{uploaded_summary['current_price']:.2f}")
                                            with col3:
                                                st.metric("Latest Change", f"{uploaded_summary['latest_change']:.2f}%")
                                            with col4:
                                                st.metric("Up Days Ratio", f"{uploaded_summary['up_days_ratio']:.1f}%")
                                            
                                            st.markdown("---")
                                            
                                            # Display the AI prediction in a styled container with black background
                                            st.markdown("#### 📊 Detailed Analysis of Your Data:")
                                            
                                            # Create a styled container for the prediction (same as sample demo)
                                            prediction_container = st.container()
                                            with prediction_container:
                                                formatted_response = format_gemini_response(ai_prediction_uploaded)
                                                st.markdown(
                                                    f"""
                                                    <div style='background-color: #000000; color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 10px 0;'>
                                                        {formatted_response}
                                                    </div>
                                                    """,
                                                    unsafe_allow_html=True
                                                )
                                            
                                            # Add future prediction for uploaded data
                                            st.markdown("#### 🔮 Extended Prediction (Next 10 Years)")
                                            future_prediction_uploaded = generate_future_prediction(uploaded_summary)
                                            st.info(future_prediction_uploaded)
                                            
                                            # Add disclaimer with prominent styling
                                            st.markdown("---")
                                            st.warning(
                                                "⚠️ **LƯU Ý QUAN TRỌNG:** Đây là phân tích AI dựa trên dữ liệu lịch sử của bạn. "
                                                "Không được coi là lời khuyên đầu tư. Thị trường chứng khoán có rủi ro cao. "
                                                "Vui lòng tham khảo ý kiến chuyên gia tài chính trước khi đưa ra quyết định đầu tư."
                                            )
                                    else:
                                        st.error(ai_prediction_uploaded)
                                
                                except Exception as e:
                                    st.error(f"Error processing uploaded data for AI prediction: {str(e)}")
                                    st.error("Please make sure your data was processed successfully before requesting AI prediction.")
                        
                        # Info message when no AI prediction button is pressed yet
                        else:
                            st.info("💡 Click the 'Get AI Prediction for Your Data' button above to analyze your uploaded data with AI")
                        
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
        
        # If no files uploaded yet, show info
        else:
            st.info("📁 Please upload CSV files to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>📊 Stock Market Prediction Demo</p>
            
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
