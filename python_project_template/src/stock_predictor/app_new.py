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

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessor import DataPreprocessor
from data.features import add_technical_indicators, FeatureEngineer
from models.traditional import TraditionalModels

st.set_page_config(
    page_title="Stock Market Prediction Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
def create_sample_data(n_days=365):
    """Create sample stock data for demonstration."""
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
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
    
    df['return'] = df['close'].pct_change() * 100
    df['return'] = df['return'].round(2).fillna(0)
    df['target'] = (df['return'] > 0).astype(int)
    
    return df

def plot_price_chart(data):
    """Create price chart with technical indicators."""
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=list(range(len(data))),
        y=data['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add moving averages if available
    if 'ma_5' in data.columns:
        fig.add_trace(go.Scatter(
            x=list(range(len(data))),
            y=data['ma_5'],
            mode='lines',
            name='MA 5',
            line=dict(color='orange', width=1)
        ))
    
    if 'ma_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=list(range(len(data))),
            y=data['ma_20'],
            mode='lines',
            name='MA 20',
            line=dict(color='red', width=1)
        ))
    
    # Add Bollinger Bands if available
    if all(col in data.columns for col in ['bb_bbh', 'bb_bbm', 'bb_bbl']):
        fig.add_trace(go.Scatter(
            x=list(range(len(data))),
            y=data['bb_bbh'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(data))),
            y=data['bb_bbl'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            showlegend=False
        ))
    
    fig.update_layout(
        title='Stock Price with Technical Indicators',
        xaxis_title='Time',
        yaxis_title='Price',
        height=500,
        showlegend=True
    )
    
    return fig

def plot_technical_indicators(data):
    """Plot technical indicators."""
    fig = go.Figure()
    
    if 'rsi_14' in data.columns:
        fig.add_trace(go.Scatter(
            x=list(range(len(data))),
            y=data['rsi_14'],
            mode='lines',
            name='RSI (14)',
            line=dict(color='purple', width=2)
        ))
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    fig.update_layout(
        title='RSI Technical Indicator',
        xaxis_title='Time',
        yaxis_title='RSI Value',
        height=300,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

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

def get_ai_prediction(data, data_source="sample"):
    """
    Get AI prediction using Gemini API.
    
    Args:
        data: Processed stock data
        data_source: Either 'sample' or 'upload'
    
    Returns:
        AI prediction text or None if error
    """
    try:
        # Configure API key
        api_key = "AIzaSyDMs-iLWgB7NuoCtJLqEj4SwG3qhM3B-gQ"
        genai.configure(api_key=api_key)
        
        # Create model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare data summary
        total_records = len(data)
        if 'date' in data.columns:
            date_range = f"from {data['date'].min()} to {data['date'].max()}"
        else:
            date_range = "Data without date information"
            
        latest_price = data['close'].iloc[-1] if 'close' in data.columns else "N/A"
        
        if 'return' in data.columns and not pd.isna(data['return'].iloc[-1]):
            latest_return = data['return'].iloc[-1]
        else:
            latest_return = 0
            
        if 'target' in data.columns:
            up_days_pct = (data['target'] == 1).sum() / len(data) * 100
        else:
            up_days_pct = 50  # Default
            
        price_max = data['close'].max() if 'close' in data.columns else "N/A"
        price_min = data['close'].min() if 'close' in data.columns else "N/A"
        
        avg_volatility = data['return'].std() if 'return' in data.columns else "N/A"
        
        # Create analysis prompt
        data_type_text = "uploaded CSV data" if data_source == "upload" else "VN30 sample data"
        
        prompt = f"""
        Phân tích dữ liệu thị trường chứng khoán sau và đưa ra dự báo chi tiết:

        **Thông tin tổng quan về {data_type_text}:**
        - Tổng số bản ghi: {total_records:,}
        - Khoảng thời gian: {date_range}
        - Giá đóng cửa hiện tại: {latest_price}
        - Thay đổi giá gần nhất: {latest_return:.2f}%
        - Tỷ lệ ngày tăng giá: {up_days_pct:.1f}%
        - Giá cao nhất: {price_max}
        - Giá thấp nhất: {price_min}
        - Độ biến động trung bình: {avg_volatility}

        **Yêu cầu phân tích:**
        1. Đánh giá xu hướng tổng quát của thị trường
        2. Phân tích mức độ rủi ro và biến động
        3. Dự báo ngắn hạn (1-3 tháng) và trung hạn (6-12 tháng)
        4. Đưa ra khuyến nghị đầu tư cụ thể
        5. Các yếu tố cần theo dõi

        Hãy trả lời bằng tiếng Việt, ngắn gọn và dễ hiểu.
        """
        
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Lỗi khi gọi API AI: {str(e)}"

def display_ai_prediction_result(prediction_text, data_source):
    """
    Display AI prediction result in a modal-like expander.
    
    Args:
        prediction_text: The AI prediction text
        data_source: Either 'sample' or 'upload'
    """
    data_type = "VN30 Sample Data" if data_source == "sample" else "Your Uploaded Data"
    
    # Create expander with custom styling
    with st.expander(f"🤖 AI Market Analysis for {data_type}", expanded=True):
        # Replace newlines with HTML breaks
        formatted_text = prediction_text.replace('\n', '<br>')
        
        st.markdown(
            f"""
            <div style="
                background-color: #1e1e1e;
                color: white;
                padding: 20px;
                border-radius: 10px;
                border: 2px solid #4CAF50;
                margin: 10px 0;
            ">
                <h3 style="color: #4CAF50; margin-top: 0;">🧠 AI Market Prediction</h3>
                <div style="line-height: 1.6; font-size: 14px;">
                    {formatted_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def show_popup_message(message, message_type="success"):
    """
    Show a popup message using Streamlit's toast functionality.
    
    Args:
        message: The message to display
        message_type: Type of message ('success', 'error', 'warning', 'info')
    """
    # Use Streamlit's toast for popup notifications
    if hasattr(st, 'toast'):
        # Use toast if available (newer Streamlit versions)
        if message_type == "success":
            st.toast(f"✅ {message}", icon="✅")
        elif message_type == "error":
            st.toast(f"❌ {message}", icon="❌")
        elif message_type == "warning":
            st.toast(f"⚠️ {message}", icon="⚠️")
        elif message_type == "info":
            st.toast(f"ℹ️ {message}", icon="ℹ️")
    else:
        # Fallback to regular messages for older Streamlit versions
        if message_type == "success":
            st.success(f"✅ {message}")
        elif message_type == "error":
            st.error(f"❌ {message}")
        elif message_type == "warning":
            st.warning(f"⚠️ {message}")
        elif message_type == "info":
            st.info(f"ℹ️ {message}")

def main():
    """Main application function."""
    
    st.markdown('<div class="main-header">📈 Stock Market Prediction Demo</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;">Dự báo chỉ số thị trường sử dụng Machine Learning</div>', unsafe_allow_html=True)
    
    # Initialize session state for UI stability
    if 'processed_sample_data' not in st.session_state:
        st.session_state['processed_sample_data'] = False
    if 'processed_upload_data' not in st.session_state:
        st.session_state['processed_upload_data'] = False
    if 'ai_prediction_result' not in st.session_state:
        st.session_state['ai_prediction_result'] = None
    if 'ai_prediction_source' not in st.session_state:
        st.session_state['ai_prediction_source'] = None
    
    # Sidebar
    st.sidebar.header("⚙️ Hệ Thống Chính")
    
    # AI Prediction Button - Check if we have data to analyze
    has_data = 'current_data' in st.session_state and st.session_state['current_data'] is not None
    data_source = st.session_state.get('current_data_source', 'sample')
    
    st.sidebar.markdown("### 🤖 AI Prediction")
    if has_data:
        data_type = "VN30 Sample Data" if data_source == "sample" else "Your Uploaded Data"
        ai_button_help = f"Get AI-powered market analysis for {data_type}"
    else:
        ai_button_help = "Load data first (either sample or upload) to enable AI prediction"
    
    use_ai_prediction = st.sidebar.button(
        "🧠 Get AI Market Prediction",
        type="primary",
        help=ai_button_help,
        use_container_width=True,
        disabled=not has_data
    )
    
    # Handle AI prediction request
    if use_ai_prediction and has_data:
        current_data = st.session_state['current_data']
        current_source = st.session_state['current_data_source']
        
        with st.spinner("🤖 Getting AI prediction..."):
            prediction = get_ai_prediction(current_data, current_source)
            st.session_state['ai_prediction_result'] = prediction
            st.session_state['ai_prediction_source'] = current_source
    
    # Demo options
    demo_option = st.sidebar.selectbox(
        "Choose Demo Type",
        ["Sample Data Demo", "Upload CSV Files"]
    )
    
    # Clear AI prediction when switching between demo types
    if 'current_demo_type' not in st.session_state:
        st.session_state['current_demo_type'] = demo_option
    elif st.session_state['current_demo_type'] != demo_option:
        # User switched tabs, clear AI prediction
        st.session_state['ai_prediction_result'] = None
        st.session_state['ai_prediction_source'] = None
        st.session_state['current_demo_type'] = demo_option
    
    # Display AI prediction result if available and matches current demo type
    if (st.session_state['ai_prediction_result'] and 
        st.session_state.get('ai_prediction_source') == data_source):
        display_ai_prediction_result(
            st.session_state['ai_prediction_result'], 
            st.session_state['ai_prediction_source']
        )
    
    if demo_option == "Sample Data Demo":
        st.markdown('<div class="section-header">📊 Sample Data Analysis</div>', unsafe_allow_html=True)
        
        # Check if sample data is already processed
        if not st.session_state.get('processed_sample_data', False):
            vn30_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/VN30_demo.csv"
            
            try:
                with st.spinner("Loading VN30 data..."):
                    preprocessor = DataPreprocessor()
                    vn30_data = preprocessor._read_csv_flexible(vn30_file_path)
                    
                    if vn30_data is None:
                        raise Exception("Could not read VN30 CSV file")
                    
                    vn30_data = preprocessor._normalize_data_format(vn30_data, "VN30")
                    
                    if vn30_data is None or vn30_data.empty:
                        raise Exception("Failed to normalize VN30 data format")
                    
                    vn30_data = preprocessor._calculate_returns_and_targets(vn30_data)
                    
                    sample_data = vn30_data.copy()
                    
                    # Add technical indicators
                    data_with_features = add_technical_indicators(sample_data)
                    
                    # Feature engineering
                    feature_engineer = FeatureEngineer()
                    enriched_data = feature_engineer.create_price_features(data_with_features)
                    enriched_data = feature_engineer.create_volume_features(enriched_data)
                    enriched_data = feature_engineer.create_lag_features(enriched_data)
                    enriched_data = feature_engineer.create_rolling_features(enriched_data)
                    
                    # Store in session state
                    st.session_state['current_data'] = enriched_data
                    st.session_state['current_data_source'] = 'sample'
                    st.session_state['time_duration'] = calculate_time_duration(sample_data)
                    st.session_state['sample_base_data'] = sample_data
                    st.session_state['processed_sample_data'] = True
                
                show_popup_message(f"Loaded VN30 data! Shape: {sample_data.shape}", "success")
                show_popup_message("Using real VN30 index data instead of synthetic data", "info")
                
            except Exception as e:
                show_popup_message(f"Could not load VN30 data: {str(e)}. Using synthetic data instead.", "warning")
                # Fallback to synthetic data
                with st.spinner("Generating sample data..."):
                    sample_data = create_sample_data()
                    # Process synthetic data similarly
                    data_with_features = add_technical_indicators(sample_data)
                    feature_engineer = FeatureEngineer()
                    enriched_data = feature_engineer.create_price_features(data_with_features)
                    enriched_data = feature_engineer.create_volume_features(enriched_data)
                    enriched_data = feature_engineer.create_lag_features(enriched_data)
                    enriched_data = feature_engineer.create_rolling_features(enriched_data)
                    
                    # Store in session state
                    st.session_state['current_data'] = enriched_data
                    st.session_state['current_data_source'] = 'sample'
                    st.session_state['time_duration'] = calculate_time_duration(sample_data)
                    st.session_state['sample_base_data'] = sample_data
                    st.session_state['processed_sample_data'] = True
        
        # Use data from session state
        enriched_data = st.session_state['current_data']
        sample_data = st.session_state['sample_base_data']
        
        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Days", len(sample_data))
        with col2:
            st.metric("Latest Price", f"{sample_data['close'].iloc[-1]:.2f}")
        with col3:
            if 'return' in sample_data.columns:
                st.metric("Price Change %", f"{sample_data['return'].iloc[-1]:.2f}%")
            else:
                st.metric("Price Change %", "N/A")
        with col4:
            if 'target' in sample_data.columns:
                up_days = (sample_data['target'] == 1).sum()
                st.metric("Up Days %", f"{100 * up_days / len(sample_data):.1f}%")
            else:
                st.metric("Up Days %", "N/A")
        
        # Add technical indicators
        st.markdown('<div class="section-header">🔧 Feature Engineering</div>', unsafe_allow_html=True)
        
        show_popup_message(f"Added technical indicators! Shape: {enriched_data.shape}", "success")
        
        # Show feature columns
        with st.expander("View Technical Indicators"):
            new_features = set(enriched_data.columns) - set(sample_data.columns)
            st.write("**New technical indicators added:**")
            for feature in sorted(new_features):
                st.write(f"• {feature}")
        
        # Visualizations
        st.markdown('<div class="section-header">📈 Charts & Analysis</div>', unsafe_allow_html=True)
        
        # Price chart
        price_fig = plot_price_chart(enriched_data)
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Technical indicators
        if 'rsi_14' in enriched_data.columns:
            rsi_fig = plot_technical_indicators(enriched_data)
            st.plotly_chart(rsi_fig, use_container_width=True)
        
        # Feature engineering demo
        st.markdown('<div class="section-header">🧠 Advanced Feature Engineering</div>', unsafe_allow_html=True)
        
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
            st.write("**Date Range:**", f"{sample_data['date'].min()} to {sample_data['date'].max()}")
            st.write("**Records:**", len(sample_data))
            if 'volume' in sample_data.columns:
                st.write("**Avg Daily Volume:**", f"{sample_data['volume'].mean():.2f}M")
            st.write("**Features after enrichment:**", len(enriched_data.columns))
        
        # Model demonstration
        st.markdown('<div class="section-header">🤖 Model Training Demo</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Run Model Training Demo", type="primary", key="model_demo_sample"):
            with st.spinner("Preparing data for modeling..."):
                # Prepare data for modeling
                # Remove rows with NaN values
                clean_data = enriched_data.dropna()
                
                if len(clean_data) > 50:  # Ensure we have enough data
                    # Split features and target
                    feature_cols = [col for col in clean_data.columns if col not in ['code', 'target']]
                    X = clean_data[feature_cols]
                    y = clean_data['target']
                    
                    # Simple train/test split
                    split_idx = int(0.8 * len(clean_data))
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    st.success(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
                    
                    # Show feature importance simulation
                    np.random.seed(42)
                    feature_importance = np.random.rand(min(10, len(feature_cols)))
                    top_features = np.array(feature_cols[:len(feature_importance)])
                    
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
                    show_popup_message("Not enough clean data for modeling", "error")
    
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
            show_popup_message(f"Uploaded {len(uploaded_files)} files", "success")
            
            # Show uploaded files info
            with st.expander("📄 Uploaded Files Details"):
                for i, uploaded_file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
            
            # Check if data is already processed
            if not st.session_state.get('processed_upload_data', False):
                # Save uploaded files temporarily and clean up first
                temp_dir = "/tmp/stock_data"
                
                # Clean up any existing files first
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)
                
                # Save files to temp directory
                for uploaded_file in uploaded_files:
                    with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                show_popup_message("Files saved to temporary directory", "success")
                
                # Add Process button
                process_button = st.button(
                    "🔄 Process Uploaded Data", 
                    type="primary", 
                    key="process_uploaded_data",
                    help="Click to process your uploaded CSV files and generate analysis"
                )
                
                if process_button:
                    with st.spinner("Processing uploaded CSV files..."):
                        try:
                            preprocessor = DataPreprocessor()
                            merged_data = preprocessor.load_and_process_all(temp_dir)
                            
                            if merged_data.empty:
                                show_popup_message("No data could be processed from the uploaded files. Please check the file format.", "error")
                            else:
                                show_popup_message(f"Successfully processed data! Shape: {merged_data.shape}", "success")
                                
                                # Add technical indicators
                                with st.spinner("Adding technical indicators..."):
                                    data_with_features = add_technical_indicators(merged_data)
                                
                                show_popup_message(f"Added technical indicators! Final shape: {data_with_features.shape}", "success")
                                
                                # Feature engineering
                                feature_engineer = FeatureEngineer()
                                
                                with st.spinner("Creating additional features..."):
                                    enriched_data = feature_engineer.create_price_features(data_with_features)
                                    enriched_data = feature_engineer.create_volume_features(enriched_data)
                                    enriched_data = feature_engineer.create_lag_features(enriched_data)
                                    enriched_data = feature_engineer.create_rolling_features(enriched_data)
                                
                                # Store in session state
                                st.session_state['current_data'] = enriched_data
                                st.session_state['current_data_source'] = 'upload'
                                st.session_state['time_duration'] = calculate_time_duration(merged_data)
                                st.session_state['upload_base_data'] = merged_data
                                st.session_state['processed_upload_data'] = True
                                
                                show_popup_message("Data processing completed and stored!", "success")
                                
                        except Exception as e:
                            show_popup_message(f"Error processing files: {str(e)}", "error")
        
        # Display processed data if available in session state
        if st.session_state.get('processed_upload_data', False) and 'current_data' in st.session_state:
            enriched_data = st.session_state['current_data']
            upload_base_data = st.session_state['upload_base_data']
            uploaded_time_duration = st.session_state['time_duration']
            
            st.markdown('<div class="section-header">📊 Uploaded Data Analysis</div>', unsafe_allow_html=True)
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Time Duration", uploaded_time_duration.split(' (')[0])
            with col2:
                st.metric("Total Records", len(enriched_data))
            with col3:
                if 'close' in enriched_data.columns:
                    st.metric("Latest Price", f"{enriched_data['close'].iloc[-1]:.2f}")
                else:
                    st.metric("Latest Price", "N/A")
            with col4:
                if 'return' in enriched_data.columns:
                    latest_return = enriched_data['return'].iloc[-1] if not pd.isna(enriched_data['return'].iloc[-1]) else 0
                    st.metric("Latest Change %", f"{latest_return:.2f}%")
                else:
                    st.metric("Latest Change %", "N/A")
            
            # Show data preview
            with st.expander("📊 Your Data Analysis", expanded=True):
                st.write("**Processed data preview (First 10 rows):**")
                display_data = enriched_data.head(10).copy()
                if 'date' in display_data.columns:
                    display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
                st.dataframe(display_data, use_container_width=True)
            
            # Charts
            st.markdown('<div class="section-header">📈 Charts & Visualization</div>', unsafe_allow_html=True)
            
            # Price chart
            st.write("**Price Chart with Technical Indicators:**")
            price_fig = plot_price_chart(enriched_data)
            st.plotly_chart(price_fig, use_container_width=True)
            
            # Technical indicators
            if 'rsi_14' in enriched_data.columns:
                rsi_fig = plot_technical_indicators(enriched_data)
                st.plotly_chart(rsi_fig, use_container_width=True)
            
            # Show final dataset info
            st.markdown('<div class="section-header">📋 Final Dataset Preview</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Features", "14")  # Base features
            with col2:
                st.metric("Final Features", len(enriched_data.columns))
            
            st.write("**Final enriched dataset preview (Your Uploaded Data):**")
            st.dataframe(enriched_data.head(), use_container_width=True)
            
            # Model Training Demo
            st.markdown('<div class="section-header">🤖 Model Training Demo</div>', unsafe_allow_html=True)
            
            if st.button("🚀 Run Model Training Demo", type="primary", key="model_demo_upload"):
                with st.spinner("Preparing data for modeling..."):
                    # Prepare data for modeling
                    clean_data = enriched_data.dropna()
                    
                    if len(clean_data) > 50:  # Ensure we have enough data
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
                        split_idx = max(1, int(0.8 * len(clean_data)))
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
                        show_popup_message(f"Not enough clean data for modeling. Need at least 50 samples, but only have {len(clean_data)} after cleaning.", "error")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>📊 Stock Market Prediction Demo | Based on Machine Learning Fusion Techniques</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
