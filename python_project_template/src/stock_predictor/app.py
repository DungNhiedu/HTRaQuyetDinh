"""
Ứng dụng web Streamlit cho Hệ Thống Dự Báo Thị Trường Chứng Khoán
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

def show_popup_message(message, message_type="success"):
    """Show popup message using streamlit toast for 3 seconds."""
    if message_type == "success":
        st.toast(f"✅ {message}", icon="✅")
    elif message_type == "error":
        st.toast(f"❌ {message}", icon="❌")
    elif message_type == "warning":
        st.toast(f"⚠️ {message}", icon="⚠️")
    elif message_type == "info":
        st.toast(f"ℹ️ {message}", icon="ℹ️")
    else:
        st.toast(message)

st.set_page_config(
    page_title="Hệ Thống Dự Báo Thị Trường Chứng Khoán",
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
        title='Chỉ Số VN30 với Các Chỉ Báo Kỹ Thuật',
        xaxis_title='Ngày',
        yaxis_title='Giá',
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
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Mua quá mức (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Bán quá mức (30)")
            
            fig.update_layout(
                title='Chỉ Báo Kỹ Thuật RSI',
                xaxis_title='Ngày',
                yaxis_title='Giá Trị RSI',
                height=300,
                yaxis=dict(range=[0, 100])
            )
        else:
            # No valid RSI data
            fig.add_annotation(
                text="Không có dữ liệu RSI",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title='Chỉ Báo Kỹ Thuật RSI (Không có dữ liệu)',
                xaxis_title='Ngày',
                yaxis_title='Giá Trị RSI',
                height=300
            )
    else:
        # No RSI columns found
        fig.add_annotation(
            text="Không tìm thấy chỉ báo RSI",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title='Chỉ Báo Kỹ Thuật RSI (Không có sẵn)',
            xaxis_title='Ngày',
            yaxis_title='Giá Trị RSI',
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
        return f"{years:.1f} năm ({start_date.strftime('%Y-%m-%d')} đến {end_date.strftime('%Y-%m-%d')})"
    else:
        # Fallback: estimate from row count
        years = len(data) / 365.25
        return f"~{years:.1f} năm ({len(data)} điểm dữ liệu)"

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
    st.markdown('<div class="main-header">📈 Hệ Thống Dự Báo Thị Trường</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;">Dự báo chỉ số thị trường sử dụng Machine Learning và AI</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Hệ Thống Chính")
    
    # Demo options - define this first
    demo_option = st.sidebar.selectbox(
        "Chọn Kiểu Demo",
        ["Demo Dữ Liệu Mẫu", "Tải File CSV", "Demo Dự Báo"]
    )
    
    # AI Prediction Button - only show if we have data
    st.sidebar.markdown("### 🤖 Dự Báo AI")
    
    # Check if we have any data available for AI prediction
    has_sample_data = demo_option == "Demo Dữ Liệu Mẫu"
    
    # For Upload CSV, check if process button was clicked or data already exists
    has_uploaded_data = False
    if demo_option == "Tải File CSV":
        # Check if data is already processed
        data_processed = st.session_state.get('upload_processed', False)
        data_exists = 'uploaded_data' in st.session_state
        
        has_uploaded_data = data_processed and data_exists
    
    if has_sample_data or has_uploaded_data:
        use_ai_prediction = st.sidebar.button(
            "🧠 Nhận Dự Báo Thị Trường AI",
            type="primary",
            help="Nhấp để nhận phân tích và dự báo thị trường bằng AI",
            use_container_width=True
        )
    else:
        use_ai_prediction = False
        if demo_option == "Tải File CSV":
            if not st.session_state.get('upload_processed', False):
                st.sidebar.info("💡 Vui lòng tải lên và xử lý file CSV trước")
            else:
                st.sidebar.warning("⚠️ Dữ liệu không sẵn sàng cho dự báo AI")
        else:
            st.sidebar.info("💡 Chọn demo hoặc tải dữ liệu để sử dụng dự báo AI")
    
    
    if demo_option == "Demo Dữ Liệu Mẫu":
        st.markdown('<div class="section-header">📊 Phân Tích Dữ Liệu Mẫu</div>', unsafe_allow_html=True)
        
        # Load real VN30 data instead of synthetic data
        vn30_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/VN30_demo.csv"
        
        try:
            with st.spinner("Đang tải dữ liệu VN30..."):
                # Load and process VN30 data using DataPreprocessor
                preprocessor = DataPreprocessor()
                vn30_data = preprocessor._read_csv_flexible(vn30_file_path)
                
                if vn30_data is None:
                    raise Exception("Không thể đọc file CSV VN30")
                
                # Process the VN30 data directly
                vn30_data = preprocessor._normalize_data_format(vn30_data, "VN30")
                
                if vn30_data is None or vn30_data.empty:
                    raise Exception("Không thể chuẩn hóa định dạng dữ liệu VN30")
                
                # Calculate returns and targets
                vn30_data = preprocessor._calculate_returns_and_targets(vn30_data)
                
                # Use this as our sample data
                sample_data = vn30_data.copy()
                
            show_popup_message("Đã tải thành công dữ liệu VN30!", "success")
            
            # Show actual VN30 data analysis
            with st.expander("📊 Phân Tích Dữ Liệu VN30", expanded=True):
                st.write("**Nguồn Dữ Liệu:** Dữ Liệu Lịch Sử Chỉ Số VN30 Việt Nam")
                
                # Show raw data sample
                st.write("**Dữ Liệu VN30 Thô (10 dòng đầu):**")
                display_data = sample_data.head(10).copy()
                if 'date' in display_data.columns:
                    display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
                st.dataframe(display_data, use_container_width=True)
            
            show_popup_message("Đang sử dụng dữ liệu chỉ số VN30 thực tế", "info")
            
        except Exception as e:
            show_popup_message(f"Không thể tải dữ liệu VN30: {str(e)}. Sử dụng dữ liệu tổng hợp thay thế.", "warning")
            # Fallback to synthetic data
            with st.spinner("Đang tạo dữ liệu mẫu..."):
                sample_data = create_sample_data()
        
        # Calculate time duration
        time_duration = calculate_time_duration(sample_data)
        
        # Display basic info
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Thời Gian", time_duration.split(' (')[0])  # Just the years part
        with col2:
            st.metric("Tổng Số Ngày", len(sample_data))
        with col3:
            st.metric("Giá Mới Nhất", f"{sample_data['close'].iloc[-1]:.2f}")
        with col4:
            if 'return' in sample_data.columns:
                st.metric("Thay Đổi Giá %", f"{sample_data['return'].iloc[-1]:.2f}%")
            else:
                st.metric("Thay Đổi Giá %", "N/A")
        with col5:
            if 'target' in sample_data.columns:
                up_days = (sample_data['target'] == 1).sum()
                st.metric("% Ngày Tăng", f"{100 * up_days / len(sample_data):.1f}%")
            else:
                st.metric("% Ngày Tăng", "N/A")
        
        # Add technical indicators
        st.markdown('<div class="section-header">🔧 Kỹ Thuật Xây Dựng Đặc Trưng</div>', unsafe_allow_html=True)
        
        with st.spinner("Đang thêm các chỉ báo kỹ thuật..."):
            data_with_features = add_technical_indicators(sample_data)
        
        show_popup_message("Đã thêm thành công các chỉ báo kỹ thuật!", "success")
        
        # Debug: Show what indicators were actually added
        with st.expander("🔍 Gỡ Lỗi: Trạng Thái Chỉ Báo Kỹ Thuật"):
            st.write("**Các cột có sẵn sau khi thêm chỉ báo kỹ thuật:**")
            all_cols = list(data_with_features.columns)
            original_cols = list(sample_data.columns)
            new_cols = [col for col in all_cols if col not in original_cols]
            
            st.write(f"Cột gốc ({len(original_cols)}): {original_cols}")
            st.write(f"Chỉ báo kỹ thuật mới ({len(new_cols)}): {new_cols}")
            
            # Check for specific MA indicators
            ma_indicators = [col for col in new_cols if 'ma_' in col.lower()]
            rsi_indicators = [col for col in new_cols if 'rsi' in col.lower()]
            bb_indicators = [col for col in new_cols if 'bb_' in col.lower()]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Chỉ Báo MA:** {ma_indicators}")
            with col2:
                st.write(f"**Chỉ Báo RSI:** {rsi_indicators}")
            with col3:
                st.write(f"**Dải Bollinger:** {bb_indicators}")
            
            # Show sample of technical indicator values
            if len(new_cols) > 0:
                st.write("**Giá trị mẫu (5 dòng cuối):**")
                sample_tech = data_with_features[new_cols].tail()
                st.dataframe(sample_tech, use_container_width=True)
        
        # Show feature columns
        with st.expander("Xem Các Chỉ Báo Kỹ Thuật"):
            new_features = set(data_with_features.columns) - set(sample_data.columns)
            st.write("**Chỉ báo kỹ thuật mới được thêm:**")
            for feature in sorted(new_features):
                st.write(f"• {feature}")
        
        # Visualizations
        st.markdown('<div class="section-header">📈 Biểu Đồ & Phân Tích</div>', unsafe_allow_html=True)
        
        # Price chart
        st.write("**Biểu Đồ Giá Chỉ Số VN30 với Các Chỉ Báo Kỹ Thuật:**")
        price_fig = plot_price_chart(data_with_features)
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Show chart info
        with st.expander("📊 Thông Tin Biểu Đồ"):
            available_indicators = []
            ma_cols = [col for col in data_with_features.columns if 'ma_' in col.lower() or 'sma_' in col.lower()]
            bb_cols = [col for col in data_with_features.columns if 'bb_' in col.lower()]
            rsi_cols = [col for col in data_with_features.columns if 'rsi' in col.lower()]
            
            if ma_cols:
                available_indicators.append(f"Đường Trung Bình Động: {ma_cols}")
            if bb_cols:
                available_indicators.append(f"Dải Bollinger: {bb_cols}")
            if rsi_cols:
                available_indicators.append(f"RSI: {rsi_cols}")
            
            if available_indicators:
                st.write("**Chỉ báo kỹ thuật hiển thị:**")
                for indicator in available_indicators:
                    st.write(f"• {indicator}")
            else:
                st.warning("Không tìm thấy chỉ báo kỹ thuật trong dữ liệu")
        
        # Technical indicators chart
        st.write("**Chỉ Báo Kỹ Thuật RSI:**")
        rsi_fig = plot_technical_indicators(data_with_features)
        st.plotly_chart(rsi_fig, use_container_width=True)
        
        # Feature engineering demo
        st.markdown('<div class="section-header">🧠 Kỹ Thuật Xây Dựng Đặc Trưng Nâng Cao</div>', unsafe_allow_html=True)
        
        feature_engineer = FeatureEngineer()
        
        with st.spinner("Đang tạo các đặc trưng bổ sung..."):
            # Add more features using the real VN30 data
            enriched_data = feature_engineer.create_price_features(data_with_features)
            enriched_data = feature_engineer.create_volume_features(enriched_data)
            enriched_data = feature_engineer.create_lag_features(enriched_data)
            enriched_data = feature_engineer.create_rolling_features(enriched_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Đặc Trưng Gốc", len(sample_data.columns))
        with col2:
            st.metric("Đặc Trưng Cuối Cùng", len(enriched_data.columns))
        
        # Show data sample
        st.markdown('<div class="section-header">📋 Xem Trước Dữ Liệu</div>', unsafe_allow_html=True)
        
        st.write("**Xem trước bộ dữ liệu cuối cùng (Dữ Liệu VN30 Thực Tế):**")
        st.dataframe(enriched_data.head(), use_container_width=True)
        
        # Additional info about the VN30 data
        with st.expander("ℹ️ Về Dữ Liệu VN30"):
            st.write("**Nguồn Dữ Liệu:** Dữ Liệu Lịch Sử Chỉ Số VN30 Việt Nam")
            st.write("**Số Bản Ghi:**", len(sample_data))
            st.write("**Đặc trưng sau khi enrich data:**", len(enriched_data.columns))
        
        # Model demonstration
        st.markdown('<div class="section-header">🤖 Demo Huấn Luyện Mô Hình</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Chạy Demo Huấn Luyện Mô Hình", type="primary"):
            with st.spinner("Đang chuẩn bị dữ liệu cho mô hình..."):
                # Prepare data for modeling
                # Show data info before cleaning
                st.info(f"📊 Dữ liệu trước khi làm sạch: {enriched_data.shape}")
                st.info(f"📊 Giá trị NaN: {enriched_data.isnull().sum().sum()}")
                
                # Remove rows with NaN values, but be more flexible
                clean_data = enriched_data.dropna()
                
                st.info(f"📊 Dữ liệu sau khi làm sạch: {clean_data.shape}")
                
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
                    
                    st.info(f"📊 Đã chọn {len(numeric_features)} đặc trưng số cho mô hình")
                    
                    # Simple train/test split
                    split_idx = max(1, int(0.8 * len(clean_data)))  # Ensure at least 1 sample for test
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    show_popup_message(f"Dữ liệu đã được chuẩn bị: {len(X_train)} mẫu huấn luyện, {len(X_test)} mẫu kiểm tra", "success")
                    
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
                        title='Mức Độ Quan Trọng Đặc Trưng Mô Phỏng'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Simulated model results
                    st.write("**Hiệu Suất Mô Hình Mô Phỏng:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Độ Chính Xác", "75.2%")
                    with col2:
                        st.metric("Độ Chính Xác", "73.8%")
                    with col3:
                        st.metric("Độ Nhạy", "76.5%")
                    
                else:
                    st.error(f"Không đủ dữ liệu sạch để mô hình hóa. Cần ít nhất {min_required_samples} mẫu, nhưng chỉ có {len(clean_data)} sau khi làm sạch.")
                    st.error("Hãy thử tải lên nhiều dữ liệu hơn hoặc kiểm tra chất lượng dữ liệu.")
        
        # AI Prediction Section
        st.markdown('<div class="section-header">🤖 Dự Báo Thị Trường Dựa Trên AI</div>', unsafe_allow_html=True)
        
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
            st.info("💡 Nhấp vào nút 'Nhận Dự Báo Thị Trường AI' trong thanh bên để nhận phân tích AI về thị trường")
    
    elif demo_option == "Demo Dự Báo":
        # Import forecaster here to avoid circular imports
        from forecast.forecaster import StockForecaster
        
        st.markdown('<div class="section-header">📈 Demo Dự Báo Giá</div>', unsafe_allow_html=True)
        
        # Initialize forecaster
        forecaster = StockForecaster()
        
        # Load forecast data
        with st.spinner("Đang tải dữ liệu dự báo..."):
            data_loaded = forecaster.load_forecast_data()
        
        if not data_loaded:
            show_popup_message("Không thể tải dữ liệu dự báo. Vui lòng kiểm tra các file dữ liệu.", "error")
            st.error("❌ Không thể tải dữ liệu dự báo từ Desktop")
            st.error("Vui lòng đảm bảo các file sau tồn tại trên Desktop:")
            st.write("- Dữ liệu Lịch sử USD_VND.csv")
            st.write("- dữ liệu lịch sử giá vàng.csv")
            return
        
        show_popup_message(f"Đã tải {len(forecaster.available_symbols)} bộ dữ liệu dự báo", "success")
        
        # Symbol selection
        selected_symbol = st.selectbox(
            "Chọn chỉ số để dự báo:",
            forecaster.available_symbols,
            help="Chọn USD/VND hoặc Gold để xem dự báo"
        )
        
        # Forecast days selection
        forecast_days = st.slider(
            "Số ngày dự báo:",
            min_value=7,
            max_value=90,
            value=30,
            help="Chọn số ngày bạn muốn dự báo vào tương lai"
        )
        
        if st.button("🔮 Tạo Dự Báo", type="primary"):
            with st.spinner(f"Đang tạo dự báo cho {selected_symbol}..."):
                # Create forecast chart
                forecast_chart = forecaster.create_forecast_chart(
                    selected_symbol, 
                    forecast_days=forecast_days,
                    historical_days=90
                )
                
                if forecast_chart is None:
                    show_popup_message("Không thể tạo dự báo. Vui lòng thử lại.", "error")
                    return
                
                # Display chart
                st.plotly_chart(forecast_chart, use_container_width=True)
                
                # Get forecast summary
                summary = forecaster.get_forecast_summary(selected_symbol, forecast_days)
                
                if summary:
                    # Display forecast summary
                    st.markdown("### 📊 Tóm Tắt Dự Báo")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Giá Hiện Tại", f"{summary['current_price']:,.0f}")
                    with col2:
                        st.metric("Giá Dự Báo", f"{summary['forecast_end_price']:,.0f}")
                    with col3:
                        st.metric("Thay Đổi", f"{summary['price_change']:,.0f}")
                    with col4:
                        st.metric("Thay Đổi %", f"{summary['price_change_pct']:.1f}%")
                    
                    # Investment recommendation
                    st.markdown("### 💰 Khuyến Nghị Đầu Tư")
                    
                    # Create colored background based on trend
                    if summary['trend_color'] == 'green':
                        bg_color = "#d4edda"
                        text_color = "#155724"
                        border_color = "#c3e6cb"
                    elif summary['trend_color'] == 'lightgreen':
                        bg_color = "#d1ecf1"
                        text_color = "#0c5460"
                        border_color = "#bee5eb"
                    elif summary['trend_color'] == 'orange':
                        bg_color = "#fff3cd"
                        text_color = "#856404"
                        border_color = "#ffeaa7"
                    else:  # red
                        bg_color = "#f8d7da"
                        text_color = "#721c24"
                        border_color = "#f5c6cb"
                    
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {bg_color}; 
                            color: {text_color}; 
                            padding: 15px; 
                            border-radius: 10px; 
                            border-left: 5px solid {border_color};
                            margin: 10px 0;
                        ">
                            <h4 style="margin: 0; color: {text_color};">Xu Hướng: {summary['trend']}</h4>
                            <p style="margin: 5px 0; color: {text_color};">
                                <strong>Biến động lịch sử:</strong> {summary['historical_volatility']:.2f}%<br>
                                <strong>Giá cao nhất dự kiến:</strong> {summary['max_forecast_price']:,.0f}<br>
                                <strong>Giá thấp nhất dự kiến:</strong> {summary['min_forecast_price']:,.0f}<br>
                                <strong>Giá trung bình dự kiến:</strong> {summary['avg_forecast_price']:,.0f}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Risk assessment
                    st.markdown("### ⚠️ Đánh Giá Rủi Ro")
                    if summary['historical_volatility'] > 5:
                        risk_level = "Cao"
                        risk_color = "#dc3545"  # Red
                        risk_bg_color = "#f8d7da"  # Light red background
                        risk_border_color = "#f5c6cb"  # Red border
                    elif summary['historical_volatility'] > 2:
                        risk_level = "Trung Bình"
                        risk_color = "#fd7e14"  # Orange
                        risk_bg_color = "#fff3cd"  # Light orange background
                        risk_border_color = "#ffeaa7"  # Orange border
                    else:
                        risk_level = "Thấp"
                        risk_color = "#28a745"  # Green
                        risk_bg_color = "#d4edda"  # Light green background
                        risk_border_color = "#c3e6cb"  # Green border
                    
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {risk_bg_color}; 
                            color: #495057; 
                            padding: 15px; 
                            border-radius: 10px; 
                            border: 2px solid {risk_border_color};
                            margin: 10px 0;
                        ">
                            <p style="margin: 0;"><strong>Mức độ rủi ro:</strong> <span style="color: {risk_color}; font-weight: bold; font-size: 1.1em;">{risk_level}</span></p>
                            <p style="margin: 10px 0 0 0; font-size: 0.9em;"><strong>Biến động lịch sử:</strong> {summary['historical_volatility']:.2f}%</p>
                            <p style="margin: 10px 0 0 0; font-size: 0.9em;"><strong>Lưu ý:</strong> Dự báo dựa trên dữ liệu lịch sử và mô hình toán học. 
                            Kết quả thực tế có thể khác biệt đáng kể do các yếu tố không lường trước được.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    show_popup_message(f"Đã tạo dự báo thành công cho {selected_symbol}", "success")
                else:
                    show_popup_message("Không thể tạo tóm tắt dự báo", "warning")
    
    elif demo_option == "Tải File CSV":
        st.markdown('<div class="section-header">📁 Tải File CSV</div>', unsafe_allow_html=True)
        
        st.info("Tải file CSV với dữ liệu chứng khoán. Hỗ trợ nhiều định dạng bao gồm định dạng VN30.")
        
        with st.expander("📋 Các Định Dạng CSV Được Hỗ Trợ", expanded=False):
            st.write("**Định dạng 1 - Định dạng chuẩn:**")
            st.write("Cột: date, open, high, low, close, volume, turnover")
            st.write("Tên file: {MÃ_CHỨNG_KHOÁN}_{gì_đó}.csv")
            
            st.write("\n**Định dạng 2 - Định dạng VN30:**")
            st.write("Cột: Date;Close;Open;High;Low;Volumn;% turnover")
            st.write("Sử dụng dấu chấm phẩy làm phân cách và dấu phẩy làm phân cách thập phân")
            
            st.write("\n**Định dạng 3 - Định dạng quốc tế:**")
            st.write("Định dạng phân cách bằng dấu phẩy chuẩn với dấu chấm làm phân cách thập phân")
        
        uploaded_files = st.file_uploader(
            "Chọn file CSV",
            accept_multiple_files=True,
            type=['csv'],
            help="Bạn có thể tải lên nhiều file CSV. Hệ thống sẽ tự động phát hiện định dạng."
        )
        
        if uploaded_files:
            show_popup_message(f"Đã tải lên {len(uploaded_files)} file", "success")
            
            # Show uploaded files info
            with st.expander("📄 Chi Tiết File Đã Tải Lên"):
                for i, uploaded_file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
            
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
            
            show_popup_message(f"File đã được lưu vào thư mục tạm: {temp_dir}", "info")
            
            # Information about next steps
            if not st.session_state.get('upload_processed', False):
                st.info("👇 **Bước tiếp theo:** Nhấp vào nút bên dưới để xử lý dữ liệu và kích hoạt tính năng dự báo AI")
            
            # Add Process button with a unique key
            process_button = st.button(
                "🔄 Xử Lý Dữ Liệu Đã Tải Lên", 
                type="primary", 
                key="process_uploaded_data_button",
                help="Nhấp để xử lý file CSV đã tải lên và tạo phân tích"
            )
            
            if process_button:
                with st.spinner("Đang xử lý file CSV đã tải lên..."):
                    try:
                        # Step 1: Load and process data
                        st.info("Bước 1: Đang tải và xử lý file CSV...")
                        preprocessor = DataPreprocessor()
                        merged_data = preprocessor.load_and_process_all(temp_dir)
                        
                        if merged_data.empty:
                            show_popup_message("Không thể xử lý dữ liệu từ file đã tải lên. Vui lòng kiểm tra định dạng file.", "error")
                            st.stop()
                        
                        show_popup_message(f"Đã xử lý thành công dữ liệu! Kích thước: {merged_data.shape}", "success")
                        st.info(f"Các cột có sẵn: {list(merged_data.columns)}")
                        
                        # Step 2: Calculate time duration
                        st.info("Bước 2: Đang tính toán khoảng thời gian...")
                        uploaded_time_duration = calculate_time_duration(merged_data)
                        show_popup_message(f"Khoảng thời gian: {uploaded_time_duration}", "success")
                        
                        # Validate required columns
                        required_columns = ['close']
                        missing_columns = [col for col in required_columns if col not in merged_data.columns]
                        
                        if missing_columns:
                            show_popup_message(f"Thiếu các cột bắt buộc: {missing_columns}", "error")
                            st.error("File CSV của bạn phải chứa ít nhất một cột giá 'close'.")
                            st.stop()
                        
                        # Store in session state for persistence
                        st.session_state['upload_processed'] = True
                        st.session_state['uploaded_data'] = merged_data
                        st.session_state['uploaded_time_duration'] = uploaded_time_duration
                        
                        # Success message about AI prediction availability
                        st.success("✅ **Dữ liệu đã được xử lý thành công!** 🤖 Tính năng dự báo AI hiện đã sẵn sàng trong thanh bên.")
                        
                        # Display basic info for uploaded data
                        st.markdown('<div class="section-header">📊 Phân Tích Dữ Liệu Đã Tải Lên</div>', unsafe_allow_html=True)
                        
                        try:
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Thời Gian", uploaded_time_duration.split(' (')[0])
                            with col2:
                                st.metric("Tổng Bản Ghi", len(merged_data))
                            with col3:
                                if 'close' in merged_data.columns:
                                    st.metric("Giá Mới Nhất", f"{merged_data['close'].iloc[-1]:.2f}")
                                else:
                                    st.metric("Giá Mới Nhất", "N/A")
                            with col4:
                                if 'return' in merged_data.columns:
                                    latest_return = merged_data['return'].iloc[-1] if not pd.isna(merged_data['return'].iloc[-1]) else 0
                                    st.metric("Thay Đổi Mới Nhất %", f"{latest_return:.2f}%")
                                else:
                                    st.metric("Thay Đổi Mới Nhất %", "N/A")
                            with col5:
                                if 'target' in merged_data.columns:
                                    up_days = (merged_data['target'] == 1).sum()
                                    st.metric("% Ngày Tăng", f"{100 * up_days / len(merged_data):.1f}%")
                                else:
                                    st.metric("% Ngày Tăng", "N/A")
                        except Exception as e:
                            st.error(f"⚠️ Lỗi hiển thị số liệu: {str(e)}")
                        
                        # Show data preview
                        try:
                            with st.expander("📊 Phân Tích Dữ Liệu Của Bạn", expanded=True):
                                st.write("**Xem trước dữ liệu đã xử lý (10 dòng đầu):**")
                                display_data = merged_data.head(10).copy()
                                if 'date' in display_data.columns:
                                    display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
                                st.dataframe(display_data, use_container_width=True)
                        except Exception as e:
                            st.error(f"⚠️ Lỗi hiển thị xem trước dữ liệu: {str(e)}")
                        
                        # Add technical indicators
                        st.markdown('<div class="section-header">🔧 Phân Tích Kỹ Thuật</div>', unsafe_allow_html=True)
                        
                        # Validate data has required columns for technical indicators
                        required_ta_columns = ['close', 'code']
                        missing_ta_columns = [col for col in required_ta_columns if col not in merged_data.columns]
                        
                        if missing_ta_columns:
                            show_popup_message(f"Không thể tính toán chỉ báo kỹ thuật. Thiếu cột: {missing_ta_columns}", "warning")
                            st.warning("Sử dụng dữ liệu gốc không có chỉ báo kỹ thuật.")
                            data_with_features = merged_data.copy()
                        else:
                            try:
                                with st.spinner("Đang thêm chỉ báo kỹ thuật..."):
                                    data_with_features = add_technical_indicators(merged_data)
                                
                                show_popup_message(f"Đã thêm chỉ báo kỹ thuật! Kích thước cuối: {data_with_features.shape}", "success")
                                
                                # Debug: Show what indicators were actually added
                                with st.expander("🔍 Chi Tiết Chỉ Báo Kỹ Thuật"):
                                    original_cols = set(merged_data.columns)
                                    new_cols = set(data_with_features.columns) - original_cols
                                    
                                    if new_cols:
                                        ma_indicators = [col for col in new_cols if 'ma_' in col.lower()]
                                        rsi_indicators = [col for col in new_cols if 'rsi' in col.lower()]
                                        bb_indicators = [col for col in new_cols if 'bb_' in col.lower()]
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.write(f"**Chỉ Báo MA:** {ma_indicators}")
                                        with col2:
                                            st.write(f"**Chỉ Báo RSI:** {rsi_indicators}")
                                        with col3:
                                            st.write(f"**Dải Bollinger:** {bb_indicators}")
                                    else:
                                        st.warning("Không có chỉ báo kỹ thuật nào được tính toán thành công")
                            except Exception as e:
                                show_popup_message(f"Lỗi tính toán chỉ báo kỹ thuật: {str(e)}", "error")
                                st.warning("Sử dụng dữ liệu gốc không có chỉ báo kỹ thuật.")
                                data_with_features = merged_data.copy()
                        
                        # Charts
                        st.markdown('<div class="section-header">📈 Biểu Đồ & Hình Ảnh Hóa</div>', unsafe_allow_html=True)
                        
                        # Price chart - with validation
                        try:
                            st.write("**Biểu Đồ Giá với Chỉ Báo Kỹ Thuật:**")
                            price_fig = plot_price_chart(data_with_features)
                            st.plotly_chart(price_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"⚠️ Lỗi tạo biểu đồ giá: {str(e)}")
                            st.warning("Bỏ qua hiển thị biểu đồ giá.")
                        
                        # RSI chart - with validation
                        try:
                            st.write("**Chỉ Báo Kỹ Thuật RSI:**")
                            rsi_fig = plot_technical_indicators(data_with_features)
                            st.plotly_chart(rsi_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"⚠️ Lỗi tạo biểu đồ RSI: {str(e)}")
                            st.warning("Bỏ qua hiển thị biểu đồ RSI.")
                        
                        # Store processed data in session state for AI prediction
                        try:
                            st.session_state['uploaded_data'] = merged_data
                            st.session_state['uploaded_features'] = data_with_features
                            st.session_state['uploaded_time_duration'] = uploaded_time_duration
                            show_popup_message("Dữ liệu đã được lưu trong phiên làm việc để dự báo AI", "success")
                        except Exception as e:
                            st.error(f"⚠️ Lỗi lưu trữ dữ liệu trong phiên làm việc: {str(e)}")
                        
                        # Advanced Feature Engineering
                        st.markdown('<div class="section-header">🧠 Kỹ Thuật Xây Dựng Đặc Trưng Nâng Cao</div>', unsafe_allow_html=True)
                        
                        try:
                            feature_engineer = FeatureEngineer()
                            
                            with st.spinner("Đang tạo các đặc trưng bổ sung..."):
                                # Add more features using the uploaded data
                                enriched_data = feature_engineer.create_price_features(data_with_features)
                                enriched_data = feature_engineer.create_volume_features(enriched_data)
                                enriched_data = feature_engineer.create_lag_features(enriched_data)
                                enriched_data = feature_engineer.create_rolling_features(enriched_data)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Đặc Trưng Gốc", len(merged_data.columns))
                            with col2:
                                st.metric("Đặc Trưng Cuối Cùng", len(enriched_data.columns))
                        except Exception as e:
                            st.error(f"⚠️ Lỗi trong kỹ thuật xây dựng đặc trưng: {str(e)}")
                            st.warning("Sử dụng dữ liệu chỉ với chỉ báo kỹ thuật.")
                            enriched_data = data_with_features.copy()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Đặc Trưng Gốc", len(merged_data.columns))
                            with col2:
                                st.metric("Đặc Trưng Cuối Cùng", len(enriched_data.columns))
                        
                        # Show final dataset preview
                        st.markdown('<div class="section-header">📋 Xem Trước Bộ Dữ Liệu Cuối Cùng</div>', unsafe_allow_html=True)
                        
                        st.write("**Xem trước bộ dữ liệu đã được enrich cuối cùng (Dữ Liệu Đã Tải Lên Của Bạn):**")
                        st.dataframe(enriched_data.head(), use_container_width=True)
                        
                        # Model Training Demo
                        st.markdown('<div class="section-header">🤖 Demo Huấn Luyện Mô Hình</div>', unsafe_allow_html=True)
                        
                        if st.button("🚀 Chạy Demo Huấn Luyện Mô Hình", type="primary", key="model_training_uploaded"):
                            with st.spinner("Đang chuẩn bị dữ liệu cho mô hình..."):
                                # Prepare data for modeling
                                # Show data info before cleaning
                                st.info(f"📊 Dữ liệu trước khi làm sạch: {enriched_data.shape}")
                                st.info(f"📊 Giá trị NaN: {enriched_data.isnull().sum().sum()}")
                                
                                # Remove rows with NaN values, but be more flexible
                                clean_data = enriched_data.dropna()
                                
                                st.info(f"📊 Dữ liệu sau khi làm sạch: {clean_data.shape}")
                                
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
                                    
                                    st.info(f"📊 Đã chọn {len(numeric_features)} đặc trưng số cho mô hình")
                                    
                                    # Simple train/test split
                                    split_idx = max(1, int(0.8 * len(clean_data)))  # Ensure at least 1 sample for test
                                    X_train, X_test = X[:split_idx], X[split_idx:]
                                    y_train, y_test = y[:split_idx], y[split_idx:]
                                    
                                    show_popup_message(f"Dữ liệu đã được chuẩn bị: {len(X_train)} mẫu huấn luyện, {len(X_test)} mẫu kiểm tra", "success")
                                    
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
                                        title='Mức Độ Quan Trọng Đặc Trưng Mô Phỏng cho Dữ Liệu Của Bạn'
                                    )
                                    st.plotly_chart(fig_importance, use_container_width=True)
                                    
                                    # Simulated model results
                                    st.write("**Hiệu Suất Mô Hình Mô Phỏng trên Dữ Liệu Của Bạn:**")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Độ Chính Xác", "78.5%")
                                    with col2:
                                        st.metric("Độ Chính Xác", "76.2%")
                                    with col3:
                                        st.metric("Độ Nhạy", "79.8%")
                                    
                                else:
                                    st.error(f"Không đủ dữ liệu sạch để mô hình hóa. Cần ít nhất {min_required_samples} mẫu, nhưng chỉ có {len(clean_data)} sau khi làm sạch.")
                                    st.error("Hãy thử tải lên nhiều dữ liệu hơn hoặc kiểm tra chất lượng dữ liệu.")
                        
                    except Exception as e:
                        st.error(f"Lỗi xử lý file: {str(e)}")
        
        # AI Prediction for uploaded data - moved outside process_button block
        if uploaded_files and st.session_state.get('upload_processed', False):
            st.markdown('<div class="section-header">🤖 Phân Tích AI cho Dữ Liệu Của Bạn</div>', unsafe_allow_html=True)
            
            # Check if sidebar AI prediction button was pressed and we have uploaded data
            if use_ai_prediction and st.session_state.get('upload_processed', False):
                with st.spinner("AI đang phân tích dữ liệu thị trường của bạn..."):
                    try:
                        # Validate that we have the required data
                        if 'uploaded_data' not in st.session_state or st.session_state['uploaded_data'].empty:
                            show_popup_message("Không tìm thấy dữ liệu đã tải lên. Vui lòng xử lý dữ liệu của bạn trước.", "error")
                            st.stop()
                        
                        # Get processed data from session state
                        processed_data = st.session_state['uploaded_data']
                        
                        # Validate required columns
                        required_columns = ['close']
                        missing_columns = [col for col in required_columns if col not in processed_data.columns]
                        
                        if missing_columns:
                            show_popup_message(f"Thiếu các cột bắt buộc: {missing_columns}", "error")
                            st.error("Vui lòng đảm bảo file CSV của bạn chứa ít nhất một cột giá 'close'.")
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
                        st.markdown("### 🔮 Phân Tích AI Dữ Liệu Thị Trường Của Bạn")
                        
                        if "Lỗi" not in ai_prediction_uploaded:
                            # Create an enhanced display for uploaded data analysis
                            with st.container():
                                show_popup_message("Phân tích dữ liệu của bạn hoàn tất!", "success")
                                
                                # Show key metrics in columns
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Điểm Dữ Liệu", f"{uploaded_summary['total_days']:,}")
                                with col2:
                                    if uploaded_summary['current_price'] > 0:
                                        st.metric("Giá Hiện Tại", f"{uploaded_summary['current_price']:.2f}")
                                with col3:
                                    st.metric("Thay Đổi Mới Nhất", f"{uploaded_summary['latest_change']:.2f}%")
                                with col4:
                                    st.metric("Tỷ Lệ Ngày Tăng", f"{uploaded_summary['up_days_ratio']:.1f}%")
                                
                                st.markdown("---")
                                
                                # Display the AI prediction in a styled container with black background
                                st.markdown("#### 📊 Phân Tích Chi Tiết Dữ Liệu Của Bạn:")
                                
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
                                st.markdown("#### 🔮 Dự Báo Mở Rộng (10 Năm Tiếp Theo)")
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
                        st.error(f"Lỗi xử lý dữ liệu đã tải lên cho dự báo AI: {str(e)}")
                        st.error("Vui lòng đảm bảo dữ liệu của bạn đã được xử lý thành công trước khi yêu cầu dự báo AI.")
            
            # Info message when no AI prediction button is pressed yet
            else:
                st.info("💡 Nhấp vào nút 'Nhận Dự Báo Thị Trường AI' trong thanh bên để phân tích dữ liệu đã tải lên với AI")
        
        # If no files uploaded yet, show info
        else:
            st.info("📁 Vui lòng tải lên file CSV để bắt đầu phân tích")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>📊 Hệ Thống Dự Báo Thị Trường Chứng Khoán</p>
            <p>Được phát triển với Machine Learning và AI</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
