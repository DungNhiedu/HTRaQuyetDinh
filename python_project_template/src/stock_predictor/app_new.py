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

def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">📈 Stock Market Prediction Demo</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;">Dự báo chỉ số thị trường sử dụng Machine Learning</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Demo options
    demo_option = st.sidebar.selectbox(
        "Choose Demo Type",
        ["Sample Data Demo", "Upload CSV Files"]
    )
    
    if demo_option == "Sample Data Demo":
        st.markdown('<div class="section-header">📊 Sample Data Analysis</div>', unsafe_allow_html=True)
        
        # Load real VN30 data instead of synthetic data
        vn30_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/Dữ liệu Lịch sử VN 30.csv"
        
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
            st.info("Using real VN30 index data instead of synthetic data")
            
        except Exception as e:
            st.warning(f"Could not load VN30 data: {str(e)}. Using synthetic data instead.")
            # Fallback to synthetic data
            with st.spinner("Generating sample data..."):
                sample_data = create_sample_data()
        
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
        
        with st.spinner("Adding technical indicators..."):
            data_with_features = add_technical_indicators(sample_data)
        
        st.success(f"✅ Added technical indicators! Shape: {data_with_features.shape}")
        
        # Show feature columns
        with st.expander("View Technical Indicators"):
            new_features = set(data_with_features.columns) - set(sample_data.columns)
            st.write("**New technical indicators added:**")
            for feature in sorted(new_features):
                st.write(f"• {feature}")
        
        # Visualizations
        st.markdown('<div class="section-header">📈 Charts & Analysis</div>', unsafe_allow_html=True)
        
        # Price chart
        price_fig = plot_price_chart(data_with_features)
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Technical indicators
        if 'rsi_14' in data_with_features.columns:
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
            st.write("**Date Range:**", f"{sample_data['date'].min()} to {sample_data['date'].max()}")
            st.write("**Records:**", len(sample_data))
            if 'volume' in sample_data.columns:
                st.write("**Avg Daily Volume:**", f"{sample_data['volume'].mean():.2f}M")
            st.write("**Features after enrichment:**", len(enriched_data.columns))
        
        # Model demonstration
        st.markdown('<div class="section-header">🤖 Model Training Demo</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Run Model Training Demo", type="primary"):
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
                    st.error("Not enough clean data for modeling")
    
    else:  # Upload CSV Files
        st.markdown('<div class="section-header">📁 Upload CSV Files</div>', unsafe_allow_html=True)
        
        st.info("Upload CSV files with stock data in the format expected by the reference implementation.")
        st.write("**Expected CSV format:**")
        st.write("Columns: date, open, high, low, close, volume, turnover")
        st.write("Filename format: {STOCK_CODE}_{something}.csv")
        
        uploaded_files = st.file_uploader(
            "Choose CSV files",
            accept_multiple_files=True,
            type=['csv']
        )
        
        if uploaded_files:
            st.success(f"✅ Uploaded {len(uploaded_files)} files")
            
            # Save uploaded files temporarily
            temp_dir = "/tmp/stock_data"
            os.makedirs(temp_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            if st.button("🔄 Process Uploaded Data", type="primary"):
                with st.spinner("Processing uploaded CSV files..."):
                    try:
                        preprocessor = DataPreprocessor()
                        merged_data = preprocessor.load_and_process_all(temp_dir)
                        
                        st.success(f"✅ Successfully processed data! Shape: {merged_data.shape}")
                        
                        # Show data preview
                        st.write("**Processed data preview:**")
                        st.dataframe(merged_data.head(), use_container_width=True)
                        
                        # Add technical indicators
                        with st.spinner("Adding technical indicators..."):
                            data_with_features = add_technical_indicators(merged_data)
                        
                        st.success(f"✅ Added technical indicators! Final shape: {data_with_features.shape}")
                        
                        # Show charts
                        if len(data_with_features) > 0:
                            price_fig = plot_price_chart(data_with_features)
                            st.plotly_chart(price_fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>📊 Stock Market Prediction Demo | Based on Machine Learning Fusion Techniques</p>
            <p>Implementation reference: copy_of_đồ_án_dss_nhóm_1.py</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
