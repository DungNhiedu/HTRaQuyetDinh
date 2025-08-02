"""
Streamlit Web Application cho Stock Market Prediction Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_predictor.main import StockPredictor
from stock_predictor.data.collector import DataCollector
from stock_predictor.evaluation.visualization import Visualizer
from stock_predictor.utils.helpers import setup_logging

# Configure page
st.set_page_config(
    page_title="Stock Market Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
setup_logging(level='INFO')

@st.cache_data
def load_stock_data(symbol: str, period: str):
    """Load stock data with caching"""
    try:
        collector = DataCollector()
        data = collector.fetch_stock_data(symbol=symbol, period=period)
        return data, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def initialize_predictor(symbol: str):
    """Initialize predictor with caching"""
    return StockPredictor(symbol=symbol)

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("📈 Market Prediction System")
    st.markdown("### Fusion of Machine Learning Techniques")
    st.markdown("Dự báo chỉ số thị trường chứng khoán sử dụng kết hợp nhiều kỹ thuật Machine Learning")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Symbol selection
    symbol_options = {
        'VN-Index (Vietnam)': '^VNI',
        'S&P 500 (US)': '^GSPC',
        'Dow Jones (US)': '^DJI',
        'NASDAQ (US)': '^IXIC',
        'Nikkei 225 (Japan)': '^N225',
        'FTSE 100 (UK)': '^FTSE',
        'Custom': 'custom'
    }
    
    selected_index = st.sidebar.selectbox("Chọn chỉ số thị trường:", list(symbol_options.keys()))
    
    if symbol_options[selected_index] == 'custom':
        symbol = st.sidebar.text_input("Nhập mã chứng khoán:", value="^VNI")
    else:
        symbol = symbol_options[selected_index]
        
    # Period selection
    period = st.sidebar.selectbox(
        "Thời gian dữ liệu:",
        ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=4  # Default to 2y
    )
    
    # Model selection
    st.sidebar.header("🤖 Model Configuration")
    
    available_models = ['Random Forest', 'XGBoost', 'LightGBM', 'SVR', 'LSTM', 'ARIMA']
    selected_models = st.sidebar.multiselect(
        "Chọn models để train:",
        available_models,
        default=['Random Forest', 'XGBoost', 'LSTM']
    )
    
    ensemble_method = st.sidebar.selectbox(
        "Phương pháp Ensemble:",
        ['voting', 'stacking', 'weighted', 'bayesian'],
        index=0
    )
    
    # Prediction settings
    st.sidebar.header("🔮 Prediction Settings")
    predict_days = st.sidebar.slider("Số ngày dự báo:", min_value=1, max_value=90, value=30)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🏋️ Model Training", 
        "📈 Predictions", 
        "📉 Model Comparison",
        "📋 Report"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("📊 Stock Data Overview")
        
        # Load data
        with st.spinner(f"Loading data for {symbol}..."):
            data, error = load_stock_data(symbol, period)
            
        if error:
            st.error(f"Error loading data: {error}")
            st.stop()
            
        if data is None or data.empty:
            st.error("No data available")
            st.stop()
            
        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Records", len(data))
        with col3:
            st.metric("Latest Price", f"{data['Close'].iloc[-1]:.2f}")
        with col4:
            price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100)
            st.metric("Daily Change (%)", f"{price_change:.2f}%")
            
        # Price chart
        st.subheader("📈 Price Chart")
        
        visualizer = Visualizer()
        price_fig = visualizer.plot_stock_data(data, title=f"{symbol} Stock Price", show_volume=True)
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Data statistics
        st.subheader("📊 Data Statistics")
        st.dataframe(data.describe())
        
        # Recent data
        st.subheader("📅 Recent Data")
        st.dataframe(data.tail(10))
        
    # Tab 2: Model Training
    with tab2:
        st.header("🏋️ Model Training")
        
        if st.button("🚀 Start Training", type="primary"):
            # Convert model names to codes
            model_mapping = {
                'Random Forest': 'rf',
                'XGBoost': 'xgb', 
                'LightGBM': 'lgb',
                'SVR': 'svr',
                'LSTM': 'lstm',
                'ARIMA': 'arima'
            }
            
            model_codes = [model_mapping[model] for model in selected_models if model in model_mapping]
            
            if not model_codes:
                st.error("Please select at least one model")
                st.stop()
                
            # Initialize predictor
            predictor = initialize_predictor(symbol)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Load data
                status_text.text("Loading and preparing data...")
                progress_bar.progress(10)
                
                predictor.raw_data = data
                predictor.prepare_features()
                predictor.split_data()
                
                # Step 2: Train models
                status_text.text("Training individual models...")
                progress_bar.progress(30)
                
                training_results = {}
                for i, model_code in enumerate(model_codes):
                    status_text.text(f"Training {model_code.upper()} model...")
                    progress = 30 + (i + 1) * 40 / len(model_codes)
                    progress_bar.progress(int(progress))
                    
                    try:
                        result = predictor.train_individual_models([model_code])
                        training_results.update(result)
                    except Exception as e:
                        st.warning(f"Failed to train {model_code}: {str(e)}")
                        
                # Step 3: Train ensemble
                if len(predictor.models) > 1:
                    status_text.text("Training ensemble...")
                    progress_bar.progress(80)
                    
                    try:
                        ensemble_result = predictor.train_ensemble(ensemble_method)
                        training_results['ensemble'] = ensemble_result
                    except Exception as e:
                        st.warning(f"Failed to train ensemble: {str(e)}")
                        
                # Step 4: Evaluate
                status_text.text("Evaluating models...")
                progress_bar.progress(90)
                
                evaluation_results = predictor.evaluate_models()
                
                # Store in session state
                st.session_state.predictor = predictor
                st.session_state.training_results = training_results
                st.session_state.evaluation_results = evaluation_results
                
                progress_bar.progress(100)
                status_text.text("Training completed!")
                
                st.success("🎉 Training completed successfully!")
                
                # Display results
                st.subheader("📊 Training Results")
                
                results_data = []
                for model_name, results in evaluation_results.items():
                    if 'regression_metrics' in results:
                        metrics = results['regression_metrics']
                        results_data.append({
                            'Model': model_name,
                            'MAE': metrics.get('mae', 'N/A'),
                            'RMSE': metrics.get('rmse', 'N/A'), 
                            'R²': metrics.get('r2', 'N/A'),
                            'Directional Accuracy': metrics.get('directional_accuracy', 'N/A')
                        })
                        
                if results_data:
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Best model
                    best_mae_idx = results_df['MAE'].astype(float).idxmin()
                    best_model = results_df.loc[best_mae_idx, 'Model']
                    st.info(f"🏆 Best Model (by MAE): {best_model}")
                    
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                
    # Tab 3: Predictions
    with tab3:
        st.header("🔮 Future Predictions")
        
        if 'predictor' not in st.session_state:
            st.warning("Please train models first in the 'Model Training' tab")
        else:
            predictor = st.session_state.predictor
            
            if st.button("📈 Generate Predictions"):
                with st.spinner(f"Generating {predict_days} days predictions..."):
                    try:
                        future_predictions = predictor.predict_future(days=predict_days)
                        
                        if future_predictions:
                            # Store predictions
                            st.session_state.future_predictions = future_predictions
                            
                            # Create prediction dates
                            last_date = predictor.processed_data.index[-1]
                            future_dates = pd.date_range(
                                start=last_date + pd.Timedelta(days=1),
                                periods=predict_days,
                                freq='D'
                            )
                            
                            # Plot predictions
                            st.subheader("📊 Future Predictions Chart")
                            
                            fig = go.Figure()
                            
                            # Historical data
                            recent_data = predictor.processed_data['Close'].tail(60)
                            fig.add_trace(
                                go.Scatter(
                                    x=recent_data.index,
                                    y=recent_data.values,
                                    mode='lines',
                                    name='Historical',
                                    line=dict(color='blue', width=2)
                                )
                            )
                            
                            # Future predictions
                            colors = px.colors.qualitative.Set1
                            for i, (model_name, predictions) in enumerate(future_predictions.items()):
                                color = colors[i % len(colors)]
                                fig.add_trace(
                                    go.Scatter(
                                        x=future_dates,
                                        y=predictions,
                                        mode='lines',
                                        name=f'{model_name}',
                                        line=dict(color=color, width=2, dash='dash')
                                    )
                                )
                                
                            fig.update_layout(
                                title=f"{symbol} - Future Predictions ({predict_days} days)",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                height=600,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Predictions table
                            st.subheader("📋 Predictions Summary")
                            
                            pred_summary = []
                            for model_name, predictions in future_predictions.items():
                                pred_summary.append({
                                    'Model': model_name,
                                    'Average Prediction': f"{predictions.mean():.2f}",
                                    'Min Prediction': f"{predictions.min():.2f}",
                                    'Max Prediction': f"{predictions.max():.2f}",
                                    'Std Deviation': f"{predictions.std():.2f}"
                                })
                                
                            pred_df = pd.DataFrame(pred_summary)
                            st.dataframe(pred_df, use_container_width=True)
                            
                            # Download predictions
                            predictions_df = pd.DataFrame(future_predictions, index=future_dates)
                            csv = predictions_df.to_csv()
                            st.download_button(
                                label="📥 Download Predictions CSV",
                                data=csv,
                                file_name=f"{symbol}_predictions_{predict_days}days.csv",
                                mime="text/csv"
                            )
                            
                        else:
                            st.error("No predictions generated")
                            
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                        
    # Tab 4: Model Comparison
    with tab4:
        st.header("📉 Model Comparison")
        
        if 'evaluation_results' not in st.session_state:
            st.warning("Please train models first in the 'Model Training' tab")
        else:
            evaluation_results = st.session_state.evaluation_results
            predictor = st.session_state.predictor
            
            # Performance comparison chart
            st.subheader("📊 Model Performance Comparison")
            
            comparison_df = predictor.evaluator.compare_models(evaluation_results)
            
            if not comparison_df.empty:
                # MAE comparison
                fig_mae = px.bar(
                    comparison_df, 
                    x='Model', 
                    y='MAE',
                    title="Model Comparison - Mean Absolute Error (Lower is Better)",
                    color='MAE',
                    color_continuous_scale='Viridis_r'
                )
                fig_mae.update_layout(height=400)
                st.plotly_chart(fig_mae, use_container_width=True)
                
                # R² comparison
                fig_r2 = px.bar(
                    comparison_df, 
                    x='Model', 
                    y='R²',
                    title="Model Comparison - R² Score (Higher is Better)",
                    color='R²',
                    color_continuous_scale='Viridis'
                )
                fig_r2.update_layout(height=400)
                st.plotly_chart(fig_r2, use_container_width=True)
                
                # Detailed comparison table
                st.subheader("📋 Detailed Performance Metrics")
                st.dataframe(comparison_df, use_container_width=True)
                
                # Model predictions comparison (on test set)
                if hasattr(predictor, 'predictions') and predictor.predictions:
                    st.subheader("📈 Test Set Predictions Comparison")
                    
                    test_dates = predictor.X_test.index if hasattr(predictor.X_test, 'index') else range(len(predictor.y_test))
                    
                    pred_fig = visualizer.plot_multiple_predictions(
                        y_true=predictor.y_test.values,
                        predictions_dict=predictor.predictions,
                        dates=test_dates,
                        title="Test Set Predictions Comparison"
                    )
                    st.plotly_chart(pred_fig, use_container_width=True)
                    
            else:
                st.error("No evaluation results available")
                
    # Tab 5: Report
    with tab5:
        st.header("📋 Comprehensive Report")
        
        if 'predictor' not in st.session_state:
            st.warning("Please train models first to generate report")
        else:
            predictor = st.session_state.predictor
            
            # Generate report
            report = predictor.create_report()
            
            # Display report
            st.subheader("📄 Model Performance Report")
            st.text(report)
            
            # Key insights
            st.subheader("💡 Key Insights")
            
            if hasattr(predictor, 'evaluation_results') and predictor.evaluation_results:
                best_model_name, best_score = predictor.evaluator.get_best_model(
                    predictor.evaluation_results, metric='mae'
                )
                
                insights = [
                    f"🏆 **Best Performing Model**: {best_model_name} (MAE: {best_score:.4f})",
                    f"📊 **Total Models Trained**: {len(predictor.models)}",
                    f"📈 **Data Period**: {period}",
                    f"🔢 **Features Used**: {len(predictor.features.columns) if predictor.features is not None else 'N/A'}",
                    f"📅 **Training Samples**: {len(predictor.X_train) if predictor.X_train is not None else 'N/A'}",
                    f"🧪 **Test Samples**: {len(predictor.X_test) if predictor.X_test is not None else 'N/A'}"
                ]
                
                for insight in insights:
                    st.markdown(insight)
                    
            # Download report
            if st.button("📥 Download Full Report"):
                report_data = {
                    'symbol': symbol,
                    'period': period,
                    'models_trained': list(predictor.models.keys()) if hasattr(predictor, 'models') else [],
                    'evaluation_results': predictor.evaluation_results if hasattr(predictor, 'evaluation_results') else {},
                    'report_text': report
                }
                
                import json
                report_json = json.dumps(report_data, indent=2, default=str)
                
                st.download_button(
                    label="📄 Download JSON Report",
                    data=report_json,
                    file_name=f"{symbol}_prediction_report.json",
                    mime="application/json"
                )
                
    # Footer
    st.markdown("---")
    st.markdown("### 📚 About This System")
    st.markdown("""
    **Stock Market Index Prediction using Fusion of Machine Learning Techniques**
    
    Hệ thống này sử dụng kết hợp nhiều kỹ thuật Machine Learning bao gồm:
    
    🤖 **Traditional ML Models**:
    - Random Forest Regression
    - XGBoost (Extreme Gradient Boosting)
    - LightGBM (Light Gradient Boosting Machine)
    - Support Vector Regression (SVR)
    
    🧠 **Deep Learning Models**:
    - LSTM (Long Short-Term Memory Networks)
    - GRU (Gated Recurrent Units)
    
    📈 **Time Series Models**:
    - ARIMA (AutoRegressive Integrated Moving Average)
    
    🔄 **Ensemble Methods**:
    - Voting Regressor
    - Stacking
    - Weighted Averaging
    - Bayesian Model Averaging
    
    📊 **Features**:
    - Technical Indicators (RSI, MACD, Bollinger Bands, etc.)
    - Price-based Features
    - Volume Features
    - Time-based Features
    - Lag Features
    """)
    
    st.markdown("**Tác giả**: Dương Thị Ngọc Dung - 24210015")

if __name__ == "__main__":
    main()
