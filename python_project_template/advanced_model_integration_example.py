"""
Advanced Model Integration Example for Streamlit App
Add this to your app.py to integrate the advanced forecaster
"""

import streamlit as st
from forecast.advanced_forecaster import AdvancedStockForecaster

def add_advanced_model_section(data):
    """Add advanced model section to the Streamlit app."""
    
    st.markdown("### 🤖 Advanced Machine Learning Models")
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type",
            ["neural_network", "random_forest", "xgboost", "svr"],
            help="Choose the machine learning model for prediction"
        )
    
    with col2:
        scaler_type = st.selectbox(
            "Select Scaler",
            ["minmax", "standard"],
            help="Choose data scaling method"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, help="Proportion of data for testing")
        add_indicators = st.checkbox("Add Technical Indicators", value=True)
    
    if st.button("🚀 Train Advanced Model", type="primary"):
        try:
            # Initialize forecaster
            forecaster = AdvancedStockForecaster(
                model_type=model_type,
                scaler_type=scaler_type
            )
            
            # Prepare data
            if add_indicators:
                with st.spinner("Adding technical indicators..."):
                    enriched_data = forecaster.add_technical_indicators(data)
                    st.success(f"Added indicators. Data shape: {enriched_data.shape}")
            else:
                enriched_data = data
            
            # Train model
            with st.spinner(f"Training {model_type} model..."):
                results = forecaster.train_model(
                    enriched_data,
                    test_size=test_size
                )
            
            # Display results
            st.success("✅ Model training completed!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Accuracy", f"{results['train_accuracy']:.2%}")
            with col2:
                st.metric("Validation Accuracy", f"{results['val_accuracy']:.2%}")
            with col3:
                st.metric("Test Accuracy", f"{results['test_accuracy']:.2%}")
            
            # Show feature importance if available
            if 'feature_importance' in results:
                st.subheader("📊 Feature Importance")
                importance_df = pd.DataFrame(
                    list(results['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 10 Most Important Features'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Make predictions on recent data
            recent_data = enriched_data.tail(30)
            predictions = forecaster.predict(recent_data)
            
            # Create prediction chart
            st.subheader("🔮 Recent Predictions")
            chart = forecaster.create_prediction_chart(
                recent_data,
                predictions,
                "Stock"
            )
            st.plotly_chart(chart, use_container_width=True)
            
            # Model summary
            with st.expander("📋 Model Summary"):
                summary = forecaster.get_model_summary()
                st.json(summary)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Try with different model settings or check your data quality")

# Add this function call in your main app where appropriate:
# add_advanced_model_section(your_data)
