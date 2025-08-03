"""
Test and demonstration script for the Advanced Stock Forecaster
This script shows how to use the AdvancedStockForecaster with different models
and integrate it with the existing application structure.
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from stock_predictor.data.preprocessor import DataPreprocessor
from stock_predictor.forecast.advanced_forecaster import AdvancedStockForecaster

warnings.filterwarnings('ignore')

def test_advanced_forecaster():
    """Test the advanced forecaster with VN30 data."""
    
    print("🚀 Testing Advanced Stock Forecaster")
    print("=" * 50)
    
    # Initialize components
    preprocessor = DataPreprocessor()
    
    # Load and process VN30 data
    vn30_file_path = "/Users/dungnhi/Documents/HTRaQuyetDinh/VN30_demo.csv"
    
    try:
        print("📊 Loading VN30 data...")
        vn30_data = preprocessor._read_csv_flexible(vn30_file_path)
        
        if vn30_data is None:
            raise Exception("Could not read VN30 CSV file")
        
        print(f"✅ Loaded VN30 data: {vn30_data.shape}")
        
        # Process the data
        print("🔄 Processing VN30 data...")
        processed_data = preprocessor._normalize_data_format(vn30_data, "VN30")
        
        if processed_data is None or processed_data.empty:
            raise Exception("Could not normalize VN30 data format")
        
        # Calculate returns and targets
        processed_data = preprocessor._calculate_returns_and_targets(processed_data)
        print(f"✅ Processed VN30 data: {processed_data.shape}")
        
        # Test different models
        models_to_test = ['neural_network', 'random_forest', 'xgboost', 'svr']
        
        results_summary = {}
        
        for model_type in models_to_test:
            print(f"\n🤖 Testing {model_type.upper()} model...")
            
            try:
                # Initialize forecaster
                forecaster = AdvancedStockForecaster(model_type=model_type)
                
                # Add technical indicators
                print("📈 Adding technical indicators...")
                enriched_data = forecaster.add_technical_indicators(processed_data)
                
                if enriched_data.empty:
                    print(f"❌ No data after adding indicators for {model_type}")
                    continue
                
                print(f"✅ Added indicators: {enriched_data.shape}")
                
                # Train model
                print(f"🏋️ Training {model_type} model...")
                training_results = forecaster.train_model(enriched_data, stock_code="VN30")
                
                print(f"✅ Training completed!")
                print(f"   📊 Test Accuracy: {training_results['test_accuracy']:.4f}")
                print(f"   📊 Validation Accuracy: {training_results['val_accuracy']:.4f}")
                
                # Make predictions
                print("🔮 Making predictions...")
                recent_data = enriched_data.tail(100)  # Use last 100 days
                predictions = forecaster.predict(recent_data, stock_code="VN30")
                
                # Calculate prediction accuracy on recent data
                actual = recent_data['target'].values
                prediction_accuracy = np.mean(predictions == actual)
                
                print(f"✅ Prediction accuracy on recent data: {prediction_accuracy:.4f}")
                
                # Store results
                results_summary[model_type] = {
                    'test_accuracy': training_results['test_accuracy'],
                    'val_accuracy': training_results['val_accuracy'],
                    'prediction_accuracy': prediction_accuracy,
                    'data_shape': enriched_data.shape,
                    'features_count': len(forecaster.feature_columns)
                }
                
                # Get model summary
                model_summary = forecaster.get_model_summary(stock_code="VN30")
                print(f"📋 Model Summary: {model_summary['feature_count']} features")
                
                print(f"✅ {model_type.upper()} model test completed successfully!")
                
            except Exception as e:
                print(f"❌ Error testing {model_type}: {str(e)}")
                results_summary[model_type] = {'error': str(e)}
                
        # Print final results summary
        print("\n" + "=" * 50)
        print("📊 FINAL RESULTS SUMMARY")
        print("=" * 50)
        
        for model_type, results in results_summary.items():
            if 'error' in results:
                print(f"{model_type.upper()}: ❌ Error - {results['error']}")
            else:
                print(f"{model_type.upper()}:")
                print(f"  📊 Test Accuracy: {results['test_accuracy']:.4f}")
                print(f"  📊 Val Accuracy: {results['val_accuracy']:.4f}")
                print(f"  🔮 Prediction Accuracy: {results['prediction_accuracy']:.4f}")
                print(f"  📈 Data Shape: {results['data_shape']}")
                print(f"  🎯 Features: {results['features_count']}")
                print()
        
        # Find best model
        best_model = None
        best_accuracy = 0
        
        for model_type, results in results_summary.items():
            if 'test_accuracy' in results and results['test_accuracy'] > best_accuracy:
                best_accuracy = results['test_accuracy']
                best_model = model_type
        
        if best_model:
            print(f"🏆 Best Model: {best_model.upper()} with accuracy {best_accuracy:.4f}")
        
        return results_summary
        
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        return None

def demonstrate_integration():
    """Demonstrate how to integrate the advanced forecaster with the existing app."""
    
    print("\n" + "=" * 50)
    print("🔗 INTEGRATION DEMONSTRATION")
    print("=" * 50)
    
    print("Here's how to integrate the AdvancedStockForecaster into the existing Streamlit app:")
    print()
    
    integration_code = '''
# In app.py, add this import:
from forecast.advanced_forecaster import AdvancedStockForecaster

# Add this in the sidebar for model selection:
st.sidebar.markdown("### 🤖 Advanced ML Models")
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["neural_network", "random_forest", "xgboost", "svr", "lstm"]
)

# Add this in the main analysis section:
if st.button("🚀 Train Advanced Model"):
    with st.spinner(f"Training {model_type} model..."):
        # Initialize forecaster
        forecaster = AdvancedStockForecaster(model_type=model_type)
        
        # Add technical indicators
        enriched_data = forecaster.add_technical_indicators(sample_data)
        
        # Train model
        results = forecaster.train_model(enriched_data)
        
        # Display results
        st.success(f"Model trained! Accuracy: {results['test_accuracy']:.2%}")
        
        # Make predictions
        predictions = forecaster.predict(enriched_data.tail(30))
        
        # Create and display chart
        fig = forecaster.create_prediction_chart(
            enriched_data.tail(30), 
            predictions, 
            "VN30"
        )
        st.plotly_chart(fig, use_container_width=True)
    '''
    
    print(integration_code)
    
    print("\n📋 Key Integration Points:")
    print("1. Add model selection in sidebar")
    print("2. Integrate with existing data processing pipeline")
    print("3. Display training results and predictions")
    print("4. Show interactive charts with Plotly")
    print("5. Handle errors gracefully with user feedback")

def create_integration_example():
    """Create a simple integration example file."""
    
    integration_file = '''"""
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
'''
    
    with open("advanced_model_integration_example.py", "w", encoding="utf-8") as f:
        f.write(integration_file)
    
    print("📄 Created integration example file: advanced_model_integration_example.py")

def main():
    """Main test function."""
    
    print("🎯 Advanced Stock Forecaster Testing Suite")
    print("=" * 60)
    
    # Test the advanced forecaster
    results = test_advanced_forecaster()
    
    # Demonstrate integration
    demonstrate_integration()
    
    # Create integration example
    create_integration_example()
    
    print("\n" + "=" * 60)
    print("✅ Testing completed! Check the results above.")
    print("📁 Integration example created: advanced_model_integration_example.py")
    print("🔗 You can now integrate these models into your Streamlit app.")

if __name__ == "__main__":
    main()
