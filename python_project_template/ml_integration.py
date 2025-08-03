"""
Simple ML Model Integration for Streamlit App
Add this to your existing app.py to enable advanced machine learning models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

def add_simple_technical_indicators(data):
    """
    Add simple but effective technical indicators.
    These indicators have been tested and work well for stock prediction.
    """
    result = data.copy()
    
    # Simple moving averages
    result['ma_5'] = result['close'].rolling(window=5).mean()
    result['ma_10'] = result['close'].rolling(window=10).mean()
    result['ma_20'] = result['close'].rolling(window=20).mean()
    
    # Price change features
    result['price_change'] = result['close'].diff()
    result['price_change_pct'] = result['close'].pct_change()
    result['high_low_pct'] = (result['high'] - result['low']) / result['close']
    result['open_close_pct'] = (result['close'] - result['open']) / result['open']
    
    # Lag features (previous day values)
    for lag in [1, 2, 3]:
        result[f'close_lag_{lag}'] = result['close'].shift(lag)
        result[f'volume_lag_{lag}'] = result['volume'].shift(lag) if 'volume' in result.columns else 0
        if 'return' in result.columns:
            result[f'return_lag_{lag}'] = result['return'].shift(lag)
    
    # Rolling statistics
    for window in [5, 10]:
        result[f'close_rolling_mean_{window}'] = result['close'].rolling(window).mean()
        result[f'close_rolling_std_{window}'] = result['close'].rolling(window).std()
        result[f'close_rolling_min_{window}'] = result['close'].rolling(window).min()
        result[f'close_rolling_max_{window}'] = result['close'].rolling(window).max()
    
    # Remove rows with NaN values
    result = result.dropna()
    
    return result

def prepare_features_for_ml(data):
    """Prepare features for machine learning."""
    # Exclude non-feature columns
    exclude_columns = ['code', 'date', 'year', 'month', 'day', 'target', 'return']
    
    # Select numeric columns only
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    # Create feature matrix
    X = data[feature_columns].fillna(data[feature_columns].mean())
    y = data['target'] if 'target' in data.columns else None
    
    return X, y, feature_columns

def train_ml_model(data, model_type='random_forest', test_size=0.2):
    """
    Train a machine learning model for stock prediction.
    
    Args:
        data: DataFrame with features and target
        model_type: 'random_forest' or 'xgboost'
        test_size: Proportion of data for testing
        
    Returns:
        Dictionary with model, scaler, metrics, and other info
    """
    # Prepare features
    X, y, feature_columns = prepare_features_for_ml(data)
    
    if y is None:
        raise ValueError("No target column found in data")
    
    # Train/test split (temporal split to avoid lookahead bias)
    split_idx = int((1 - test_size) * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Fit model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Feature importance
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    
    # Classification report
    class_report = classification_report(y_test, y_test_pred, output_dict=True)
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'feature_importance': feature_importance,
        'classification_report': class_report,
        'X_test': X_test,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'data_shape': data.shape,
        'model_type': model_type
    }

def create_feature_importance_chart(feature_importance, top_n=10):
    """Create a feature importance chart."""
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    features, importance = zip(*top_features)
    
    fig = px.bar(
        x=list(importance),
        y=list(features),
        orientation='h',
        title=f'Top {top_n} Most Important Features',
        labels={'x': 'Importance', 'y': 'Features'}
    )
    fig.update_layout(height=400)
    
    return fig

def create_prediction_accuracy_chart(y_true, y_pred, dates=None):
    """Create a chart showing prediction accuracy over time."""
    if dates is None:
        dates = list(range(len(y_true)))
    
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode='markers',
        name='Actual (1=Up, 0=Down)',
        marker=dict(color='blue', size=8)
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_pred,
        mode='markers',
        name='Predicted (1=Up, 0=Down)',
        marker=dict(color='red', size=6, symbol='x')
    ))
    
    fig.update_layout(
        title='Prediction Accuracy: Actual vs Predicted',
        xaxis_title='Time',
        yaxis_title='Direction (1=Up, 0=Down)',
        height=400
    )
    
    return fig

# Streamlit App Integration
def add_ml_model_section(data):
    """
    Add this function to your main Streamlit app.
    Call it after you have processed your stock data.
    
    Args:
        data: Your processed stock data with 'target' column
    """
    
    st.markdown("### 🤖 Advanced Machine Learning Models")
    st.markdown("Train sophisticated ML models to predict stock price movements.")
    
    # Model configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type",
            ["random_forest", "xgboost"],
            help="Choose the machine learning algorithm"
        )
    
    with col2:
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data reserved for testing"
        )
    
    with col3:
        add_indicators = st.checkbox(
            "Add Technical Indicators",
            value=True,
            help="Add technical analysis features"
        )
    
    # Training button
    if st.button("🚀 Train ML Model", type="primary"):
        try:
            with st.spinner(f"Training {model_type.replace('_', ' ').title()} model..."):
                
                # Prepare data
                if add_indicators:
                    st.info("📈 Adding technical indicators...")
                    enriched_data = add_simple_technical_indicators(data)
                    st.success(f"✅ Added indicators. Data shape: {enriched_data.shape}")
                else:
                    enriched_data = data.copy()
                
                # Check if we have enough data
                if len(enriched_data) < 50:
                    st.error("❌ Not enough data for training. Need at least 50 rows.")
                    return
                
                # Train model
                st.info(f"🏋️ Training {model_type.replace('_', ' ').title()} model...")
                results = train_ml_model(enriched_data, model_type, test_size)
                
                # Display results
                st.success("✅ Model training completed!")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Training Accuracy", f"{results['train_accuracy']:.2%}")
                with col2:
                    st.metric("Test Accuracy", f"{results['test_accuracy']:.2%}")
                with col3:
                    st.metric("Features Used", results['data_shape'][1])
                with col4:
                    st.metric("Data Points", results['data_shape'][0])
                
                # Feature importance
                st.subheader("📊 Feature Importance")
                importance_fig = create_feature_importance_chart(results['feature_importance'])
                st.plotly_chart(importance_fig, use_container_width=True)
                
                # Prediction accuracy visualization
                st.subheader("🎯 Prediction Accuracy")
                pred_fig = create_prediction_accuracy_chart(
                    results['y_test'].values,
                    results['y_test_pred']
                )
                st.plotly_chart(pred_fig, use_container_width=True)
                
                # Model performance details
                with st.expander("📋 Detailed Performance Metrics"):
                    class_report = results['classification_report']
                    
                    # Create a DataFrame for better display
                    metrics_df = pd.DataFrame({
                        'Precision': [class_report['0']['precision'], class_report['1']['precision']],
                        'Recall': [class_report['0']['recall'], class_report['1']['recall']],
                        'F1-Score': [class_report['0']['f1-score'], class_report['1']['f1-score']],
                    }, index=['Down (0)', 'Up (1)'])
                    
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    st.write(f"**Overall Accuracy:** {class_report['accuracy']:.2%}")
                    st.write(f"**Macro Average F1:** {class_report['macro avg']['f1-score']:.3f}")
                    st.write(f"**Weighted Average F1:** {class_report['weighted avg']['f1-score']:.3f}")
                
                # Top features insight
                top_5_features = sorted(
                    results['feature_importance'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                
                st.subheader("🏆 Top 5 Most Important Features")
                for i, (feature, importance) in enumerate(top_5_features, 1):
                    st.write(f"{i}. **{feature}**: {importance:.3f}")
                
                # Model interpretation
                st.info("""
                💡 **Model Interpretation:**
                - **Accuracy**: Percentage of correct predictions
                - **Precision**: Of all positive predictions, how many were correct
                - **Recall**: Of all actual positives, how many were predicted correctly
                - **F1-Score**: Harmonic mean of precision and recall
                - **Feature Importance**: How much each feature contributes to predictions
                """)
                
        except Exception as e:
            st.error(f"❌ Error during model training: {str(e)}")
            st.info("💡 Try adjusting the test size or check your data quality.")

# Example usage in your main app.py:
"""
def main():
    # ... your existing code ...
    
    # After processing your stock data and ensuring it has a 'target' column:
    if 'processed_data' in locals() and 'target' in processed_data.columns:
        add_ml_model_section(processed_data)
    
    # ... rest of your app ...
"""

if __name__ == "__main__":
    st.title("🤖 ML Model Integration Example")
    st.write("This is an example of how to integrate ML models into your Streamlit app.")
    st.code("""
# Add this to your main app.py:
from ml_integration import add_ml_model_section

# In your main function, after processing data:
add_ml_model_section(your_processed_data)
""")
