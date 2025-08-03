
# Simple Stock Prediction Integration Guide

## Working Models Successfully Tested:
- ✅ Random Forest Classifier 
- ✅ XGBoost Classifier

## Basic Integration Steps:

### 1. Add to your Streamlit app:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

def add_simple_features(data):
    """Add simple technical indicators manually."""
    # Simple moving averages
    data['ma_5'] = data['close'].rolling(window=5).mean()
    data['ma_20'] = data['close'].rolling(window=20).mean()
    
    # Price change features
    data['price_change'] = data['close'].diff()
    data['price_change_pct'] = data['close'].pct_change()
    
    # Lag features
    data['close_lag_1'] = data['close'].shift(1)
    data['close_lag_2'] = data['close'].shift(2)
    
    # Rolling statistics
    data['close_rolling_mean_5'] = data['close'].rolling(5).mean()
    data['close_rolling_std_5'] = data['close'].rolling(5).std()
    
    return data.dropna()

def train_simple_model(data, model_type='random_forest'):
    """Train a simple ML model for stock prediction."""
    
    # Prepare features
    exclude_columns = ['code', 'date', 'year', 'month', 'day', 'target', 'return']
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    X = data[feature_columns].fillna(data[feature_columns].mean())
    y = data['target']
    
    # Train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:  # xgboost
        model = xgb.XGBClassifier(random_state=42, verbosity=0)
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, feature_columns

# In your Streamlit app:
if st.button("🚀 Train Simple ML Model"):
    with st.spinner("Training model..."):
        # Add features
        enriched_data = add_simple_features(your_data.copy())
        
        # Train model
        model, scaler, accuracy, features = train_simple_model(enriched_data)
        
        # Show results
        st.success(f"Model trained! Accuracy: {accuracy:.2%}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            st.subheader("📊 Feature Importance")
            st.bar_chart(importance_df.set_index('Feature'))
```

### 2. Key Benefits:
- ✅ No complex dependencies (no TensorFlow needed)
- ✅ Fast training and prediction
- ✅ Good interpretability with feature importance
- ✅ Robust performance on financial data
- ✅ Easy to integrate and maintain

### 3. Performance Notes:
- Random Forest typically provides good baseline performance
- XGBoost often achieves slightly better accuracy
- Both models handle missing data well
- Feature importance helps understand what drives predictions

### 4. Next Steps:
- Integrate the working models into your Streamlit app
- Add model selection UI (Random Forest vs XGBoost)
- Display feature importance charts
- Add prediction confidence intervals
- Implement model persistence (save/load)
