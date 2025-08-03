"""
Advanced Stock Forecasting Module
Provides advanced machine learning models for stock price prediction including:
- Support Vector Regression (SVR) 
- Neural Networks
- Technical Indicators Integration
- Multiple timeframe predictions
Based on the reference implementation from copy_of_đồ_án_dss_nhóm_1.py
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings

# Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Technical Analysis
import ta

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedStockForecaster:
    """Advanced stock forecasting with multiple ML models and technical indicators."""
    
    def __init__(self, model_type: str = 'neural_network', scaler_type: str = 'minmax'):
        """
        Initialize the advanced forecaster.
        
        Args:
            model_type: Type of model ('svr', 'neural_network', 'lstm', 'random_forest', 'xgboost')
            scaler_type: Type of scaler ('minmax', 'standard')
        """
        self.model_type = model_type
        self.scaler_type = scaler_type
        self.models = {}
        self.scalers = {}
        self.feature_scalers = {}
        self.data = {}
        self.processed_data = {}
        self.feature_columns = []
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to the DataFrame.
        Based on the reference implementation with all major indicators.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        logger.info("Adding technical indicators...")
        all_data = []
        
        for code in df['code'].unique():
            # Sort by date for proper calculation
            if 'date' in df.columns:
                sub_df = df[df['code'] == code].sort_values('date').copy()
            else:
                sub_df = df[df['code'] == code].sort_values(['year', 'month', 'day']).copy()
            
            if len(sub_df) < 50:  # Need enough data for indicators
                logger.warning(f"Insufficient data for {code}: {len(sub_df)} rows")
                continue
                
            try:
                # 1. Simple Moving Averages (SMA)
                sub_df['ma_5'] = ta.trend.sma_indicator(sub_df['close'], window=5).round(2)
                sub_df['ma_10'] = ta.trend.sma_indicator(sub_df['close'], window=10).round(2)
                sub_df['ma_20'] = ta.trend.sma_indicator(sub_df['close'], window=20).round(2)
                sub_df['ma_50'] = ta.trend.sma_indicator(sub_df['close'], window=50).round(2)
                
                # 2. Exponential Moving Averages (EMA)
                sub_df['ema_12'] = ta.trend.ema_indicator(sub_df['close'], window=12).round(2)
                sub_df['ema_26'] = ta.trend.ema_indicator(sub_df['close'], window=26).round(2)
                
                # 3. MACD (Moving Average Convergence Divergence)
                sub_df['macd'] = ta.trend.macd(sub_df['close']).round(2)
                sub_df['macd_signal'] = ta.trend.macd_signal(sub_df['close']).round(2)
                sub_df['macd_diff'] = ta.trend.macd_diff(sub_df['close']).round(2)
                
                # 4. RSI (Relative Strength Index)
                sub_df['rsi_14'] = ta.momentum.rsi(sub_df['close'], window=14).round(2)
                sub_df['rsi_21'] = ta.momentum.rsi(sub_df['close'], window=21).round(2)
                
                # 5. Bollinger Bands
                bb = ta.volatility.BollingerBands(sub_df['close'], window=20, window_dev=2)
                sub_df['bb_bbm'] = bb.bollinger_mavg().round(2)   # Middle band (MA)
                sub_df['bb_bbh'] = bb.bollinger_hband().round(2)  # Upper band
                sub_df['bb_bbl'] = bb.bollinger_lband().round(2)  # Lower band
                sub_df['bb_width'] = (sub_df['bb_bbh'] - sub_df['bb_bbl']).round(2)
                sub_df['bb_position'] = ((sub_df['close'] - sub_df['bb_bbl']) / 
                                       (sub_df['bb_bbh'] - sub_df['bb_bbl'])).round(2)
                
                # 6. ATR (Average True Range)
                sub_df['atr_14'] = ta.volatility.average_true_range(
                    sub_df['high'], sub_df['low'], sub_df['close'], window=14
                ).round(2)
                
                # 7. OBV (On-Balance Volume)
                sub_df['obv'] = ta.volume.on_balance_volume(sub_df['close'], sub_df['volume']).round(2)
                
                # 8. Stochastic Oscillator
                sub_df['stoch'] = ta.momentum.stoch(sub_df['high'], sub_df['low'], sub_df['close']).round(2)
                sub_df['stoch_signal'] = ta.momentum.stoch_signal(sub_df['high'], sub_df['low'], sub_df['close']).round(2)
                
                # 9. Williams %R
                sub_df['williams_r'] = ta.momentum.williams_r(sub_df['high'], sub_df['low'], sub_df['close']).round(2)
                
                # 10. Volume indicators
                # Use a simple volume moving average instead of volume_sma
                sub_df['volume_sma'] = sub_df['volume'].rolling(window=20).mean().round(2)
                
                # 11. Price-based features
                sub_df['price_change'] = sub_df['close'].diff().round(2)
                sub_df['price_change_pct'] = sub_df['close'].pct_change().round(4)
                sub_df['high_low_pct'] = ((sub_df['high'] - sub_df['low']) / sub_df['close']).round(4)
                sub_df['close_open_pct'] = ((sub_df['close'] - sub_df['open']) / sub_df['open']).round(4)
                
                # 12. Lag features
                for lag in [1, 2, 3, 5]:
                    sub_df[f'close_lag_{lag}'] = sub_df['close'].shift(lag)
                    sub_df[f'volume_lag_{lag}'] = sub_df['volume'].shift(lag)
                    sub_df[f'return_lag_{lag}'] = sub_df['return'].shift(lag) if 'return' in sub_df.columns else 0
                
                # 13. Rolling statistics
                for window in [5, 10, 20]:
                    sub_df[f'close_rolling_mean_{window}'] = sub_df['close'].rolling(window).mean().round(2)
                    sub_df[f'close_rolling_std_{window}'] = sub_df['close'].rolling(window).std().round(2)
                    sub_df[f'volume_rolling_mean_{window}'] = sub_df['volume'].rolling(window).mean().round(2)
                
                # Drop rows with NaN values from indicator calculations
                sub_df = sub_df.dropna().reset_index(drop=True)
                all_data.append(sub_df)
                
                logger.info(f"Added indicators for {code}: {len(sub_df)} rows")
                
            except Exception as e:
                logger.error(f"Error adding indicators for {code}: {str(e)}")
                continue
        
        if not all_data:
            logger.error("No data processed successfully")
            return pd.DataFrame()
            
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"Technical indicators added. Final shape: {result.shape}")
        return result
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'target') -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature matrix for machine learning.
        
        Args:
            df: Input DataFrame with indicators
            target_column: Name of target column
            
        Returns:
            Tuple of (features_df, feature_column_names)
        """
        logger.info("Preparing features for ML models...")
        
        # Exclude non-feature columns
        exclude_columns = [
            'code', 'date', 'year', 'month', 'day', target_column,
            'return'  # Don't use future return as feature
        ]
        
        # Select numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Ensure we have features
        if not feature_columns:
            raise ValueError("No suitable feature columns found")
            
        # Create feature matrix
        features_df = df[feature_columns].copy()
        
        # Handle any remaining missing values
        features_df = features_df.fillna(features_df.mean())
        
        logger.info(f"Prepared {len(feature_columns)} features: {feature_columns[:10]}...")
        self.feature_columns = feature_columns
        
        return features_df, feature_columns
    
    def create_sequences(self, data: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/GRU models.
        
        Args:
            data: Input data array
            sequence_length: Length of sequences
            
        Returns:
            X, y arrays for training
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, 0])  # Assuming first column is target
            
        return np.array(X), np.array(y)
    
    def build_neural_network(self, input_shape: int, model_type: str = 'dense') -> Sequential:
        """
        Build neural network model.
        
        Args:
            input_shape: Number of input features
            model_type: Type of neural network ('dense', 'lstm', 'gru')
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        if model_type == 'dense':
            # Dense neural network
            model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
            model.add(Dropout(0.3))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))  # Binary classification
            
        elif model_type == 'lstm':
            # LSTM network
            model.add(LSTM(50, return_sequences=True, input_shape=(input_shape, 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(50))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))
            
        elif model_type == 'gru':
            # GRU network
            model.add(GRU(50, return_sequences=True, input_shape=(input_shape, 1)))
            model.add(Dropout(0.2))
            model.add(GRU(50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(GRU(50))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, df: pd.DataFrame, stock_code: str = None, 
                   test_size: float = 0.2, validation_size: float = 0.1) -> Dict:
        """
        Train the selected model on the provided data.
        
        Args:
            df: DataFrame with features and target
            stock_code: Specific stock code to train on (None for all)
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Filter data if specific stock code is provided
        if stock_code:
            df = df[df['code'] == stock_code].copy()
            
        if len(df) < 100:
            raise ValueError(f"Insufficient data for training: {len(df)} rows")
        
        # Prepare features and target
        features_df, feature_columns = self.prepare_features(df)
        target = df['target'].values
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            features_df.values, target, test_size=test_size, random_state=42, stratify=target
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        # Scale features
        if self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
            
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        model_key = stock_code if stock_code else 'general'
        self.scalers[model_key] = scaler
        
        # Train model based on type
        results = {}
        
        if self.model_type == 'svr':
            # Support Vector Regression for classification
            from sklearn.svm import SVC
            
            # Grid search for best parameters
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
            
            svm_model = SVC(probability=True)
            grid_search = GridSearchCV(svm_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            
            model = grid_search.best_estimator_
            
            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Metrics
            results = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'best_params': grid_search.best_params_
            }
            
        elif self.model_type in ['neural_network', 'lstm', 'gru']:
            # Neural network models
            if self.model_type == 'neural_network':
                model = self.build_neural_network(X_train_scaled.shape[1], 'dense')
                X_train_model = X_train_scaled
                X_val_model = X_val_scaled
                X_test_model = X_test_scaled
            else:
                # For LSTM/GRU, reshape data
                model = self.build_neural_network(X_train_scaled.shape[1], self.model_type)
                X_train_model = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
                X_val_model = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
                X_test_model = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train_model, y_train,
                validation_data=(X_val_model, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Predictions
            y_train_pred_prob = model.predict(X_train_model)
            y_val_pred_prob = model.predict(X_val_model)
            y_test_pred_prob = model.predict(X_test_model)
            
            y_train_pred = (y_train_pred_prob > 0.5).astype(int).flatten()
            y_val_pred = (y_val_pred_prob > 0.5).astype(int).flatten()
            y_test_pred = (y_test_pred_prob > 0.5).astype(int).flatten()
            
            # Metrics
            results = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'history': history.history
            }
            
        elif self.model_type == 'random_forest':
            # Random Forest
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Feature importance
            feature_importance = dict(zip(feature_columns, model.feature_importances_))
            
            # Metrics
            results = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'feature_importance': feature_importance
            }
            
        elif self.model_type == 'xgboost':
            # XGBoost
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Feature importance
            feature_importance = dict(zip(feature_columns, model.feature_importances_))
            
            # Metrics
            results = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'feature_importance': feature_importance
            }
        
        # Store model
        self.models[model_key] = model
        
        # Add detailed classification report
        results['classification_report'] = classification_report(
            y_test, y_test_pred, output_dict=True
        )
        
        logger.info(f"Model training completed. Test accuracy: {results['test_accuracy']:.4f}")
        
        return results
    
    def predict(self, df: pd.DataFrame, stock_code: str = None) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            df: DataFrame with features
            stock_code: Specific stock code (None for general model)
            
        Returns:
            Array of predictions
        """
        model_key = stock_code if stock_code else 'general'
        
        if model_key not in self.models:
            raise ValueError(f"Model for {model_key} not found. Train the model first.")
            
        model = self.models[model_key]
        scaler = self.scalers[model_key]
        
        # Prepare features
        features_df, _ = self.prepare_features(df)
        features_scaled = scaler.transform(features_df.values)
        
        # Make predictions based on model type
        if self.model_type in ['neural_network', 'lstm', 'gru']:
            if self.model_type != 'neural_network':
                features_scaled = features_scaled.reshape((features_scaled.shape[0], features_scaled.shape[1], 1))
            
            predictions_prob = model.predict(features_scaled)
            predictions = (predictions_prob > 0.5).astype(int).flatten()
        else:
            predictions = model.predict(features_scaled)
            
        return predictions
    
    def create_prediction_chart(self, df: pd.DataFrame, predictions: np.ndarray, 
                              stock_code: str = 'STOCK') -> go.Figure:
        """
        Create a chart showing predictions vs actual values.
        
        Args:
            df: DataFrame with actual data
            predictions: Array of predictions
            stock_code: Stock code for title
            
        Returns:
            Plotly figure
        """
        # Ensure we have date information
        if 'date' in df.columns:
            dates = df['date'].iloc[-len(predictions):]
        else:
            dates = pd.date_range(start='2023-01-01', periods=len(predictions), freq='D')
        
        actual = df['target'].iloc[-len(predictions):].values
        close_prices = df['close'].iloc[-len(predictions):].values
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Price Movement', 'Predictions vs Actual'],
            vertical_spacing=0.1
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=close_prices,
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Predictions vs Actual
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=actual,
                mode='markers',
                name='Actual (1=Up, 0=Down)',
                marker=dict(color='green', size=8)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=predictions,
                mode='markers',
                name='Predicted (1=Up, 0=Down)',
                marker=dict(color='red', size=6, symbol='x')
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Stock Prediction Analysis - {stock_code} ({self.model_type.upper()})',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Direction", row=2, col=1)
        
        return fig
    
    def get_model_summary(self, stock_code: str = None) -> Dict:
        """
        Get summary information about the trained model.
        
        Args:
            stock_code: Specific stock code (None for general model)
            
        Returns:
            Dictionary with model information
        """
        model_key = stock_code if stock_code else 'general'
        
        if model_key not in self.models:
            return {'error': f'Model for {model_key} not found'}
            
        model = self.models[model_key]
        
        summary = {
            'model_type': self.model_type,
            'scaler_type': self.scaler_type,
            'feature_count': len(self.feature_columns),
            'feature_columns': self.feature_columns[:10]  # First 10 features
        }
        
        # Add model-specific information
        if self.model_type == 'random_forest':
            summary['n_estimators'] = model.n_estimators
            summary['max_depth'] = model.max_depth
        elif self.model_type == 'xgboost':
            summary['n_estimators'] = model.n_estimators
            summary['max_depth'] = model.max_depth
            summary['learning_rate'] = model.learning_rate
        elif self.model_type in ['neural_network', 'lstm', 'gru']:
            summary['model_layers'] = len(model.layers)
            summary['total_params'] = model.count_params()
            
        return summary

    def save_model(self, filepath: str, stock_code: str = None):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            stock_code: Specific stock code (None for general model)
        """
        import joblib
        
        model_key = stock_code if stock_code else 'general'
        
        if model_key not in self.models:
            raise ValueError(f"Model for {model_key} not found")
            
        model_data = {
            'model': self.models[model_key],
            'scaler': self.scalers[model_key],
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'scaler_type': self.scaler_type
        }
        
        # Save based on model type
        if self.model_type in ['neural_network', 'lstm', 'gru']:
            # Save Keras model separately
            self.models[model_key].save(f"{filepath}_keras_model.h5")
            # Save other components
            model_data['model'] = None  # Don't save keras model in joblib
            joblib.dump(model_data, f"{filepath}_metadata.pkl")
        else:
            joblib.dump(model_data, f"{filepath}.pkl")
            
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, stock_code: str = None):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
            stock_code: Specific stock code (None for general model)
        """
        import joblib
        from tensorflow.keras.models import load_model
        
        model_key = stock_code if stock_code else 'general'
        
        try:
            # Try to load Keras model first
            if self.model_type in ['neural_network', 'lstm', 'gru']:
                model = load_model(f"{filepath}_keras_model.h5")
                metadata = joblib.load(f"{filepath}_metadata.pkl")
                metadata['model'] = model
                model_data = metadata
            else:
                model_data = joblib.load(f"{filepath}.pkl")
            
            # Restore model components
            self.models[model_key] = model_data['model']
            self.scalers[model_key] = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_type = model_data['model_type']
            self.scaler_type = model_data['scaler_type']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
