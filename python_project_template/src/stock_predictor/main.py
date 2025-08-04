"""
Main Stock Predictor class - Entry point cho stock market prediction system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import os
import sys
from datetime import datetime, timedelta

# Add current directory to path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import modules using absolute imports
try:
    from stock_predictor.data.collector import DataCollector
    from stock_predictor.data.preprocessor import DataPreprocessor
    from stock_predictor.data.features import FeatureEngineer
    from stock_predictor.models.ensemble import EnsemblePredictor
    from stock_predictor.models.traditional import TraditionalModels
    from stock_predictor.models.deep_learning import DeepLearningModels
    # from stock_predictor.models.arima import ARIMAModel  # Disabled for now
    from stock_predictor.evaluation.metrics import ModelEvaluator
except ImportError:
    # Fallback to relative imports if absolute imports fail
    from data.collector import DataCollector
    from data.preprocessor import DataPreprocessor
    from data.features import FeatureEngineer
    from models.ensemble import EnsemblePredictor
    from models.traditional import TraditionalModels
    from models.deep_learning import DeepLearningModels
    # from models.arima import ARIMAModel  # Disabled for now
    from evaluation.metrics import ModelEvaluator
from .evaluation.visualization import Visualizer
from .utils.config import data_config, model_config, evaluation_config
from .utils.helpers import setup_logging, validate_data_format, save_object, load_object

logger = logging.getLogger(__name__)

class StockPredictor:
    """
    Main class cho Stock Market Prediction System
    
    Kết hợp nhiều techniques machine learning để dự báo chỉ số thị trường chứng khoán
    """
    
    def __init__(self, symbol: str = '^VNI', **kwargs):
        """
        Args:
            symbol: Mã chứng khoán để dự báo (default: ^VNI - VN-Index)
            **kwargs: Additional configuration
        """
        self.symbol = symbol
        self.config = kwargs
        
        # Initialize components
        self.data_collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.targets = None
        
        # Training/test splits
        self.X_train = None
        self.X_val = None  
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Models
        self.models = {}
        self.ensemble = None
        self.best_model = None
        
        # Results
        self.training_results = {}
        self.evaluation_results = {}
        self.predictions = {}
        
        # Setup logging
        setup_logging(level=kwargs.get('log_level', 'INFO'))
        logger.info(f"StockPredictor initialized for symbol: {symbol}")
        
    def load_data(
        self, 
        period: str = '2y',
        interval: str = '1d',
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load dữ liệu chứng khoán
        
        Args:
            period: Thời gian dữ liệu ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Khoảng thời gian ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            force_refresh: Có refresh cache không
            
        Returns:
            DataFrame with stock data
        """
        logger.info(f"Loading data for {self.symbol}...")
        
        try:
            # Fetch data
            self.raw_data = self.data_collector.fetch_stock_data(
                symbol=self.symbol,
                period=period,
                interval=interval,
                force_refresh=force_refresh
            )
            
            # Validate data format
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not validate_data_format(self.raw_data, required_columns):
                raise ValueError("Invalid data format")
                
            logger.info(f"Data loaded successfully: {len(self.raw_data)} records")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def prepare_features(
        self, 
        target_col: str = 'Close',
        include_technical_indicators: bool = True,
        include_price_features: bool = True,
        include_time_features: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Chuẩn bị features và targets
        
        Args:
            target_col: Cột target để dự báo
            include_technical_indicators: Có include technical indicators không
            include_price_features: Có include price features không
            include_time_features: Có include time features không
            
        Returns:
            Tuple (features, targets)
        """
        logger.info("Preparing features...")
        
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        try:
            # Start with raw data
            data = self.raw_data.copy()
            
            # Add technical indicators
            if include_technical_indicators:
                data = self.feature_engineer.add_technical_indicators(data)
                
            # Add price features
            if include_price_features:
                data = self.feature_engineer.add_price_features(data)
                data = self.feature_engineer.add_volatility_features(data)
                data = self.feature_engineer.add_momentum_features(data)
                
            # Add time features
            if include_time_features:
                data = self.feature_engineer.add_time_features(data)
                
            # Store processed data
            self.processed_data = data
            
            # Prepare features and targets
            self.features, self.targets = self.preprocessor.prepare_features_targets(
                data=data,
                target_col=target_col,
                lag_periods=data_config.LAG_PERIODS
            )
            
            logger.info(f"Features prepared: {len(self.features.columns)} features, {len(self.targets)} samples")
            
            return self.features, self.targets
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
            
    def split_data(
        self, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        time_based: bool = True
    ) -> None:
        """
        Chia dữ liệu thành train/validation/test sets
        
        Args:
            train_ratio: Tỉ lệ train set
            val_ratio: Tỉ lệ validation set
            test_ratio: Tỉ lệ test set
            time_based: Có chia theo thời gian không (quan trọng với time series)
        """
        logger.info("Splitting data...")
        
        if self.features is None or self.targets is None:
            raise ValueError("No features prepared. Call prepare_features() first.")
            
        try:
            # Split data
            splits = self.preprocessor.split_data(
                features=self.features,
                targets=self.targets,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                time_based=time_based
            )
            
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = splits
            
            # Scale features
            self.X_train, self.X_val, self.X_test = self.preprocessor.scale_features(
                self.X_train, self.X_val, self.X_test
            )
            
            logger.info("Data split and scaled successfully")
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
            
    def train_individual_models(self, model_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train individual models
        
        Args:
            model_types: List model types to train. If None, train all available
            
        Returns:
            Dict with training results
        """
        logger.info("Training individual models...")
        
        if self.X_train is None:
            raise ValueError("No training data. Call split_data() first.")
            
        if model_types is None:
            model_types = ['rf', 'xgb', 'lgb', 'svr', 'lstm', 'arima']
            
        results = {}
        
        for model_type in model_types:
            try:
                logger.info(f"Training {model_type} model...")
                
                if model_type in ['rf', 'xgb', 'lgb', 'svr']:
                    # Traditional ML models
                    model = TraditionalModels.create_model(model_type)
                    
                elif model_type in ['lstm', 'gru']:
                    # Deep learning models
                    model = DeepLearningModels.create_model(model_type)
                    
                elif model_type == 'arima':
                    # ARIMA model - temporarily disabled
                    logger.warning("ARIMA model is temporarily disabled")
                    continue
                    # model = ARIMAModel()
                    
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue
                    
                # Train model
                training_result = model.train(
                    X_train=self.X_train,
                    y_train=self.y_train,
                    X_val=self.X_val,
                    y_val=self.y_val
                )
                
                # Store model and results
                self.models[model_type] = model
                results[model_type] = training_result
                
                logger.info(f"Model {model_type} trained successfully")
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
                results[model_type] = {'error': str(e)}
                continue
                
        self.training_results = results
        logger.info(f"Training completed for {len(self.models)} models")
        
        return results
        
    def train_ensemble(self, ensemble_method: str = 'voting') -> Dict[str, Any]:
        """
        Train ensemble model
        
        Args:
            ensemble_method: Ensemble method ('voting', 'stacking', 'weighted', 'bayesian')
            
        Returns:
            Training results
        """
        logger.info(f"Training ensemble with {ensemble_method} method...")
        
        if len(self.models) == 0:
            logger.info("No individual models found. Training default ensemble...")
            
        try:
            # Create ensemble
            self.ensemble = EnsemblePredictor(ensemble_method=ensemble_method)
            
            # Add trained models to ensemble
            for name, model in self.models.items():
                if model.is_trained:
                    self.ensemble.add_model(model, name)
                    
            # Train ensemble
            ensemble_result = self.ensemble.train(
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val
            )
            
            logger.info("Ensemble trained successfully")
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error training ensemble: {str(e)}")
            raise
            
    def evaluate_models(self) -> Dict[str, Any]:
        """
        Evaluate all trained models
        
        Returns:
            Dict with evaluation results
        """
        logger.info("Evaluating models...")
        
        if self.X_test is None or self.y_test is None:
            raise ValueError("No test data available")
            
        results = {}
        
        # Evaluate individual models
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    # Make predictions
                    y_pred = model.predict(self.X_test)
                    
                    # Evaluate
                    model_result = self.evaluator.evaluate_model(
                        y_true=self.y_test.values,
                        y_pred=y_pred,
                        model_name=name
                    )
                    
                    results[name] = model_result
                    self.predictions[name] = y_pred
                    
                except Exception as e:
                    logger.error(f"Error evaluating {name}: {str(e)}")
                    results[name] = {'error': str(e)}
                    
        # Evaluate ensemble
        if self.ensemble and self.ensemble.is_trained:
            try:
                ensemble_pred = self.ensemble.predict(self.X_test)
                
                ensemble_result = self.evaluator.evaluate_model(
                    y_true=self.y_test.values,
                    y_pred=ensemble_pred,
                    model_name=f"Ensemble_{self.ensemble.ensemble_method}"
                )
                
                results[f"Ensemble_{self.ensemble.ensemble_method}"] = ensemble_result
                self.predictions[f"Ensemble_{self.ensemble.ensemble_method}"] = ensemble_pred
                
            except Exception as e:
                logger.error(f"Error evaluating ensemble: {str(e)}")
                results["ensemble"] = {'error': str(e)}
                
        self.evaluation_results = results
        
        # Find best model
        best_model_name, best_score = self.evaluator.get_best_model(results, metric='mae')
        self.best_model = best_model_name
        
        logger.info(f"Evaluation completed. Best model: {best_model_name} (MAE: {best_score:.4f})")
        
        return results
        
    def predict_future(self, days: int = 30) -> Dict[str, np.ndarray]:
        """
        Dự báo cho future periods
        
        Args:
            days: Số ngày để dự báo
            
        Returns:
            Dict with predictions from each model
        """
        logger.info(f"Predicting future {days} days...")
        
        if self.processed_data is None:
            raise ValueError("No processed data available")
            
        try:
            # Use last features as input for prediction
            last_features = self.features.tail(days).copy()
            
            # If not enough data, repeat last row
            if len(last_features) < days:
                last_row = self.features.iloc[-1:].copy()
                missing_rows = days - len(last_features)
                
                for _ in range(missing_rows):
                    last_features = pd.concat([last_features, last_row], ignore_index=False)
                    
            # Scale features
            if hasattr(self.preprocessor, 'scalers') and 'features' in self.preprocessor.scalers:
                scaler = self.preprocessor.scalers['features']
                last_features_scaled = pd.DataFrame(
                    scaler.transform(last_features),
                    index=last_features.index,
                    columns=last_features.columns
                )
            else:
                last_features_scaled = last_features
                
            # Make predictions with each model
            future_predictions = {}
            
            for name, model in self.models.items():
                if model.is_trained:
                    try:
                        pred = model.predict(last_features_scaled)
                        future_predictions[name] = pred
                    except Exception as e:
                        logger.warning(f"Error predicting with {name}: {str(e)}")
                        
            # Ensemble predictions
            if self.ensemble and self.ensemble.is_trained:
                try:
                    ensemble_pred = self.ensemble.predict(last_features_scaled)
                    future_predictions[f"Ensemble_{self.ensemble.ensemble_method}"] = ensemble_pred
                except Exception as e:
                    logger.warning(f"Error predicting with ensemble: {str(e)}")
                    
            logger.info(f"Future predictions generated for {len(future_predictions)} models")
            
            return future_predictions
            
        except Exception as e:
            logger.error(f"Error predicting future: {str(e)}")
            raise
            
    def create_report(self) -> str:
        """
        Tạo báo cáo tổng hợp
        
        Returns:
            String report
        """
        if not self.evaluation_results:
            return "No evaluation results available. Run evaluate_models() first."
            
        return self.evaluator.create_performance_summary(self.evaluation_results)
        
    def save_model(self, filepath: str, model_name: Optional[str] = None) -> None:
        """
        Lưu model ra file
        
        Args:
            filepath: Đường dẫn file
            model_name: Tên model để lưu. Nếu None, lưu best model
        """
        try:
            if model_name is None:
                model_name = self.best_model
                
            if model_name is None:
                raise ValueError("No best model found")
                
            if model_name in self.models:
                model = self.models[model_name]
            elif model_name.startswith("Ensemble") and self.ensemble:
                model = self.ensemble
            else:
                raise ValueError(f"Model {model_name} not found")
                
            # Save model
            model.save_model(filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def save_results(self, directory: str) -> None:
        """
        Lưu tất cả results ra directory
        
        Args:
            directory: Directory để lưu
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save evaluation results
            if self.evaluation_results:
                self.evaluator.save_results(os.path.join(directory, 'evaluation_results.json'))
                
            # Save predictions
            if self.predictions:
                predictions_df = pd.DataFrame(self.predictions)
                predictions_df.to_csv(os.path.join(directory, 'predictions.csv'))
                
            # Save data
            if self.processed_data is not None:
                self.processed_data.to_csv(os.path.join(directory, 'processed_data.csv'))
                
            # Save models
            models_dir = os.path.join(directory, 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            for name, model in self.models.items():
                if model.is_trained:
                    model.save_model(os.path.join(models_dir, f'{name}.pkl'))
                    
            if self.ensemble and self.ensemble.is_trained:
                self.ensemble.save_ensemble(os.path.join(models_dir, 'ensemble.pkl'))
                
            logger.info(f"Results saved to {directory}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
            
    def run_complete_pipeline(
        self, 
        period: str = '2y',
        model_types: Optional[List[str]] = None,
        ensemble_method: str = 'voting'
    ) -> Dict[str, Any]:
        """
        Chạy complete pipeline từ load data đến evaluation
        
        Args:
            period: Period để load data
            model_types: List model types để train
            ensemble_method: Ensemble method
            
        Returns:
            Dict with all results
        """
        logger.info("Running complete pipeline...")
        
        try:
            # Step 1: Load data
            self.load_data(period=period)
            
            # Step 2: Prepare features
            self.prepare_features()
            
            # Step 3: Split data
            self.split_data()
            
            # Step 4: Train models
            training_results = self.train_individual_models(model_types)
            
            # Step 5: Train ensemble
            ensemble_results = self.train_ensemble(ensemble_method)
            
            # Step 6: Evaluate
            evaluation_results = self.evaluate_models()
            
            # Step 7: Future predictions
            future_predictions = self.predict_future(days=30)
            
            pipeline_results = {
                'training_results': training_results,
                'ensemble_results': ensemble_results,
                'evaluation_results': evaluation_results,
                'future_predictions': future_predictions,
                'best_model': self.best_model,
                'data_info': {
                    'symbol': self.symbol,
                    'total_samples': len(self.features) if self.features is not None else 0,
                    'features_count': len(self.features.columns) if self.features is not None else 0,
                    'train_samples': len(self.X_train) if self.X_train is not None else 0,
                    'test_samples': len(self.X_test) if self.X_test is not None else 0
                }
            }
            
            logger.info("Complete pipeline finished successfully")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Error in complete pipeline: {str(e)}")
            raise
    
    def load_and_preprocess_data(self, data_dir: str) -> pd.DataFrame:
        """
        Load and preprocess data from CSV files directory.
        Based on the reference implementation.
        
        Args:
            data_dir: Directory containing CSV files
            
        Returns:
            Preprocessed DataFrame with features
        """
        logger.info(f"Loading and preprocessing data from {data_dir}...")
        
        # Step 1: Load and process all CSV files
        merged_data = self.preprocessor.load_and_process_all(data_dir)
        logger.info(f"Merged data shape: {merged_data.shape}")
        
        # Step 2: Add technical indicators
        data_with_features = self.feature_engineer.create_technical_indicators(merged_data)
        logger.info(f"Data with features shape: {data_with_features.shape}")
        
        # Store the processed data
        self.processed_data = data_with_features
        
        return data_with_features