"""
Ensemble Methods cho kết hợp nhiều models (Fusion of ML Techniques)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from .base_model import BaseModel
from .traditional import TraditionalModels
from .deep_learning import DeepLearningModels

logger = logging.getLogger(__name__)

class EnsemblePredictor(BaseModel):
    """
    Ensemble Predictor kết hợp nhiều models bằng các fusion techniques
    """
    
    def __init__(self, ensemble_method: str = 'voting', **kwargs):
        """
        Args:
            ensemble_method: Phương pháp ensemble ('voting', 'stacking', 'weighted', 'bayesian')
            **kwargs: Additional parameters
        """
        super().__init__(f"Ensemble_{ensemble_method}", **kwargs)
        
        self.ensemble_method = ensemble_method
        self.base_models = []
        self.model_names = []
        self.weights = None
        self.meta_model = None
        self.model_performances = {}
        
    def add_model(self, model: BaseModel, name: Optional[str] = None) -> None:
        """
        Thêm base model vào ensemble
        
        Args:
            model: BaseModel instance
            name: Tên cho model (optional)
        """
        if name is None:
            name = f"model_{len(self.base_models)}"
            
        self.base_models.append(model)
        self.model_names.append(name)
        
        logger.info(f"Đã thêm model {name} vào ensemble")
        
    def create_default_ensemble(self) -> None:
        """Tạo ensemble mặc định với các models phổ biến"""
        
        logger.info("Tạo default ensemble...")
        
        # Traditional ML models
        rf_model = TraditionalModels.create_model('rf', n_estimators=100, max_depth=10)
        xgb_model = TraditionalModels.create_model('xgb', n_estimators=100, max_depth=6)
        lgb_model = TraditionalModels.create_model('lgb', n_estimators=100, max_depth=6)
        svr_model = TraditionalModels.create_model('svr', C=1.0, gamma='scale')
        
        # Deep learning models
        lstm_model = DeepLearningModels.create_model('lstm', units=50, epochs=50)
        gru_model = DeepLearningModels.create_model('gru', units=50, epochs=50)
        
        # Thêm vào ensemble
        self.add_model(rf_model, "RandomForest")
        self.add_model(xgb_model, "XGBoost")
        self.add_model(lgb_model, "LightGBM")
        self.add_model(svr_model, "SVR")
        self.add_model(lstm_model, "LSTM")
        self.add_model(gru_model, "GRU")
        
        logger.info(f"Đã tạo ensemble với {len(self.base_models)} models")
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Huấn luyện ensemble
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dict chứa training results
        """
        logger.info(f"Training ensemble với {len(self.base_models)} models...")
        
        if len(self.base_models) == 0:
            self.create_default_ensemble()
            
        self.validate_inputs(X_train, y_train)
        
        # Train từng base model
        model_results = {}
        valid_models = []
        valid_names = []
        
        for i, (model, name) in enumerate(zip(self.base_models, self.model_names)):
            try:
                logger.info(f"Training model {i+1}/{len(self.base_models)}: {name}")
                
                result = model.train(X_train, y_train, X_val, y_val, **kwargs)
                model_results[name] = result
                
                valid_models.append(model)
                valid_names.append(name)
                
                logger.info(f"Model {name} trained successfully. MAE: {result.get('train_mae', 'N/A'):.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to train model {name}: {str(e)}")
                continue
                
        # Cập nhật danh sách models thành công
        self.base_models = valid_models
        self.model_names = valid_names
        self.model_performances = model_results
        
        if len(self.base_models) == 0:
            raise ValueError("Không có model nào được train thành công")
            
        # Train ensemble method
        if self.ensemble_method == 'voting':
            self._train_voting()
        elif self.ensemble_method == 'stacking':
            self._train_stacking(X_train, y_train, X_val, y_val)
        elif self.ensemble_method == 'weighted':
            self._train_weighted(X_val, y_val)
        elif self.ensemble_method == 'bayesian':
            self._train_bayesian(X_val, y_val)
        else:
            raise ValueError(f"Ensemble method không được hỗ trợ: {self.ensemble_method}")
            
        self.is_trained = True
        
        # Đánh giá ensemble
        ensemble_results = self._evaluate_ensemble(X_train, y_train, X_val, y_val)
        
        logger.info(f"Ensemble training completed. Train MAE: {ensemble_results.get('train_mae', 'N/A'):.4f}")
        
        return {
            'ensemble_results': ensemble_results,
            'individual_results': model_results,
            'ensemble_method': self.ensemble_method,
            'num_models': len(self.base_models)
        }
        
    def _train_voting(self) -> None:
        """Train simple voting ensemble"""
        logger.info("Training voting ensemble...")
        
        # Simple equal weight voting
        self.weights = np.ones(len(self.base_models)) / len(self.base_models)
        
    def _train_stacking(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """Train stacking ensemble với meta-learner"""
        logger.info("Training stacking ensemble...")
        
        # Tạo out-of-fold predictions cho stacking
        if X_val is not None and y_val is not None:
            # Sử dụng validation set để tạo meta-features
            meta_features = []
            
            for model in self.base_models:
                try:
                    pred = model.predict(X_val)
                    meta_features.append(pred)
                except Exception as e:
                    logger.warning(f"Model prediction failed: {str(e)}")
                    # Fallback: sử dụng giá trị trung bình
                    meta_features.append(np.full(len(X_val), y_val.mean()))
                    
            # Stack predictions
            meta_X = np.column_stack(meta_features)
            
            # Train meta-learner
            self.meta_model = LinearRegression()
            self.meta_model.fit(meta_X, y_val)
            
        else:
            logger.warning("Không có validation set cho stacking, sử dụng simple voting")
            self._train_voting()
            
    def _train_weighted(
        self, 
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """Train weighted ensemble dựa trên performance"""
        logger.info("Training weighted ensemble...")
        
        if X_val is None or y_val is None:
            logger.warning("Không có validation set cho weighted ensemble, sử dụng equal weights")
            self._train_voting()
            return
            
        # Tính performance của từng model
        model_errors = []
        
        for model, name in zip(self.base_models, self.model_names):
            try:
                pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, pred)
                model_errors.append(mae)
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {str(e)}")
                model_errors.append(float('inf'))  # Penalty cho failed model
                
        # Tính weights nghịch đảo với error (model tốt hơn có weight cao hơn)
        model_errors = np.array(model_errors)
        
        # Tránh division by zero
        model_errors = np.where(model_errors == 0, 1e-8, model_errors)
        model_errors = np.where(model_errors == float('inf'), np.max(model_errors[model_errors != float('inf')]) * 2, model_errors)
        
        # Inverse weights
        inverse_errors = 1.0 / model_errors
        self.weights = inverse_errors / np.sum(inverse_errors)
        
        logger.info(f"Model weights: {dict(zip(self.model_names, self.weights))}")
        
    def _train_bayesian(
        self, 
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """Train Bayesian Model Averaging"""
        logger.info("Training Bayesian ensemble...")
        
        # Simplified Bayesian averaging based on likelihood
        if X_val is None or y_val is None:
            self._train_voting()
            return
            
        # Tính likelihood cho từng model
        likelihoods = []
        
        for model in self.base_models:
            try:
                pred = model.predict(X_val)
                mse = mean_squared_error(y_val, pred)
                
                # Likelihood proportional to exp(-MSE/2σ²)
                # Simplified: use exp(-MSE)
                likelihood = np.exp(-mse)
                likelihoods.append(likelihood)
                
            except Exception:
                likelihoods.append(1e-10)  # Very small likelihood for failed models
                
        # Normalize to get Bayesian weights
        likelihoods = np.array(likelihoods)
        self.weights = likelihoods / np.sum(likelihoods)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Dự báo với ensemble
        
        Args:
            X: Features để dự báo
            
        Returns:
            Array predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble chưa được trained")
            
        if len(self.base_models) == 0:
            raise ValueError("Không có base models")
            
        # Lấy predictions từ tất cả models
        predictions = []
        
        for model, name in zip(self.base_models, self.model_names):
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {str(e)}")
                # Fallback: sử dụng giá trị constant
                predictions.append(np.full(len(X), X.iloc[:, 0].mean() if len(X) > 0 else 0))
                
        # Combine predictions
        if self.ensemble_method == 'stacking' and self.meta_model is not None:
            # Stacking với meta-learner
            meta_features = np.column_stack(predictions)
            ensemble_pred = self.meta_model.predict(meta_features)
        else:
            # Weighted combination
            predictions = np.array(predictions)
            
            if self.weights is not None:
                # Sử dụng weights đã học
                ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
            else:
                # Simple average
                ensemble_pred = np.mean(predictions, axis=0)
                
        return ensemble_pred
        
    def _evaluate_ensemble(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Đánh giá performance của ensemble"""
        
        results = {}
        
        # Train set evaluation
        train_pred = self.predict(X_train)
        results['train_mae'] = mean_absolute_error(y_train, train_pred)
        results['train_rmse'] = np.sqrt(mean_squared_error(y_train, train_pred))
        results['train_r2'] = r2_score(y_train, train_pred)
        
        # Validation set evaluation
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            results['val_mae'] = mean_absolute_error(y_val, val_pred)
            results['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
            results['val_r2'] = r2_score(y_val, val_pred)
            
        return results
        
    def get_model_weights(self) -> Dict[str, float]:
        """Lấy weights của từng model trong ensemble"""
        if self.weights is None:
            return {name: 1.0/len(self.model_names) for name in self.model_names}
        
        return dict(zip(self.model_names, self.weights))
        
    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Lấy predictions từ từng model riêng lẻ"""
        predictions = {}
        
        for model, name in zip(self.base_models, self.model_names):
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {str(e)}")
                predictions[name] = np.full(len(X), np.nan)
                
        return predictions
        
    def get_model_performances(self) -> Dict[str, Dict[str, float]]:
        """Lấy performance của từng model"""
        return self.model_performances.copy()
        
    def save_ensemble(self, filepath: str) -> None:
        """Lưu ensemble ra file"""
        try:
            ensemble_data = {
                'base_models': self.base_models,
                'model_names': self.model_names,
                'weights': self.weights,
                'meta_model': self.meta_model,
                'ensemble_method': self.ensemble_method,
                'model_performances': self.model_performances,
                'is_trained': self.is_trained
            }
            
            joblib.dump(ensemble_data, filepath)
            logger.info(f"Đã lưu ensemble ra {filepath}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu ensemble: {str(e)}")
            raise
            
    def load_ensemble(self, filepath: str) -> None:
        """Load ensemble từ file"""
        try:
            ensemble_data = joblib.load(filepath)
            
            self.base_models = ensemble_data['base_models']
            self.model_names = ensemble_data['model_names']
            self.weights = ensemble_data['weights']
            self.meta_model = ensemble_data['meta_model']
            self.ensemble_method = ensemble_data['ensemble_method']
            self.model_performances = ensemble_data['model_performances']
            self.is_trained = ensemble_data['is_trained']
            
            logger.info(f"Đã load ensemble từ {filepath}")
            
        except Exception as e:
            logger.error(f"Lỗi khi load ensemble: {str(e)}")
            raise
