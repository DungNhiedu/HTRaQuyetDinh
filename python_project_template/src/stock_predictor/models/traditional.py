"""
Traditional Machine Learning Models
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class RandomForestModel(BaseModel):
    """Random Forest Regression Model"""
    
    def __init__(self, **kwargs):
        super().__init__("RandomForest", **kwargs)
        
        # Default parameters
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.max_depth = kwargs.get('max_depth', 10)
        self.random_state = kwargs.get('random_state', 42)
        self.n_jobs = kwargs.get('n_jobs', -1)
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Huấn luyện Random Forest model"""
        
        logger.info(f"Training {self.name} model...")
        
        self.validate_inputs(X_train, y_train)
        
        # Khởi tạo model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **kwargs
        )
        
        # Training
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Lưu feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Đánh giá trên training set
        train_pred = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
        
        results = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2
        }
        
        # Đánh giá trên validation set nếu có
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            
            results.update({
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            })
            
        self.training_history = results
        logger.info(f"Training completed. Train MAE: {train_mae:.4f}")
        
        return results
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Dự báo với Random Forest"""
        if not self.is_trained:
            raise ValueError("Model chưa được trained")
            
        return self.model.predict(X)

class XGBoostModel(BaseModel):
    """XGBoost Regression Model"""
    
    def __init__(self, **kwargs):
        super().__init__("XGBoost", **kwargs)
        
        # Default parameters
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.max_depth = kwargs.get('max_depth', 6)
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.random_state = kwargs.get('random_state', 42)
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Huấn luyện XGBoost model"""
        
        logger.info(f"Training {self.name} model...")
        
        self.validate_inputs(X_train, y_train)
        
        # Khởi tạo model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            **kwargs
        )
        
        # Chuẩn bị evaluation set
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            
        # Training với early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_trained = True
        
        # Lưu feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Đánh giá
        train_pred = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
        
        results = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            
            results.update({
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            })
            
        self.training_history = results
        logger.info(f"Training completed. Train MAE: {train_mae:.4f}")
        
        return results
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Dự báo với XGBoost"""
        if not self.is_trained:
            raise ValueError("Model chưa được trained")
            
        return self.model.predict(X)

class LightGBMModel(BaseModel):
    """LightGBM Regression Model"""
    
    def __init__(self, **kwargs):
        super().__init__("LightGBM", **kwargs)
        
        # Default parameters
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.max_depth = kwargs.get('max_depth', 6)
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.random_state = kwargs.get('random_state', 42)
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Huấn luyện LightGBM model"""
        
        logger.info(f"Training {self.name} model...")
        
        self.validate_inputs(X_train, y_train)
        
        # Khởi tạo model
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=-1,
            **kwargs
        )
        
        # Chuẩn bị evaluation set
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            
        # Training
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        self.is_trained = True
        
        # Lưu feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Đánh giá
        train_pred = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
        
        results = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            
            results.update({
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            })
            
        self.training_history = results
        logger.info(f"Training completed. Train MAE: {train_mae:.4f}")
        
        return results
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Dự báo với LightGBM"""
        if not self.is_trained:
            raise ValueError("Model chưa được trained")
            
        return self.model.predict(X)

class SVRModel(BaseModel):
    """Support Vector Regression Model"""
    
    def __init__(self, **kwargs):
        super().__init__("SVR", **kwargs)
        
        # Default parameters
        self.C = kwargs.get('C', 1.0)
        self.gamma = kwargs.get('gamma', 'scale')
        self.kernel = kwargs.get('kernel', 'rbf')
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Huấn luyện SVR model"""
        
        logger.info(f"Training {self.name} model...")
        
        self.validate_inputs(X_train, y_train)
        
        # Khởi tạo model
        self.model = SVR(
            C=self.C,
            gamma=self.gamma,
            kernel=self.kernel,
            **kwargs
        )
        
        # Training
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Đánh giá
        train_pred = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
        
        results = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            
            results.update({
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            })
            
        self.training_history = results
        logger.info(f"Training completed. Train MAE: {train_mae:.4f}")
        
        return results
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Dự báo với SVR"""
        if not self.is_trained:
            raise ValueError("Model chưa được trained")
            
        return self.model.predict(X)

class TraditionalModels:
    """Factory class cho traditional ML models"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """
        Tạo model theo type
        
        Args:
            model_type: Loại model ('rf', 'xgb', 'lgb', 'svr')
            **kwargs: Parameters cho model
            
        Returns:
            BaseModel instance
        """
        model_map = {
            'rf': RandomForestModel,
            'random_forest': RandomForestModel,
            'xgb': XGBoostModel,
            'xgboost': XGBoostModel,
            'lgb': LightGBMModel,
            'lightgbm': LightGBMModel,
            'svr': SVRModel,
            'svm': SVRModel
        }
        
        if model_type.lower() not in model_map:
            raise ValueError(f"Model type không được hỗ trợ: {model_type}")
            
        return model_map[model_type.lower()](**kwargs)
        
    @staticmethod
    def get_available_models() -> list:
        """Lấy danh sách models có sẵn"""
        return ['RandomForest', 'XGBoost', 'LightGBM', 'SVR']
