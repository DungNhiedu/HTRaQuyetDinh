"""
Base Model class cho tất cả prediction models
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging
import joblib
import os

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class cho tất cả prediction models"""
    
    def __init__(self, name: str, **kwargs):
        """
        Args:
            name: Tên model
            **kwargs: Các parameters khác
        """
        self.name = name
        self.model = None
        self.is_trained = False
        self.training_params = kwargs
        self.feature_importance = None
        self.training_history = {}
        
    @abstractmethod
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Huấn luyện model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dict chứa training metrics và thông tin
        """
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Dự báo
        
        Args:
            X: Features để dự báo
            
        Returns:
            Array predictions
        """
        pass
        
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Dự báo xác suất (nếu model hỗ trợ)
        
        Args:
            X: Features để dự báo
            
        Returns:
            Array probabilities hoặc None
        """
        logger.warning(f"Model {self.name} không hỗ trợ predict_proba")
        return None
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Lấy feature importance
        
        Returns:
            Array feature importance hoặc None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        else:
            logger.warning(f"Model {self.name} không hỗ trợ feature importance")
            return None
            
    def save_model(self, filepath: str) -> None:
        """
        Lưu model ra file
        
        Args:
            filepath: Đường dẫn lưu file
        """
        try:
            # Tạo directory nếu chưa có
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Lưu model và metadata
            model_data = {
                'model': self.model,
                'name': self.name,
                'is_trained': self.is_trained,
                'training_params': self.training_params,
                'feature_importance': self.feature_importance,
                'training_history': self.training_history
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Đã lưu model {self.name} ra {filepath}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu model {self.name}: {str(e)}")
            raise
            
    def load_model(self, filepath: str) -> None:
        """
        Load model từ file
        
        Args:
            filepath: Đường dẫn file
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.name = model_data['name']
            self.is_trained = model_data['is_trained']
            self.training_params = model_data['training_params']
            self.feature_importance = model_data.get('feature_importance')
            self.training_history = model_data.get('training_history', {})
            
            logger.info(f"Đã load model {self.name} từ {filepath}")
            
        except Exception as e:
            logger.error(f"Lỗi khi load model từ {filepath}: {str(e)}")
            raise
            
    def get_params(self) -> Dict[str, Any]:
        """
        Lấy parameters của model
        
        Returns:
            Dict parameters
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        else:
            return self.training_params
            
    def set_params(self, **params) -> None:
        """
        Set parameters cho model
        
        Args:
            **params: Parameters để set
        """
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        
        # Update training params
        self.training_params.update(params)
        
    def validate_inputs(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Validate input data
        
        Args:
            X: Features
            y: Targets (optional)
        """
        if X.empty:
            raise ValueError("Features DataFrame trống")
            
        if X.isnull().any().any():
            logger.warning("Features chứa NaN values")
            
        if y is not None:
            if len(X) != len(y):
                raise ValueError("Số lượng features và targets không khớp")
                
            if y.isnull().any():
                logger.warning("Targets chứa NaN values")
                
    def reset(self) -> None:
        """Reset model về trạng thái ban đầu"""
        self.model = None
        self.is_trained = False
        self.feature_importance = None
        self.training_history = {}
        logger.info(f"Đã reset model {self.name}")
        
    def clone(self) -> 'BaseModel':
        """
        Tạo bản copy của model
        
        Returns:
            Model copy
        """
        # Tạo instance mới cùng class
        cloned = self.__class__(self.name + "_copy", **self.training_params)
        return cloned
        
    def __str__(self) -> str:
        """String representation"""
        status = "trained" if self.is_trained else "not trained"
        return f"{self.__class__.__name__}(name={self.name}, status={status})"
        
    def __repr__(self) -> str:
        """Detailed representation"""
        return self.__str__()
