"""
Deep Learning Models cho time series prediction
Currently disabled due to TensorFlow compatibility issues with Python 3.13
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """LSTM Model placeholder - TensorFlow not available in Python 3.13"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model_type = 'LSTM'
        logger.warning("LSTM Model is disabled due to TensorFlow compatibility issues with Python 3.13")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Placeholder fit method"""
        raise NotImplementedError("LSTM Model is not available due to TensorFlow compatibility issues")
    
    def predict(self, X):
        """Placeholder predict method"""
        raise NotImplementedError("LSTM Model is not available due to TensorFlow compatibility issues")


class GRUModel(BaseModel):
    """GRU Model placeholder - TensorFlow not available in Python 3.13"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model_type = 'GRU'
        logger.warning("GRU Model is disabled due to TensorFlow compatibility issues with Python 3.13")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Placeholder fit method"""
        raise NotImplementedError("GRU Model is not available due to TensorFlow compatibility issues")
    
    def predict(self, X):
        """Placeholder predict method"""
        raise NotImplementedError("GRU Model is not available due to TensorFlow compatibility issues")


class DeepLearningModels:
    """Deep Learning Models container class"""
    
    def __init__(self):
        """Initialize DeepLearningModels"""
        self.models = {
            'LSTM': LSTMModel,
            'GRU': GRUModel
        }
        logger.warning("Deep Learning models are disabled due to TensorFlow compatibility issues with Python 3.13")
    
    def get_model(self, model_name: str, **kwargs):
        """Get a specific deep learning model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Available models: {list(self.models.keys())}")
        
        return self.models[model_name](**kwargs)
    
    def get_available_models(self):
        """Get list of available models"""
        return list(self.models.keys())
