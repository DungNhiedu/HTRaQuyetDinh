"""
Stock Market Index Prediction using Fusion of Machine Learning Techniques

A comprehensive package for predicting stock market indices using ensemble 
of machine learning models including traditional ML, deep learning, and 
time series methods.
"""

__version__ = "1.0.0"
__author__ = "Duong Thi Ngoc Dung"
__email__ = "dung@example.com"

from .main import StockPredictor
from .data.collector import DataCollector
from .data.preprocessor import DataPreprocessor
from .data.features import FeatureEngineer
from .models.ensemble import EnsemblePredictor
from .evaluation.metrics import ModelEvaluator

__all__ = [
    "StockPredictor",
    "DataCollector", 
    "DataPreprocessor",
    "FeatureEngineer",
    "EnsemblePredictor",
    "ModelEvaluator"
]

__all__ = [
    "StockPredictor",
    "StockDataCollector", 
    "ModelFusion"
]