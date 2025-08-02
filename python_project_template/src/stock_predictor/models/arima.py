"""
ARIMA Model cho time series forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

from .base_model import BaseModel

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class ARIMAModel(BaseModel):
    """ARIMA Model cho time series forecasting"""
    
    def __init__(self, **kwargs):
        super().__init__("ARIMA", **kwargs)
        
        # ARIMA parameters (p, d, q)
        self.order = kwargs.get('order', (1, 1, 1))  # Default ARIMA(1,1,1)
        self.seasonal_order = kwargs.get('seasonal_order', None)
        self.auto_order = kwargs.get('auto_order', True)  # Tự động tìm order tốt nhất
        
    def check_stationarity(self, series: pd.Series) -> Tuple[bool, float]:
        """
        Kiểm tra tính dừng của time series
        
        Args:
            series: Time series data
            
        Returns:
            Tuple (is_stationary, p_value)
        """
        try:
            result = adfuller(series.dropna())
            p_value = result[1]
            is_stationary = p_value < 0.05
            
            logger.info(f"ADF test p-value: {p_value:.4f}, Stationary: {is_stationary}")
            return is_stationary, p_value
            
        except Exception as e:
            logger.warning(f"Lỗi khi test stationarity: {str(e)}")
            return False, 1.0
            
    def find_best_order(self, series: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[int, int, int]:
        """
        Tự động tìm order tốt nhất cho ARIMA
        
        Args:
            series: Time series data
            max_p: Maximum p value to test
            max_d: Maximum d value to test  
            max_q: Maximum q value to test
            
        Returns:
            Tuple (best_p, best_d, best_q)
        """
        logger.info("Tìm ARIMA order tốt nhất...")
        
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            
                    except Exception:
                        continue
                        
        logger.info(f"Best ARIMA order: {best_order} with AIC: {best_aic:.4f}")
        return best_order
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Huấn luyện ARIMA model
        
        Note: ARIMA chỉ sử dụng y_train (target series), không cần X_train
        """
        logger.info(f"Training {self.name} model...")
        
        # ARIMA chỉ cần target series
        series = y_train.copy()
        
        # Kiểm tra stationarity
        is_stationary, p_value = self.check_stationarity(series)
        
        # Tự động tìm order nếu được yêu cầu
        if self.auto_order:
            self.order = self.find_best_order(series)
            
        try:
            # Khởi tạo và fit ARIMA model
            arima_model = ARIMA(series, order=self.order, seasonal_order=self.seasonal_order)
            self.model = arima_model.fit()
            self.is_trained = True
            
            # Đánh giá trên training set
            train_pred = self.model.fittedvalues
            
            # Đảm bảo cùng length
            min_len = min(len(series), len(train_pred))
            y_true = series.iloc[-min_len:]
            y_pred = train_pred.iloc[-min_len:]
            
            train_mae = mean_absolute_error(y_true, y_pred)
            train_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            train_r2 = r2_score(y_true, y_pred)
            
            results = {
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'aic': self.model.aic,
                'bic': self.model.bic,
                'order': self.order,
                'is_stationary': is_stationary,
                'stationarity_p_value': p_value
            }
            
            # Validation evaluation nếu có
            if y_val is not None:
                # Dự báo cho validation period
                forecast_steps = len(y_val)
                val_forecast = self.model.forecast(steps=forecast_steps)
                
                val_mae = mean_absolute_error(y_val, val_forecast)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_forecast))
                val_r2 = r2_score(y_val, val_forecast)
                
                results.update({
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2
                })
                
            self.training_history = results
            logger.info(f"Training completed. AIC: {self.model.aic:.4f}, Train MAE: {train_mae:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Lỗi khi training ARIMA model: {str(e)}")
            # Fallback: thử với order đơn giản hơn
            try:
                logger.info("Thử với ARIMA(1,1,1)...")
                simple_model = ARIMA(series, order=(1, 1, 1))
                self.model = simple_model.fit()
                self.is_trained = True
                self.order = (1, 1, 1)
                
                # Basic evaluation
                train_pred = self.model.fittedvalues
                min_len = min(len(series), len(train_pred))
                y_true = series.iloc[-min_len:]
                y_pred = train_pred.iloc[-min_len:]
                
                train_mae = mean_absolute_error(y_true, y_pred)
                
                return {
                    'train_mae': train_mae,
                    'aic': self.model.aic,
                    'order': self.order,
                    'fallback': True
                }
                
            except Exception as e2:
                logger.error(f"Fallback ARIMA cũng failed: {str(e2)}")
                raise e
                
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Dự báo với ARIMA
        
        Note: X được sử dụng chỉ để xác định số steps cần forecast
        """
        if not self.is_trained:
            raise ValueError("Model chưa được trained")
            
        try:
            # Số steps để forecast
            steps = len(X)
            
            # Forecast
            forecast = self.model.forecast(steps=steps)
            
            return np.array(forecast)
            
        except Exception as e:
            logger.error(f"Lỗi khi predict với ARIMA: {str(e)}")
            # Fallback: return constant value
            return np.full(len(X), self.model.fittedvalues.iloc[-1] if hasattr(self.model, 'fittedvalues') else 0)
            
    def predict_with_confidence(self, steps: int, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Dự báo với confidence intervals
        
        Args:
            steps: Số steps để forecast
            alpha: Significance level (default 0.05 for 95% CI)
            
        Returns:
            Tuple (forecast, lower_ci, upper_ci)
        """
        if not self.is_trained:
            raise ValueError("Model chưa được trained")
            
        try:
            forecast_result = self.model.get_forecast(steps=steps, alpha=alpha)
            
            forecast = forecast_result.predicted_mean.values
            ci = forecast_result.conf_int()
            lower_ci = ci.iloc[:, 0].values
            upper_ci = ci.iloc[:, 1].values
            
            return forecast, lower_ci, upper_ci
            
        except Exception as e:
            logger.error(f"Lỗi khi predict with confidence: {str(e)}")
            # Fallback
            fallback_forecast = np.full(steps, self.model.fittedvalues.iloc[-1])
            return fallback_forecast, fallback_forecast, fallback_forecast
            
    def get_residuals(self) -> pd.Series:
        """Lấy residuals từ fitted model"""
        if not self.is_trained:
            raise ValueError("Model chưa được trained")
            
        return self.model.resid
        
    def plot_diagnostics(self) -> None:
        """Plot diagnostic plots cho ARIMA model"""
        if not self.is_trained:
            raise ValueError("Model chưa được trained")
            
        try:
            import matplotlib.pyplot as plt
            
            # Plot diagnostic charts
            self.model.plot_diagnostics(figsize=(12, 8))
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib không có sẵn cho plotting")
        except Exception as e:
            logger.warning(f"Lỗi khi plot diagnostics: {str(e)}")
            
    def get_model_summary(self) -> str:
        """Lấy summary của ARIMA model"""
        if not self.is_trained:
            return "Model chưa được trained"
            
        try:
            return str(self.model.summary())
        except Exception as e:
            return f"Lỗi khi lấy summary: {str(e)}"
