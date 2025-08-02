"""
Model Evaluator cho đánh giá performance và metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    explained_variance_score,
    max_error
)
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class ModelEvaluator:
    """Class để đánh giá performance của models"""
    
    def __init__(self):
        self.results = {}
        
    def calculate_regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Unknown"
    ) -> Dict[str, float]:
        """
        Tính toán các regression metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Tên model
            
        Returns:
            Dict chứa các metrics
        """
        try:
            # Ensure arrays are numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Loại bỏ NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                logger.warning(f"Không có dữ liệu valid để đánh giá {model_name}")
                return {
                    'mae': float('inf'),
                    'rmse': float('inf'),
                    'mape': float('inf'),
                    'r2': -float('inf'),
                    'explained_variance': -float('inf'),
                    'max_error': float('inf'),
                    'directional_accuracy': 0.0
                }
            
            # Basic metrics
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_clean, y_pred_clean)
            explained_var = explained_variance_score(y_true_clean, y_pred_clean)
            max_err = max_error(y_true_clean, y_pred_clean)
            
            # MAPE (Mean Absolute Percentage Error)
            # Tránh division by zero
            mask_nonzero = y_true_clean != 0
            if np.sum(mask_nonzero) > 0:
                mape = np.mean(np.abs((y_true_clean[mask_nonzero] - y_pred_clean[mask_nonzero]) / y_true_clean[mask_nonzero])) * 100
            else:
                mape = float('inf')
                
            # Directional Accuracy (cho time series)
            if len(y_true_clean) > 1:
                true_direction = np.diff(y_true_clean) > 0
                pred_direction = np.diff(y_pred_clean) > 0
                directional_accuracy = np.mean(true_direction == pred_direction)
            else:
                directional_accuracy = 0.0
                
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'explained_variance': explained_var,
                'max_error': max_err,
                'directional_accuracy': directional_accuracy,
                'num_samples': len(y_true_clean)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Lỗi khi tính metrics cho {model_name}: {str(e)}")
            return {
                'mae': float('inf'),
                'rmse': float('inf'),
                'mape': float('inf'),
                'r2': -float('inf'),
                'explained_variance': -float('inf'),
                'max_error': float('inf'),
                'directional_accuracy': 0.0,
                'error': str(e)
            }
            
    def calculate_trading_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        initial_investment: float = 10000
    ) -> Dict[str, float]:
        """
        Tính toán trading-specific metrics
        
        Args:
            y_true: Actual prices
            y_pred: Predicted prices
            initial_investment: Số tiền đầu tư ban đầu
            
        Returns:
            Dict chứa trading metrics
        """
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Tính returns
            true_returns = np.diff(y_true) / y_true[:-1]
            pred_returns = np.diff(y_pred) / y_pred[:-1]
            
            # Trading signals (1 for buy, -1 for sell, 0 for hold)
            signals = np.sign(pred_returns)
            
            # Strategy returns (nếu signal > 0 thì long, < 0 thì short)
            strategy_returns = signals * true_returns
            
            # Cumulative returns
            cumulative_returns = np.cumprod(1 + strategy_returns)
            total_return = cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0
            
            # Buy and hold return
            buy_hold_return = (y_true[-1] - y_true[0]) / y_true[0]
            
            # Sharpe ratio (simplified, assuming risk-free rate = 0)
            sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
            
            # Maximum drawdown
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / peak
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            # Win rate
            positive_returns = strategy_returns > 0
            win_rate = np.mean(positive_returns) if len(positive_returns) > 0 else 0
            
            # Calmar ratio
            calmar_ratio = total_return / (max_drawdown + 1e-8)
            
            return {
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': total_return - buy_hold_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'num_trades': len(strategy_returns)
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi tính trading metrics: {str(e)}")
            return {
                'total_return': 0,
                'buy_hold_return': 0,
                'excess_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 1,
                'calmar_ratio': 0,
                'win_rate': 0,
                'error': str(e)
            }
            
    def evaluate_model(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str,
        include_trading_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Đánh giá tổng thể model
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Tên model
            include_trading_metrics: Có tính trading metrics không
            
        Returns:
            Dict chứa tất cả metrics
        """
        logger.info(f"Đánh giá model {model_name}...")
        
        # Regression metrics
        regression_metrics = self.calculate_regression_metrics(y_true, y_pred, model_name)
        
        results = {
            'model_name': model_name,
            'regression_metrics': regression_metrics
        }
        
        # Trading metrics
        if include_trading_metrics:
            trading_metrics = self.calculate_trading_metrics(y_true, y_pred)
            results['trading_metrics'] = trading_metrics
            
        # Lưu kết quả
        self.results[model_name] = results
        
        return results
        
    def compare_models(self, results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        So sánh multiple models
        
        Args:
            results_dict: Dict với key là model name, value là evaluation results
            
        Returns:
            DataFrame so sánh models
        """
        comparison_data = []
        
        for model_name, results in results_dict.items():
            row = {'Model': model_name}
            
            # Regression metrics
            if 'regression_metrics' in results:
                reg_metrics = results['regression_metrics']
                row.update({
                    'MAE': reg_metrics.get('mae', float('inf')),
                    'RMSE': reg_metrics.get('rmse', float('inf')),
                    'MAPE (%)': reg_metrics.get('mape', float('inf')),
                    'R²': reg_metrics.get('r2', -float('inf')),
                    'Directional Accuracy': reg_metrics.get('directional_accuracy', 0)
                })
                
            # Trading metrics
            if 'trading_metrics' in results:
                trade_metrics = results['trading_metrics']
                row.update({
                    'Total Return (%)': trade_metrics.get('total_return', 0) * 100,
                    'Sharpe Ratio': trade_metrics.get('sharpe_ratio', 0),
                    'Max Drawdown (%)': trade_metrics.get('max_drawdown', 1) * 100,
                    'Win Rate (%)': trade_metrics.get('win_rate', 0) * 100
                })
                
            comparison_data.append(row)
            
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sắp xếp theo MAE (ascending)
        if 'MAE' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('MAE')
            
        return comparison_df
        
    def get_best_model(
        self, 
        results_dict: Dict[str, Dict[str, Any]], 
        metric: str = 'mae'
    ) -> Tuple[str, float]:
        """
        Tìm model tốt nhất theo metric
        
        Args:
            results_dict: Dict evaluation results
            metric: Metric để so sánh ('mae', 'rmse', 'r2', 'sharpe_ratio', etc.)
            
        Returns:
            Tuple (best_model_name, best_score)
        """
        best_model = None
        best_score = None
        
        # Xác định hướng optimization (lower is better vs higher is better)
        lower_is_better = metric.lower() in ['mae', 'rmse', 'mape', 'max_drawdown']
        
        for model_name, results in results_dict.items():
            score = None
            
            # Tìm metric trong regression_metrics
            if 'regression_metrics' in results:
                score = results['regression_metrics'].get(metric.lower())
                
            # Tìm metric trong trading_metrics
            if score is None and 'trading_metrics' in results:
                score = results['trading_metrics'].get(metric.lower())
                
            if score is not None and not np.isnan(score) and not np.isinf(score):
                if best_score is None:
                    best_model = model_name
                    best_score = score
                elif (lower_is_better and score < best_score) or (not lower_is_better and score > best_score):
                    best_model = model_name
                    best_score = score
                    
        return best_model, best_score
        
    def calculate_ensemble_diversity(
        self, 
        predictions_dict: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Tính toán diversity giữa các models trong ensemble
        
        Args:
            predictions_dict: Dict với key là model name, value là predictions
            
        Returns:
            Dict chứa diversity metrics
        """
        try:
            model_names = list(predictions_dict.keys())
            predictions = np.array([predictions_dict[name] for name in model_names])
            
            # Correlation matrix
            corr_matrix = np.corrcoef(predictions)
            
            # Mean correlation (excluding diagonal)
            n_models = len(model_names)
            if n_models > 1:
                mask = ~np.eye(n_models, dtype=bool)
                mean_correlation = np.mean(corr_matrix[mask])
                
                # Diversity = 1 - mean_correlation
                diversity = 1 - mean_correlation
                
                # Pairwise disagreement
                disagreement_scores = []
                for i in range(n_models):
                    for j in range(i + 1, n_models):
                        disagreement = np.mean(np.abs(predictions[i] - predictions[j]))
                        disagreement_scores.append(disagreement)
                        
                mean_disagreement = np.mean(disagreement_scores)
                
            else:
                mean_correlation = 1.0
                diversity = 0.0
                mean_disagreement = 0.0
                
            return {
                'mean_correlation': mean_correlation,
                'diversity': diversity,
                'mean_disagreement': mean_disagreement,
                'num_models': n_models
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi tính diversity: {str(e)}")
            return {
                'mean_correlation': 0,
                'diversity': 0,
                'mean_disagreement': 0,
                'error': str(e)
            }
            
    def create_performance_summary(self, results_dict: Dict[str, Dict[str, Any]]) -> str:
        """
        Tạo summary text về performance của models
        
        Args:
            results_dict: Dict evaluation results
            
        Returns:
            String summary
        """
        if not results_dict:
            return "Không có kết quả đánh giá nào."
            
        summary = ["=== MODEL PERFORMANCE SUMMARY ===\n"]
        
        # Comparison table
        comparison_df = self.compare_models(results_dict)
        summary.append("Model Comparison:")
        summary.append(comparison_df.to_string(index=False))
        summary.append("\n")
        
        # Best models
        best_mae_model, best_mae = self.get_best_model(results_dict, 'mae')
        best_r2_model, best_r2 = self.get_best_model(results_dict, 'r2')
        
        summary.append(f"Best Model (MAE): {best_mae_model} (MAE: {best_mae:.4f})")
        summary.append(f"Best Model (R²): {best_r2_model} (R²: {best_r2:.4f})")
        
        # Trading performance
        if any('trading_metrics' in results for results in results_dict.values()):
            best_sharpe_model, best_sharpe = self.get_best_model(results_dict, 'sharpe_ratio')
            summary.append(f"Best Model (Sharpe): {best_sharpe_model} (Sharpe: {best_sharpe:.4f})")
            
        return "\n".join(summary)
        
    def save_results(self, filepath: str) -> None:
        """Lưu evaluation results ra file"""
        try:
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for model_name, results in self.results.items():
                serializable_results[model_name] = self._make_serializable(results)
                
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
            logger.info(f"Đã lưu evaluation results ra {filepath}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu results: {str(e)}")
            
    def _make_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
