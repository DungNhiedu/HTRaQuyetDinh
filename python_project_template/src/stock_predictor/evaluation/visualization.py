"""
Visualization cho plotting charts và graphs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class Visualizer:
    """Class để tạo visualizations cho stock market prediction"""
    
    def __init__(self, style: str = 'plotly'):
        """
        Args:
            style: Style cho plots ('plotly', 'matplotlib', 'seaborn')
        """
        self.style = style
        
        # Set style
        if style == 'seaborn':
            sns.set_style("whitegrid")
        elif style == 'matplotlib':
            plt.style.use('seaborn-v0_8')
            
    def plot_stock_data(
        self, 
        data: pd.DataFrame, 
        title: str = "Stock Price Data",
        show_volume: bool = True
    ) -> go.Figure:
        """
        Plot stock price data với candlestick chart
        
        Args:
            data: DataFrame với OHLCV data
            title: Title cho chart
            show_volume: Có hiển thị volume không
            
        Returns:
            Plotly Figure
        """
        try:
            # Create subplots
            rows = 2 if show_volume else 1
            fig = make_subplots(
                rows=rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(title, "Volume") if show_volume else (title,),
                row_width=[0.7, 0.3] if show_volume else [1.0]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Volume chart
            if show_volume and 'Volume' in data.columns:
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name='Volume',
                        marker_color='rgba(158,202,225,0.8)'
                    ),
                    row=2, col=1
                )
                
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Price",
                height=600 if show_volume else 400,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Lỗi khi plot stock data: {str(e)}")
            # Return empty figure
            return go.Figure()
            
    def plot_predictions(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Actual vs Predicted",
        model_name: str = "Model"
    ) -> go.Figure:
        """
        Plot actual vs predicted values
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Dates for x-axis
            title: Chart title
            model_name: Model name for legend
            
        Returns:
            Plotly Figure
        """
        try:
            fig = go.Figure()
            
            x_axis = dates if dates is not None else list(range(len(y_true)))
            
            # Actual values
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=y_true,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Predicted values
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=y_pred,
                    mode='lines',
                    name=f'Predicted ({model_name})',
                    line=dict(color='red', width=2, dash='dash')
                )
            )
            
            fig.update_layout(
                title=title,
                xaxis_title="Date" if dates is not None else "Time",
                yaxis_title="Value",
                height=500,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Lỗi khi plot predictions: {str(e)}")
            return go.Figure()
            
    def plot_multiple_predictions(
        self, 
        y_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Model Comparison"
    ) -> go.Figure:
        """
        Plot nhiều model predictions
        
        Args:
            y_true: Actual values
            predictions_dict: Dict với key là model name, value là predictions
            dates: Dates for x-axis
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        try:
            fig = go.Figure()
            
            x_axis = dates if dates is not None else list(range(len(y_true)))
            
            # Actual values
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=y_true,
                    mode='lines',
                    name='Actual',
                    line=dict(color='black', width=3)
                )
            )
            
            # Model predictions
            colors = px.colors.qualitative.Set1
            for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=y_pred,
                        mode='lines',
                        name=model_name,
                        line=dict(color=color, width=2)
                    )
                )
                
            fig.update_layout(
                title=title,
                xaxis_title="Date" if dates is not None else "Time",
                yaxis_title="Value",
                height=600,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Lỗi khi plot multiple predictions: {str(e)}")
            return go.Figure()
            
    def plot_feature_importance(
        self, 
        importance_data: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance"
    ) -> go.Figure:
        """
        Plot feature importance
        
        Args:
            importance_data: DataFrame với columns ['feature', 'importance']
            top_n: Số lượng top features để hiển thị
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        try:
            # Lấy top N features
            top_features = importance_data.head(top_n)
            
            fig = go.Figure(
                go.Bar(
                    x=top_features['importance'],
                    y=top_features['feature'],
                    orientation='h',
                    marker_color='skyblue'
                )
            )
            
            fig.update_layout(
                title=title,
                xaxis_title="Importance",
                yaxis_title="Features",
                height=max(400, top_n * 20),  # Dynamic height based on number of features
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Lỗi khi plot feature importance: {str(e)}")
            return go.Figure()
            
    def plot_model_comparison(
        self, 
        comparison_df: pd.DataFrame,
        metric: str = 'MAE'
    ) -> go.Figure:
        """
        Plot so sánh performance các models
        
        Args:
            comparison_df: DataFrame từ ModelEvaluator.compare_models()
            metric: Metric để plot
            
        Returns:
            Plotly Figure
        """
        try:
            if metric not in comparison_df.columns:
                available_metrics = [col for col in comparison_df.columns if col != 'Model']
                if available_metrics:
                    metric = available_metrics[0]
                else:
                    logger.warning("Không có metric nào để plot")
                    return go.Figure()
                    
            fig = go.Figure(
                go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    marker_color='lightcoral'
                )
            )
            
            fig.update_layout(
                title=f"Model Comparison - {metric}",
                xaxis_title="Model",
                yaxis_title=metric,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Lỗi khi plot model comparison: {str(e)}")
            return go.Figure()
            
    def plot_residuals(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        title: str = "Residuals Plot"
    ) -> go.Figure:
        """
        Plot residuals analysis
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        try:
            residuals = y_true - y_pred
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Residuals vs Fitted", 
                    "Residuals Distribution",
                    "Q-Q Plot", 
                    "Residuals vs Time"
                )
            )
            
            # Residuals vs Fitted
            fig.add_trace(
                go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='blue', opacity=0.6)
                ),
                row=1, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Residuals distribution
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    name='Distribution',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
            
            # Q-Q Plot (simplified)
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = np.linspace(-3, 3, len(sorted_residuals))
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_residuals,
                    mode='markers',
                    name='Q-Q',
                    marker=dict(color='green', opacity=0.6)
                ),
                row=2, col=1
            )
            
            # Residuals vs Time
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(residuals))),
                    y=residuals,
                    mode='lines+markers',
                    name='Time Series',
                    line=dict(color='purple')
                ),
                row=2, col=2
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
            
            fig.update_layout(
                title=title,
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Lỗi khi plot residuals: {str(e)}")
            return go.Figure()
            
    def plot_correlation_matrix(
        self, 
        data: pd.DataFrame,
        title: str = "Correlation Matrix"
    ) -> go.Figure:
        """
        Plot correlation matrix
        
        Args:
            data: DataFrame để tính correlation
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        try:
            # Chỉ lấy numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            corr_matrix = numeric_data.corr()
            
            fig = go.Figure(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hoverongaps=False
                )
            )
            
            fig.update_layout(
                title=title,
                height=600,
                width=800
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Lỗi khi plot correlation matrix: {str(e)}")
            return go.Figure()
            
    def plot_technical_indicators(
        self, 
        data: pd.DataFrame,
        indicators: List[str],
        title: str = "Technical Indicators"
    ) -> go.Figure:
        """
        Plot technical indicators
        
        Args:
            data: DataFrame chứa price data và indicators
            indicators: List tên indicators để plot
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        try:
            # Determine number of subplots needed
            n_indicators = len(indicators)
            rows = min(n_indicators + 1, 4)  # +1 cho price chart, max 4 rows
            
            fig = make_subplots(
                rows=rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=["Price"] + indicators[:rows-1]
            )
            
            # Price chart
            if 'Close' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
            # Indicators
            colors = px.colors.qualitative.Set1
            for i, indicator in enumerate(indicators[:rows-1]):
                if indicator in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[indicator],
                            mode='lines',
                            name=indicator,
                            line=dict(color=colors[i % len(colors)])
                        ),
                        row=i+2, col=1
                    )
                    
            fig.update_layout(
                title=title,
                height=200 * rows,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Lỗi khi plot technical indicators: {str(e)}")
            return go.Figure()
            
    def create_dashboard_layout(
        self, 
        figures: List[go.Figure],
        titles: List[str]
    ) -> go.Figure:
        """
        Tạo dashboard layout với multiple figures
        
        Args:
            figures: List các Plotly figures
            titles: List titles cho từng figure
            
        Returns:
            Combined Plotly Figure
        """
        try:
            n_figs = len(figures)
            if n_figs == 0:
                return go.Figure()
                
            # Determine grid layout
            if n_figs == 1:
                rows, cols = 1, 1
            elif n_figs <= 2:
                rows, cols = 1, 2
            elif n_figs <= 4:
                rows, cols = 2, 2
            else:
                rows, cols = 3, 2
                
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=titles[:rows*cols]
            )
            
            # Add traces from each figure
            for i, source_fig in enumerate(figures[:rows*cols]):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                for trace in source_fig.data:
                    fig.add_trace(trace, row=row, col=col)
                    
            fig.update_layout(
                height=400 * rows,
                showlegend=False,
                title="Stock Market Prediction Dashboard"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo dashboard layout: {str(e)}")
            return go.Figure()
