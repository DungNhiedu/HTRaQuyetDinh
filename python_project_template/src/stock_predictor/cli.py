"""
Command Line Interface cho Stock Market Prediction System
"""

import argparse
import sys
import os
from typing import List, Optional
import logging

from .main import StockPredictor
from .utils.helpers import setup_logging

def create_parser() -> argparse.ArgumentParser:
    """Tạo argument parser cho CLI"""
    
    parser = argparse.ArgumentParser(
        description="Market Index Prediction using Fusion of Machine Learning Techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dự báo VN-Index cho 30 ngày tới
  stock-predict predict --symbol ^VNI --days 30
  
  # Train models cho VN-Index với 1 năm dữ liệu
  stock-predict train --symbol ^VNI --period 1y --models rf xgb lstm
  
  # Chạy complete pipeline
  stock-predict run --symbol ^VNI --period 2y --ensemble voting
  
  # Evaluate models trên test set
  stock-predict evaluate --symbol ^VNI --load-models models/
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--symbol', 
        type=str, 
        default='^VNI',
        help='Stock symbol to predict (default: ^VNI)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument(
        '--days', 
        type=int, 
        default=30,
        help='Number of days to predict (default: 30)'
    )
    predict_parser.add_argument(
        '--model',
        type=str,
        choices=['rf', 'xgb', 'lgb', 'svr', 'lstm', 'gru', 'arima', 'ensemble'],
        default='ensemble',
        help='Model to use for prediction (default: ensemble)'
    )
    predict_parser.add_argument(
        '--load-model',
        type=str,
        help='Path to load trained model'
    )
    predict_parser.add_argument(
        '--output',
        type=str,
        help='Output file for predictions'
    )
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument(
        '--period',
        type=str,
        default='2y',
        help='Data period to use (default: 2y)'
    )
    train_parser.add_argument(
        '--models',
        nargs='+',
        choices=['rf', 'xgb', 'lgb', 'svr', 'lstm', 'gru', 'arima'],
        default=['rf', 'xgb', 'lstm'],
        help='Models to train (default: rf xgb lstm)'
    )
    train_parser.add_argument(
        '--ensemble',
        type=str,
        choices=['voting', 'stacking', 'weighted', 'bayesian'],
        default='voting',
        help='Ensemble method (default: voting)'
    )
    train_parser.add_argument(
        '--save-models',
        type=str,
        help='Directory to save trained models'
    )
    
    # Run command (complete pipeline)
    run_parser = subparsers.add_parser('run', help='Run complete pipeline')
    run_parser.add_argument(
        '--period',
        type=str,
        default='2y',
        help='Data period to use (default: 2y)'
    )
    run_parser.add_argument(
        '--models',
        nargs='+',
        choices=['rf', 'xgb', 'lgb', 'svr', 'lstm', 'gru', 'arima'],
        help='Models to train (default: all available)'
    )
    run_parser.add_argument(
        '--ensemble',
        type=str,
        choices=['voting', 'stacking', 'weighted', 'bayesian'],
        default='voting',
        help='Ensemble method (default: voting)'
    )
    run_parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    run_parser.add_argument(
        '--predict-days',
        type=int,
        default=30,
        help='Days to predict into future (default: 30)'
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    evaluate_parser.add_argument(
        '--load-models',
        type=str,
        required=True,
        help='Directory containing trained models'
    )
    evaluate_parser.add_argument(
        '--test-period',
        type=str,
        default='6mo',
        help='Period for test data (default: 6mo)'
    )
    evaluate_parser.add_argument(
        '--output',
        type=str,
        help='Output file for evaluation results'
    )
    
    return parser

def predict_command(args) -> None:
    """Handle predict command"""
    print(f"🔮 Making predictions for {args.symbol}...")
    
    try:
        # Initialize predictor
        predictor = StockPredictor(symbol=args.symbol)
        
        if args.load_model:
            # Load pre-trained model
            print(f"Loading model from {args.load_model}...")
            # Implementation depends on model format
            pass
        else:
            # Quick train for demo
            print("Training quick model for prediction...")
            predictor.load_data(period='1y')
            predictor.prepare_features()
            predictor.split_data()
            predictor.train_individual_models(['rf'])  # Quick model
            
        # Make predictions
        print(f"Predicting {args.days} days into the future...")
        future_predictions = predictor.predict_future(days=args.days)
        
        # Display results
        print("\n📊 Prediction Results:")
        for model_name, predictions in future_predictions.items():
            avg_prediction = predictions.mean()
            print(f"{model_name}: Average predicted value = {avg_prediction:.2f}")
            
        # Save if requested
        if args.output:
            import pandas as pd
            pred_df = pd.DataFrame(future_predictions)
            pred_df.to_csv(args.output)
            print(f"Predictions saved to {args.output}")
            
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        sys.exit(1)

def train_command(args) -> None:
    """Handle train command"""
    print(f"🏋️ Training models for {args.symbol}...")
    
    try:
        # Initialize predictor
        predictor = StockPredictor(symbol=args.symbol)
        
        # Load and prepare data
        print(f"Loading {args.period} of data...")
        predictor.load_data(period=args.period)
        predictor.prepare_features()
        predictor.split_data()
        
        # Train individual models
        print(f"Training models: {', '.join(args.models)}")
        training_results = predictor.train_individual_models(args.models)
        
        # Train ensemble
        print(f"Training ensemble with {args.ensemble} method...")
        ensemble_results = predictor.train_ensemble(args.ensemble)
        
        # Evaluate
        print("Evaluating models...")
        evaluation_results = predictor.evaluate_models()
        
        # Display results
        print("\n📈 Training Results:")
        for model_name, results in evaluation_results.items():
            if 'regression_metrics' in results:
                metrics = results['regression_metrics']
                mae = metrics.get('mae', 'N/A')
                r2 = metrics.get('r2', 'N/A')
                print(f"{model_name}: MAE = {mae:.4f}, R² = {r2:.4f}")
                
        # Save models if requested
        if args.save_models:
            print(f"Saving models to {args.save_models}...")
            predictor.save_results(args.save_models)
            
    except Exception as e:
        print(f"❌ Error in training: {str(e)}")
        sys.exit(1)

def run_command(args) -> None:
    """Handle run command (complete pipeline)"""
    print(f"🚀 Running complete pipeline for {args.symbol}...")
    
    try:
        # Initialize predictor
        predictor = StockPredictor(symbol=args.symbol)
        
        # Run complete pipeline
        print("Running complete pipeline...")
        results = predictor.run_complete_pipeline(
            period=args.period,
            model_types=args.models,
            ensemble_method=args.ensemble
        )
        
        # Display summary
        print("\n📊 Pipeline Results Summary:")
        print(f"Symbol: {args.symbol}")
        print(f"Data period: {args.period}")
        print(f"Total samples: {results['data_info']['total_samples']}")
        print(f"Features: {results['data_info']['features_count']}")
        print(f"Best model: {results['best_model']}")
        
        # Show model performance
        print("\n📈 Model Performance:")
        for model_name, results_detail in results['evaluation_results'].items():
            if 'regression_metrics' in results_detail:
                metrics = results_detail['regression_metrics']
                mae = metrics.get('mae', 'N/A')
                r2 = metrics.get('r2', 'N/A')
                print(f"  {model_name}: MAE = {mae:.4f}, R² = {r2:.4f}")
                
        # Future predictions
        if args.predict_days > 0:
            print(f"\n🔮 Future Predictions ({args.predict_days} days):")
            future_preds = predictor.predict_future(days=args.predict_days)
            for model_name, predictions in future_preds.items():
                avg_pred = predictions.mean()
                print(f"  {model_name}: Average = {avg_pred:.2f}")
                
        # Save results
        print(f"\nSaving results to {args.output_dir}...")
        predictor.save_results(args.output_dir)
        
        # Generate report
        report = predictor.create_report()
        report_file = os.path.join(args.output_dir, 'report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
            
        print(f"✅ Pipeline completed successfully!")
        print(f"📁 Results saved in: {args.output_dir}")
        print(f"📄 Report: {report_file}")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        sys.exit(1)

def evaluate_command(args) -> None:
    """Handle evaluate command"""
    print(f"📊 Evaluating models for {args.symbol}...")
    
    try:
        # Initialize predictor
        predictor = StockPredictor(symbol=args.symbol)
        
        # Load test data
        print(f"Loading test data ({args.test_period})...")
        predictor.load_data(period=args.test_period)
        predictor.prepare_features()
        predictor.split_data()
        
        # Load models
        print(f"Loading models from {args.load_models}...")
        # Implementation for loading models
        
        # Evaluate
        print("Evaluating models...")
        evaluation_results = predictor.evaluate_models()
        
        # Display results
        print("\n📈 Evaluation Results:")
        for model_name, results in evaluation_results.items():
            if 'regression_metrics' in results:
                metrics = results['regression_metrics']
                mae = metrics.get('mae', 'N/A')
                rmse = metrics.get('rmse', 'N/A')
                r2 = metrics.get('r2', 'N/A')
                print(f"{model_name}:")
                print(f"  MAE: {mae:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  R²: {r2:.4f}")
                
        # Save results if requested
        if args.output:
            predictor.evaluator.save_results(args.output)
            print(f"Results saved to {args.output}")
            
    except Exception as e:
        print(f"❌ Error in evaluation: {str(e)}")
        sys.exit(1)

def main() -> None:
    """Main entry point cho CLI"""
    
    # Create parser
    parser = create_parser()
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(level=log_level, log_file=args.log_file)
    
    # Print header
    print("=" * 60)
    print("📈 Market Prediction System")
    print("   Fusion of Machine Learning Techniques")
    print("=" * 60)
    
    # Route to appropriate command
    try:
        if args.command == 'predict':
            predict_command(args)
        elif args.command == 'train':
            train_command(args)
        elif args.command == 'run':
            run_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()