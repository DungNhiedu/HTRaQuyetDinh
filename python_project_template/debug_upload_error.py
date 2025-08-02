#!/usr/bin/env python3
"""
Debug the upload processing error more specifically.
"""

import sys
import os
import tempfile
import shutil
import traceback
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stock_predictor.data.preprocessor import DataPreprocessor
from stock_predictor.data.features import add_technical_indicators, FeatureEngineer

def debug_upload_processing():
    """Debug each step of upload processing to find the exact error."""
    print("🔍 Debugging upload processing step by step...")
    
    # Create a temporary directory (simulating upload)
    temp_dir = tempfile.mkdtemp()
    print(f"✅ Created temp directory: {temp_dir}")
    
    try:
        # Copy VN30 demo file to temp directory (simulating upload)
        source_file = "/Users/dungnhi/Documents/HTRaQuyetDinh/VN30_demo.csv"
        target_file = os.path.join(temp_dir, "VN30_demo.csv")
        
        shutil.copy2(source_file, target_file)
        print(f"✅ Copied file to temp directory")
        
        # Step 1: DataPreprocessor loading
        print("\n🔄 Step 1: Loading and processing with DataPreprocessor...")
        try:
            preprocessor = DataPreprocessor()
            merged_data = preprocessor.load_and_process_all(temp_dir)
            
            if merged_data.empty:
                print("❌ No data could be processed from the uploaded files")
                return False
            
            print(f"✅ Successfully processed uploaded data! Shape: {merged_data.shape}")
            print(f"✅ Columns: {list(merged_data.columns)}")
            
        except Exception as e:
            print(f"❌ Error in DataPreprocessor: {str(e)}")
            traceback.print_exc()
            return False
        
        # Step 2: Calculate time duration
        print("\n🔄 Step 2: Calculating time duration...")
        try:
            # Simulate calculate_time_duration function
            if 'date' in merged_data.columns:
                import pandas as pd
                dates = pd.to_datetime(merged_data['date'])
                start_date = dates.min()
                end_date = dates.max()
                duration = end_date - start_date
                years = duration.days / 365.25
                uploaded_time_duration = f"{years:.1f} years ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
            else:
                # Fallback: estimate from row count
                years = len(merged_data) / 365.25
                uploaded_time_duration = f"~{years:.1f} years ({len(merged_data)} data points)"
            
            print(f"✅ Time duration calculated: {uploaded_time_duration}")
            
        except Exception as e:
            print(f"❌ Error calculating time duration: {str(e)}")
            traceback.print_exc()
            return False
        
        # Step 3: Technical indicators
        print("\n🔄 Step 3: Adding technical indicators...")
        try:
            # Check required columns first
            required_ta_columns = ['close', 'code']
            missing_ta_columns = [col for col in required_ta_columns if col not in merged_data.columns]
            
            if missing_ta_columns:
                print(f"⚠️ Missing columns for technical indicators: {missing_ta_columns}")
                data_with_features = merged_data.copy()
            else:
                data_with_features = add_technical_indicators(merged_data)
                print(f"✅ Added technical indicators! Final shape: {data_with_features.shape}")
                
                # Show new columns
                original_cols = set(merged_data.columns)
                new_cols = set(data_with_features.columns) - original_cols
                print(f"✅ New technical indicator columns: {list(new_cols)}")
            
        except Exception as e:
            print(f"❌ Error adding technical indicators: {str(e)}")
            traceback.print_exc()
            data_with_features = merged_data.copy()
            print("⚠️ Using original data without technical indicators")
        
        # Step 4: Feature engineering
        print("\n🔄 Step 4: Advanced feature engineering...")
        try:
            feature_engineer = FeatureEngineer()
            
            # Try each feature engineering step separately
            print("  - Creating price features...")
            enriched_data = feature_engineer.create_price_features(data_with_features)
            print(f"    Shape after price features: {enriched_data.shape}")
            
            print("  - Creating volume features...")
            enriched_data = feature_engineer.create_volume_features(enriched_data)
            print(f"    Shape after volume features: {enriched_data.shape}")
            
            print("  - Creating lag features...")
            enriched_data = feature_engineer.create_lag_features(enriched_data)
            print(f"    Shape after lag features: {enriched_data.shape}")
            
            print("  - Creating rolling features...")
            enriched_data = feature_engineer.create_rolling_features(enriched_data)
            print(f"    Shape after rolling features: {enriched_data.shape}")
            
            print(f"✅ Feature engineering completed! Final shape: {enriched_data.shape}")
            
        except Exception as e:
            print(f"❌ Error in feature engineering: {str(e)}")
            traceback.print_exc()
            enriched_data = data_with_features.copy()
            print("⚠️ Using data with technical indicators only")
        
        # Step 5: Session state simulation
        print("\n🔄 Step 5: Session state simulation...")
        try:
            session_state = {
                'uploaded_data': merged_data,
                'uploaded_features': data_with_features, 
                'uploaded_time_duration': uploaded_time_duration
            }
            print("✅ Session state simulation successful")
            
        except Exception as e:
            print(f"❌ Error in session state simulation: {str(e)}")
            traceback.print_exc()
            return False
        
        # Step 6: AI prediction data preparation
        print("\n🔄 Step 6: AI prediction data preparation...")
        try:
            # Check session state
            if 'uploaded_data' not in session_state or session_state['uploaded_data'].empty:
                print("❌ No uploaded data found in session state")
                return False
            
            processed_data = session_state['uploaded_data']
            
            # Validate required columns
            required_columns = ['close']
            missing_columns = [col for col in required_columns if col not in processed_data.columns]
            
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                return False
            
            # Calculate latest return safely
            import pandas as pd
            if 'return' in processed_data.columns and len(processed_data) > 1:
                latest_return = processed_data['return'].iloc[-1] if not pd.isna(processed_data['return'].iloc[-1]) else 0
            else:
                # Calculate manually if return column doesn't exist
                if len(processed_data) >= 2:
                    latest_close = processed_data['close'].iloc[-1]
                    prev_close = processed_data['close'].iloc[-2]
                    latest_return = ((latest_close - prev_close) / prev_close) * 100
                else:
                    latest_return = 0
            
            # Prepare data summary
            uploaded_summary = {
                'total_days': len(processed_data),
                'time_duration': uploaded_time_duration,
                'current_price': processed_data['close'].iloc[-1] if len(processed_data) > 0 else 0,
                'latest_change': latest_return,
                'up_days_ratio': 100 * (processed_data['target'] == 1).sum() / len(processed_data) if 'target' in processed_data.columns and len(processed_data) > 0 else 50,
                'highest_price': processed_data['close'].max() if len(processed_data) > 0 else 0,
                'lowest_price': processed_data['close'].min() if len(processed_data) > 0 else 0,
                'avg_volatility': abs(processed_data['return']).mean() if 'return' in processed_data.columns and len(processed_data) > 0 else 0
            }
            
            print("✅ AI prediction data summary prepared successfully:")
            for key, value in uploaded_summary.items():
                print(f"   {key}: {value}")
                
        except Exception as e:
            print(f"❌ Error in AI prediction data preparation: {str(e)}")
            traceback.print_exc()
            return False
        
        print("\n🎉 All steps completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"✅ Cleaned up temp directory")

def main():
    """Main debug function."""
    print("🚀 Debugging upload CSV processing pipeline...\n")
    
    success = debug_upload_processing()
    
    if success:
        print("\n🎉 All steps working! The error might be somewhere else.")
    else:
        print("\n⚠️ Found the problem! Check the error details above.")

if __name__ == "__main__":
    main()
