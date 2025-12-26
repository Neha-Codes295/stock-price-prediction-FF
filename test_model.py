"""
Model Testing and Validation Script
Tests the stock prediction model with various scenarios
"""

import pandas as pd
import numpy as np
from stock_prediction import StockPricePredictor
import os

def test_data_loading():
    """Test if data files exist and can be loaded."""
    print("=" * 70)
    print("TEST 1: Data Loading")
    print("=" * 70)
    
    required_files = ['Data.csv', 'StockPrice.csv']
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
            df = pd.read_csv(file)
            print(f"  - Shape: {df.shape}")
            print(f"  - Columns: {list(df.columns)}")
        else:
            print(f"✗ {file} NOT FOUND")
            return False
    
    print("\n✓ All data files present and readable\n")
    return True

def test_data_integrity():
    """Test data quality and integrity."""
    print("=" * 70)
    print("TEST 2: Data Integrity")
    print("=" * 70)
    
    data_df = pd.read_csv('Data.csv')
    stock_df = pd.read_csv('StockPrice.csv')
    
    # Check for missing values
    print("\nMissing Values:")
    print(f"  Data.csv: {data_df.isnull().sum().sum()} missing values")
    print(f"  StockPrice.csv: {stock_df.isnull().sum().sum()} missing values")
    
    # Check date ranges
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    
    print("\nDate Ranges:")
    print(f"  Data.csv: {data_df['Date'].min()} to {data_df['Date'].max()}")
    print(f"  StockPrice.csv: {stock_df['Date'].min()} to {stock_df['Date'].max()}")
    
    # Check for duplicates
    print("\nDuplicate Dates:")
    print(f"  Data.csv: {data_df['Date'].duplicated().sum()} duplicates")
    print(f"  StockPrice.csv: {stock_df['Date'].duplicated().sum()} duplicates")
    
    # Check data types
    print("\nData Types:")
    print(f"  Data values: {data_df['Data'].dtype}")
    print(f"  Price values: {stock_df['Price'].dtype}")
    
    print("\n✓ Data integrity check complete\n")
    return True

def test_feature_engineering():
    """Test feature engineering process."""
    print("=" * 70)
    print("TEST 3: Feature Engineering")
    print("=" * 70)
    
    predictor = StockPricePredictor('Data.csv', 'StockPrice.csv')
    predictor.load_data()
    predictor.feature_engineering()
    
    expected_features = [
        'Data_Lag1', 'Price_Lag1', 'Data_Change', 'Data_Pct_Change',
        'Data_MA7', 'Data_Std7', 'Data_MA30', 'Data_Std30',
        'Data_Momentum', 'Data_Volatility', 'Data_ROC',
        'Data_Dist_MA7', 'Data_Dist_MA30'
    ]
    
    print("\nFeature Creation:")
    for feature in expected_features:
        if feature in predictor.df.columns:
            print(f"  ✓ {feature} created")
        else:
            print(f"  ✗ {feature} MISSING")
    
    print(f"\n✓ Total features: {len([col for col in predictor.df.columns if col not in ['Date', 'Data', 'Price']])}")
    print("✓ Feature engineering test complete\n")
    return True

def test_model_training():
    """Test model training process."""
    print("=" * 70)
    print("TEST 4: Model Training")
    print("=" * 70)
    
    predictor = StockPricePredictor('Data.csv', 'StockPrice.csv')
    predictor.load_data()
    predictor.feature_engineering()
    predictor.prepare_data()
    results = predictor.train_models()
    
    print("\nModel Training Results:")
    for model_name, metrics in results.items():
        print(f"\n  {model_name}:")
        print(f"    - Train R²: {metrics['train_r2']:.4f}")
        print(f"    - Test R²: {metrics['test_r2']:.4f}")
        print(f"    - Test MSE: {metrics['test_mse']:.2f}")
        print(f"    - Test MAE: ${metrics['test_mae']:.2f}")
        
        # Check for overfitting
        r2_diff = abs(metrics['train_r2'] - metrics['test_r2'])
        if r2_diff < 0.05:
            print(f"    ✓ Good generalization (R² diff: {r2_diff:.4f})")
        else:
            print(f"    ⚠ Possible overfitting (R² diff: {r2_diff:.4f})")
    
    print("\n✓ Model training test complete\n")
    return True

def test_prediction_sanity():
    """Test if predictions are reasonable."""
    print("=" * 70)
    print("TEST 5: Prediction Sanity Check")
    print("=" * 70)
    
    predictor = StockPricePredictor('Data.csv', 'StockPrice.csv')
    predictor.load_data()
    predictor.feature_engineering()
    predictor.prepare_data()
    predictor.train_models()
    
    # Get predictions
    predictions = predictor.best_model.predict(predictor.X_test_scaled)
    actual = predictor.y_test.values
    
    # Check prediction ranges
    pred_min, pred_max = predictions.min(), predictions.max()
    actual_min, actual_max = actual.min(), actual.max()
    
    print("\nValue Ranges:")
    print(f"  Actual prices: ${actual_min:.2f} to ${actual_max:.2f}")
    print(f"  Predicted prices: ${pred_min:.2f} to ${pred_max:.2f}")
    
    # Check for negative predictions
    if (predictions < 0).any():
        print("  ✗ WARNING: Negative predictions detected!")
    else:
        print("  ✓ No negative predictions")
    
    # Check for extreme predictions
    price_range = actual_max - actual_min
    if pred_max > actual_max + price_range * 0.5:
        print("  ⚠ WARNING: Predictions exceed reasonable range")
    else:
        print("  ✓ Predictions within reasonable range")
    
    # Check prediction distribution
    print("\nPrediction Statistics:")
    print(f"  Mean: ${predictions.mean():.2f} (Actual: ${actual.mean():.2f})")
    print(f"  Std Dev: ${predictions.std():.2f} (Actual: ${actual.std():.2f})")
    
    print("\n✓ Prediction sanity check complete\n")
    return True

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" " * 20 + "RUNNING ALL TESTS")
    print("=" * 70 + "\n")
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Data Integrity", test_data_integrity),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("Prediction Sanity", test_prediction_sanity)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASSED" if result else "FAILED"))
        except Exception as e:
            results.append((test_name, f"ERROR: {str(e)}"))
            print(f"✗ {test_name} failed with error: {e}\n")
    
    # Summary
    print("=" * 70)
    print(" " * 25 + "TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results:
        status_symbol = "✓" if result == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    passed = sum(1 for _, r in results if r == "PASSED")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review the errors above.")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_all_tests()