"""
Stock Price Prediction Model
This script predicts stock prices using machine learning based on historical data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class StockPricePredictor:
    """
    A comprehensive class for stock price prediction using machine learning.
    """
    
    def __init__(self, data_path, stock_price_path):
        """
        Initialize the predictor with data paths.
        
        Args:
            data_path: Path to the independent variable dataset
            stock_price_path: Path to the stock price dataset
        """
        self.data_path = data_path
        self.stock_price_path = stock_price_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = None
        
    def load_data(self):
        """Load and merge the datasets."""
        print("=" * 70)
        print("STEP 1: LOADING DATA")
        print("=" * 70)
        
        # Load datasets
        data_df = pd.read_csv(self.data_path)
        stock_df = pd.read_csv(self.stock_price_path)
        
        # Convert Date columns to datetime
        data_df['Date'] = pd.to_datetime(data_df['Date'])
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        
        # Merge datasets on Date
        self.df = pd.merge(data_df, stock_df, on='Date', how='inner')
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        print(f"✓ Data loaded successfully")
        print(f"  Total records: {len(self.df)}")
        print(f"  Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"  Columns: {list(self.df.columns)}")
        print()
        
        return self.df
    
    def feature_engineering(self):
        """Create relevant features for prediction."""
        print("=" * 70)
        print("STEP 2: FEATURE ENGINEERING")
        print("=" * 70)
        
        df = self.df.copy()
        
        # Previous day's data (lag features)
        df['Data_Lag1'] = df['Data'].shift(1)
        df['Price_Lag1'] = df['Price'].shift(1)
        
        # Day-over-day changes
        df['Data_Change'] = df['Data'] - df['Data_Lag1']
        df['Data_Pct_Change'] = ((df['Data'] - df['Data_Lag1']) / df['Data_Lag1']) * 100
        
        # Rolling statistics (7-day window)
        df['Data_MA7'] = df['Data'].rolling(window=7).mean()
        df['Data_Std7'] = df['Data'].rolling(window=7).std()
        
        # Rolling statistics (30-day window)
        df['Data_MA30'] = df['Data'].rolling(window=30).mean()
        df['Data_Std30'] = df['Data'].rolling(window=30).std()
        
        # Momentum indicators
        df['Data_Momentum'] = df['Data'] - df['Data'].shift(5)
        
        # Volatility (rolling standard deviation)
        df['Data_Volatility'] = df['Data'].rolling(window=10).std()
        
        # Rate of change
        df['Data_ROC'] = ((df['Data'] - df['Data'].shift(10)) / df['Data'].shift(10)) * 100
        
        # Distance from moving averages
        df['Data_Dist_MA7'] = df['Data'] - df['Data_MA7']
        df['Data_Dist_MA30'] = df['Data'] - df['Data_MA30']
        
        # Drop rows with NaN values (due to lag and rolling operations)
        df = df.dropna().reset_index(drop=True)
        
        self.df = df
        
        print(f"✓ Feature engineering completed")
        print(f"  Total features created: {len(df.columns) - 3}")  # Excluding Date, Data, Price
        print(f"  Records after cleaning: {len(df)}")
        print(f"\nFeatures created:")
        feature_cols = [col for col in df.columns if col not in ['Date', 'Data', 'Price']]
        for i, feat in enumerate(feature_cols, 1):
            print(f"  {i}. {feat}")
        print()
        
        return self.df
    
    def prepare_data(self):
        """Prepare training and testing datasets."""
        print("=" * 70)
        print("STEP 3: DATA PREPARATION")
        print("=" * 70)
        
        # Select features (all except Date and target Price)
        feature_cols = [col for col in self.df.columns if col not in ['Date', 'Price', 'Data']]
        self.feature_names = feature_cols
        
        X = self.df[feature_cols]
        y = self.df['Price']
        
        # Split data (80% train, 20% test)
        # Using time-based split to maintain temporal order
        split_index = int(len(X) * 0.8)
        self.X_train = X[:split_index]
        self.X_test = X[split_index:]
        self.y_train = y[:split_index]
        self.y_test = y[split_index:]
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✓ Data split completed")
        print(f"  Training set size: {len(self.X_train)} ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"  Testing set size: {len(self.X_test)} ({len(self.X_test)/len(X)*100:.1f}%)")
        print(f"  Number of features: {len(feature_cols)}")
        print(f"\nTarget variable statistics:")
        print(f"  Mean: ${y.mean():.2f}")
        print(f"  Std Dev: ${y.std():.2f}")
        print(f"  Min: ${y.min():.2f}")
        print(f"  Max: ${y.max():.2f}")
        print()
        
    def train_models(self):
        """Train multiple models and select the best one."""
        print("=" * 70)
        print("STEP 4: MODEL TRAINING")
        print("=" * 70)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predictions
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            
            # Metrics
            train_mse = mean_squared_error(self.y_train, y_pred_train)
            test_mse = mean_squared_error(self.y_test, y_pred_test)
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'predictions': y_pred_test
            }
            
            print(f"  ✓ Training R²: {train_r2:.4f}")
            print(f"  ✓ Testing R²: {test_r2:.4f}")
            print(f"  ✓ Testing MSE: {test_mse:.2f}")
            print(f"  ✓ Testing MAE: ${test_mae:.2f}")
        
        # Select best model based on test R²
        best_model_name = max(results, key=lambda x: results[x]['test_r2'])
        self.best_model = results[best_model_name]['model']
        self.results = results
        
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"{'='*70}")
        print()
        
        return results
    
    def evaluate_model(self):
        """Detailed evaluation of the best model."""
        print("=" * 70)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 70)
        
        best_name = [name for name, res in self.results.items() 
                     if res['model'] == self.best_model][0]
        best_res = self.results[best_name]
        
        print(f"\nModel: {best_name}")
        print(f"\nPerformance Metrics:")
        print(f"  • R² Score: {best_res['test_r2']:.4f}")
        print(f"    (Explains {best_res['test_r2']*100:.2f}% of variance)")
        print(f"  • Mean Squared Error: {best_res['test_mse']:.2f}")
        print(f"  • Root Mean Squared Error: {np.sqrt(best_res['test_mse']):.2f}")
        print(f"  • Mean Absolute Error: ${best_res['test_mae']:.2f}")
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 5 Most Important Features:")
            for idx, row in feature_importance.head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print()
        
    # from matplotlib.backends.backend_pdf import PdfPages

    def visualize_results(self):
        """Create clean visualizations on separate pages."""
        print("=" * 70)
        print("STEP 6: VISUALIZATION")
        print("=" * 70)

        best_name = [name for name, res in self.results.items()
                    if res['model'] == self.best_model][0]
        predictions = self.results[best_name]['predictions']
        residuals = self.y_test.values - predictions

        with PdfPages("stock_prediction_results.pdf") as pdf:

            # -------------------- 1. Actual vs Predicted --------------------
            plt.figure(figsize=(10, 6))
            plt.plot(self.y_test.values, label="Actual Price", linewidth=2)
            plt.plot(predictions, label="Predicted Price", linewidth=2)
            plt.title("Actual vs Predicted Stock Prices")
            plt.xlabel("Time Index")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            pdf.savefig()
            plt.close()

            # -------------------- 2. Prediction Scatter --------------------
            plt.figure(figsize=(8, 6))
            plt.scatter(self.y_test, predictions, alpha=0.6)
            plt.plot(
                [self.y_test.min(), self.y_test.max()],
                [self.y_test.min(), self.y_test.max()],
                "r--"
            )
            plt.title("Actual vs Predicted Scatter Plot")
            plt.xlabel("Actual Price")
            plt.ylabel("Predicted Price")
            plt.grid(True)
            pdf.savefig()
            plt.close()

            # -------------------- 3. Residual Plot --------------------
            plt.figure(figsize=(8, 6))
            plt.scatter(predictions, residuals, alpha=0.6)
            plt.axhline(0, color="red", linestyle="--")
            plt.title("Residual Plot")
            plt.xlabel("Predicted Price")
            plt.ylabel("Residual")
            plt.grid(True)
            pdf.savefig()
            plt.close()

            # -------------------- 4. Residual Distribution --------------------
            plt.figure(figsize=(8, 6))
            plt.hist(residuals, bins=30, edgecolor="black")
            plt.axvline(0, color="red", linestyle="--")
            plt.title("Residual Distribution")
            plt.xlabel("Residual")
            plt.ylabel("Frequency")
            plt.grid(True)
            pdf.savefig()
            plt.close()

            # -------------------- 5. Model Comparison --------------------
            plt.figure(figsize=(8, 6))
            model_names = list(self.results.keys())
            r2_scores = [self.results[m]["test_r2"] for m in model_names]

            plt.bar(model_names, r2_scores)
            plt.title("Model Performance Comparison")
            plt.ylabel("R² Score")
            plt.grid(True, axis="y")
            pdf.savefig()
            plt.close()

            # -------------------- 6. Feature Importance --------------------
            if hasattr(self.best_model, "feature_importances_"):
                importances = self.best_model.feature_importances_
                feature_importance = pd.DataFrame({
                    "Feature": self.feature_names,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False).head(10)

                plt.figure(figsize=(10, 6))
                plt.barh(
                    feature_importance["Feature"],
                    feature_importance["Importance"]
                )
                plt.title("Top 10 Feature Importances")
                plt.xlabel("Importance")
                plt.grid(True)
                pdf.savefig()
                plt.close()

        print("✅ All graphs saved neatly in: stock_prediction_results.pdf")

        
    def generate_insights(self):
        """Generate insights from the analysis."""
        print("=" * 70)
        print("STEP 7: KEY INSIGHTS")
        print("=" * 70)
        
        best_name = [name for name, res in self.results.items() 
                     if res['model'] == self.best_model][0]
        best_res = self.results[best_name]
        
        print(f"\n1. MODEL PERFORMANCE:")
        print(f"   The {best_name} model achieved an R² score of {best_res['test_r2']:.4f},")
        print(f"   explaining {best_res['test_r2']*100:.2f}% of the variance in stock prices.")
        
        print(f"\n2. PREDICTION ACCURACY:")
        print(f"   On average, predictions deviate by ${best_res['test_mae']:.2f} from actual prices.")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            top_feature = feature_importance.iloc[0]
            print(f"\n3. KEY DRIVERS:")
            print(f"   The most important predictor is '{top_feature['feature']}'")
            print(f"   with an importance score of {top_feature['importance']:.4f}.")
        
        print(f"\n4. DATA IMPACT:")
        print(f"   Changes in previous day's data significantly influence stock price movements.")
        print(f"   Rolling averages and momentum indicators provide additional predictive power.")
        
        print(f"\n5. MODEL RECOMMENDATION:")
        print(f"   The {best_name} is recommended for production deployment")
        print(f"   based on superior performance metrics.")
        
        print("\n" + "=" * 70)
        print()
        
    def run_complete_analysis(self):
        """Execute the complete analysis pipeline."""
        print("\n" + "=" * 70)
        print(" " * 15 + "STOCK PRICE PREDICTION ANALYSIS")
        print(" " * 20 + "Futures First - Python Intern 2026")
        print("=" * 70 + "\n")
        
        self.load_data()
        self.feature_engineering()
        self.prepare_data()
        self.train_models()
        self.evaluate_model()
        self.visualize_results()
        self.generate_insights()
        
        print("=" * 70)
        print(" " * 25 + "ANALYSIS COMPLETE!")
        print("=" * 70)
        print()


# Main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = StockPricePredictor(
        data_path='Data.csv',
        stock_price_path='StockPrice.csv'
    )
    
    # Run complete analysis
    predictor.run_complete_analysis()