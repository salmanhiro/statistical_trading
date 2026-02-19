"""
Statistical models for stock price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Optional
import pickle


class StockPricePredictor:
    """
    Predicts stock prices using various statistical and machine learning models.
    """
    
    def __init__(self, model_type: str = 'linear'):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model to use ('linear', 'ridge', 'lasso', 'random_forest')
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
    
    def _create_model(self, model_type: str):
        """Create the appropriate model based on type."""
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type]
    
    def prepare_data(
        self, 
        features: pd.DataFrame, 
        target: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.
        
        Args:
            features: Feature DataFrame
            target: Target series
            test_size: Proportion of data to use for testing
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Select only numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        self.feature_names = numeric_features.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            numeric_features, target, test_size=test_size, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"{self.model_type} model trained successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from file.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True
        
        print(f"Model loaded from {filepath}")


class TimeSeriesAnalyzer:
    """
    Analyzes time series properties of stock data.
    """
    
    @staticmethod
    def check_stationarity(data: pd.Series) -> dict:
        """
        Check if time series is stationary using ADF test.
        
        Args:
            data: Time series data
        
        Returns:
            Dictionary with test results
        """
        from statsmodels.tsa.stattools import adfuller
        
        result = adfuller(data.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    @staticmethod
    def calculate_autocorrelation(data: pd.Series, lags: int = 40) -> pd.Series:
        """
        Calculate autocorrelation for different lags.
        
        Args:
            data: Time series data
            lags: Number of lags to calculate
        
        Returns:
            Series with autocorrelation values
        """
        from statsmodels.tsa.stattools import acf
        
        acf_values = acf(data.dropna(), nlags=lags)
        return pd.Series(acf_values, index=range(len(acf_values)))
