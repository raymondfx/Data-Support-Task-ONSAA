import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import xgboost as xgb
import joblib
import logging
from datetime import datetime

class ModelTrainer:
    """
    A comprehensive model trainer class that handles:
    - Multiple model types
    - Model training and evaluation
    - Model persistence
    - Performance tracking
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize ModelTrainer with specified model type.
        
        Args:
            model_type (str): Type of model to train 
                            ('random_forest', 'gradient_boosting', 'linear', 'xgboost', 'lasso', 'ridge')
        """
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.model = self._initialize_model()
        self.metrics_history = []
        
    def _initialize_model(self) -> Any:
        """
        Initialize the specified model type with default parameters.
        
        Returns:
            Model instance
        """
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ),
            'linear': LinearRegression(),
            'xgboost': xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                random_state=42
            ),
            'lasso': Lasso(alpha=1.0, random_state=42),
            'ridge': Ridge(alpha=1.0, random_state=42)
        }
        
        if self.model_type not in models:
            raise ValueError(f"Model type '{self.model_type}' not supported")
            
        return models[self.model_type]
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              validation_split: float = 0.2,
              **kwargs) -> Dict[str, float]:
        """
        Train the model and evaluate performance.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            validation_split (float): Proportion of data to use for validation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        # Update model parameters if provided
        if kwargs:
            self.model.set_params(**kwargs)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=validation_split, 
            random_state=42
        )
        
        # Train model
        self.logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate performance
        metrics = self._evaluate_model(X_train, X_val, y_train, y_val)
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        return metrics
    
    def _evaluate_model(self, 
                       X_train: np.ndarray,
                       X_val: np.ndarray,
                       y_train: np.ndarray,
                       y_val: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on training and validation sets.
        
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        # Make predictions
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'val_mse': mean_squared_error(y_val, val_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred)
        }
        
        # Add cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("No model to save. Train a model first.")
            
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'metrics_history': self.metrics_history
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.metrics_history = model_data['metrics_history']
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if the model supports it.
        
        Returns:
            Optional[pd.DataFrame]: Feature importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None
            
        importance_df = pd.DataFrame({
            'feature': range(len(self.model.feature_importances_)),
            'importance': self.model.feature_importances_
        })
        return importance_df.sort_values('importance', ascending=False)
    
    def get_training_history(self) -> pd.DataFrame:
        """
        Get the training history with metrics.
        
        Returns:
            pd.DataFrame: Training history
        """
        return pd.DataFrame([
            {
                'timestamp': h['timestamp'],
                **h['metrics']
            }
            for h in self.metrics_history
        ])

# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.random.rand(100, 4)
    y = np.random.rand(100)
    
    # Initialize and train model
    trainer = ModelTrainer(model_type='random_forest')
    metrics = trainer.train(X, y)
    
    # Make predictions
    predictions = trainer.predict(X)
    
    # Save model
    trainer.save_model('model.joblib')
    
    # Load model
    new_trainer = ModelTrainer()
    new_trainer.load_model('model.joblib')
    
    # Get feature importance
    importance = trainer.get_feature_importance()
    
    # Get training history
    history = trainer.get_training_history()