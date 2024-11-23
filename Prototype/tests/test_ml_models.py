import pytest
import numpy as np
from ml_models.model_trainer import ModelTrainer
from ml_models.data_cleaner import DataCleaner
import pandas as pd

class ModelTrainer:
    @pytest.fixture
    def sample_dataset(self):
        """
        Create a sample dataset for testing
        """
        np.random.seed(42)
        data = pd.DataFrame({
            'area': np.random.uniform(1000, 3000, 100),
            'bedrooms': np.random.randint(1, 6, 100),
            'location': np.random.choice(['urban', 'suburban', 'rural'], 100),
            'age': np.random.uniform(0, 50, 100),
            'price': np.random.uniform(100000, 500000, 100)
        })
        return data

    def test_model_training(self, sample_dataset):
        """
        Test model training process
        """
        # Prepare data
        cleaner = DataCleaner()
        predictor = ModelTrainer()
        
        # Separate features and target
        X = sample_dataset.drop('price', axis=1)
        y = sample_dataset['price']
        
        # Clean data
        X_processed = cleaner.clean_data(X)
        
        # Train model
        predictor.train(X_processed, y)
        
        # Verify model is trained
        assert hasattr(predictor.model, 'predict'), "Model should have prediction capability"

    def test_model_prediction(self, sample_dataset):
        """
        Test model prediction functionality
        """
        # Prepare data
        cleaner = DataCleaner()
        predictor = ModelTrainer()
        
        # Separate features and target
        X = sample_dataset.drop('price', axis=1)
        y = sample_dataset['price']
        
        # Clean data
        X_processed = cleaner.clean_data(X)
        
        # Train model
        predictor.train(X_processed, y)
        
        # Make predictions
        predictions = predictor.predict(X_processed)
        
        # Verify predictions
        assert predictions.shape[0] == X_processed.shape[0], "Prediction count should match input"
        assert np.all(predictions > 0), "All predictions should be positive"

    def test_model_performance(self, sample_dataset):
        """
        Test model performance metrics
        """
        # Prepare data
        cleaner = DataCleaner()
        predictor = ModelTrainer()
        
        # Separate features and target
        X = sample_dataset.drop('price', axis=1)
        y = sample_dataset['price']
        
        # Clean data
        X_processed = cleaner.clean_data(X)
        
        # Train model
        predictor.train(X_processed, y)
        
        # Make predictions
        predictions = predictor.predict(X_processed)
        
        # Calculate basic performance metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        # Performance assertions
        assert mae > 0, "Mean Absolute Error should be positive"
        assert mse > 0, "Mean Squared Error should be positive"
        assert -1 <= r2 <= 1, "R2 Score should be between -1 and 1"

# Optional: Utility for running all tests
def run_tests():
    """
    Run all tests in the project
    """
    import pytest
    pytest.main([__file__])