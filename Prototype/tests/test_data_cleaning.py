import pytest
import pandas as pd
import numpy as np
from ml_models.data_cleaner import HousingDataCleaner

class TestDataCleaning:
    @pytest.fixture
    def sample_dataframe(self):
        """
        Create a sample DataFrame for testing data cleaning
        """
        return pd.DataFrame({
            'price': [250000, 300000, 200000],
            'area': [1500, 1800, 1200],
            'bedrooms': [3, 4, 2],
            'location': ['urban', 'suburban', 'rural'],
            'age': [10, 5, 20],
            'missing_col': [np.nan, 100, np.nan]
        })

    def test_data_cleaning_basic(self, sample_dataframe):
        """
        Test basic data cleaning functionality
        """
        cleaner = HousingDataCleaner()
        
        # Test preprocessing creation
        preprocessor = cleaner.create_preprocessor(sample_dataframe)
        assert preprocessor is not None, "Preprocessor should be created"
        
        # Test data cleaning
        cleaned_data = cleaner.clean_data(sample_dataframe)
        assert cleaned_data is not None, "Cleaned data should not be None"
        
        # Verify data transformation
        assert cleaned_data.shape[0] > 0, "Cleaned data should have rows"

    def test_handling_missing_values(self, sample_dataframe):
        """
        Test handling of missing values
        """
        cleaner = HousingDataCleaner()
        cleaned_data = cleaner.clean_data(sample_dataframe)
        
        # Verify no missing values remain
        assert not np.any(np.isnan(cleaned_data)), "No NaN values should remain"

    def test_feature_scaling(self, sample_dataframe):
        """
        Test feature scaling and normalization
        """
        cleaner = HousingDataCleaner()
        preprocessor = cleaner.create_preprocessor(sample_dataframe)
        
        # Transform the data
        scaled_data = preprocessor.fit_transform(sample_dataframe)
        
        # Check scaling properties
        numeric_cols = sample_dataframe.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            # Verify columns are scaled close to standard normal distribution
            col_data = scaled_data[:, list(sample_dataframe.columns).index(col)]
            assert np.isclose(np.mean(col_data), 0, atol=1), f"Column {col} should be centered around 0"
            assert np.isclose(np.std(col_data), 1, atol=1), f"Column {col} should have standard deviation close to 1"