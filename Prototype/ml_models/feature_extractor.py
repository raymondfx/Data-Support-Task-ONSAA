# ml_models/feature_extractor.py
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler,
    OneHotEncoder, 
    LabelEncoder
)
from sklearn.feature_selection import (
    SelectKBest, 
    f_regression, 
    mutual_info_regression
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Advanced feature extraction and engineering class
    
    Supports multiple feature extraction techniques:
    - Scaling
    - Encoding
    - Feature selection
    - Dimensionality reduction
    """
    
    def __init__(self, 
                 scaling_method: str = 'standard',
                 encoding_method: str = 'onehot',
                 feature_selection: Optional[str] = None,
                 selection_k: int = 10,
                 pca_components: Optional[int] = None):
        """
        Initialize feature extractor with various options
        
        Args:
            scaling_method (str): Scaling technique 
                ['standard', 'minmax', 'robust']
            encoding_method (str): Categorical encoding 
                ['onehot', 'label']
            feature_selection (str, optional): Feature selection method 
                [None, 'f_score', 'mutual_info']
            selection_k (int): Number of top features to select
            pca_components (int, optional): Number of PCA components
        """
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.feature_selection = feature_selection
        self.selection_k = selection_k
        self.pca_components = pca_components
        
        # Scalers mapping
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        # Encoders mapping
        self.encoders = {
            'onehot': OneHotEncoder(sparse=False, handle_unknown='ignore'),
            'label': LabelEncoder()
        }
        
        # Feature selection mapping
        self.feature_selectors = {
            'f_score': SelectKBest(score_func=f_regression),
            'mutual_info': SelectKBest(score_func=mutual_info_regression)
        }
        
        # Internal storage for fitted transformers
        self.numeric_scaler = None
        self.categorical_encoder = None
        self.feature_selector = None
        self.pca_transformer = None
        
    def _get_column_types(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize columns by data type
        
        Args:
            X (pd.DataFrame): Input DataFrame
        
        Returns:
            Dict of column types
        """
        return {
            'numeric': X.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical': X.select_dtypes(include=['object', 'category']).columns.tolist()
        }
    
    def _create_feature_pipeline(self, 
                                 numeric_columns: List[str], 
                                 categorical_columns: List[str]) -> Pipeline:
        """
        Create a comprehensive feature processing pipeline
        
        Args:
            numeric_columns (List[str]): Numeric feature columns
            categorical_columns (List[str]): Categorical feature columns
        
        Returns:
            sklearn Pipeline
        """
        from sklearn.compose import ColumnTransformer
        
        # Numeric transformer
        numeric_transformer = Pipeline(steps=[
            ('scaler', self.scalers[self.scaling_method])
        ])
        
        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('encoder', self.encoders[self.encoding_method])
        ])
        
        # Column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_columns),
                ('cat', categorical_transformer, categorical_columns)
            ])
        
        # Build full pipeline with optional feature selection and PCA
        pipeline_steps = [('preprocessor', preprocessor)]
        
        if self.feature_selection:
            pipeline_steps.append((
                'selector', 
                self.feature_selectors[self.feature_selection]
            ))
        
        if self.pca_components:
            pipeline_steps.append((
                'pca', 
                PCA(n_components=self.pca_components)
            ))
        
        return Pipeline(steps=pipeline_steps)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None):
        """
        Fit feature extraction pipeline
        
        Args:
            X (DataFrame or ndarray): Input features
            y (ndarray, optional): Target variable for feature selection
        
        Returns:
            self
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Identify column types
        column_types = self._get_column_types(X)
        
        # Create feature pipeline
        self.feature_pipeline = self._create_feature_pipeline(
            column_types['numeric'], 
            column_types['categorical']
        )
        
        # Fit the pipeline
        if self.feature_selection and y is not None:
            # If feature selection is used, pass target variable
            self.feature_pipeline.fit(X, y)
        else:
            # Otherwise, fit without target
            self.feature_pipeline.fit(X)
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform input features
        
        Args:
            X (DataFrame or ndarray): Input features
        
        Returns:
            Transformed feature matrix
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Transform features
        return self.feature_pipeline.transform(X)
    
    def fit_transform(self, 
                      X: Union[pd.DataFrame, np.ndarray], 
                      y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            X (DataFrame or ndarray): Input features
            y (ndarray, optional): Target variable
        
        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """
        Get transformed feature names
        
        Args:
            X (DataFrame): Original input DataFrame
        
        Returns:
            List of feature names after transformation
        """
        column_types = self._get_column_types(X)
        
        # Get feature names from the pipeline
        feature_names = []
        
        # Add numeric feature names
        if self.scaling_method:
            feature_names.extend(
                [f'scaled_{col}' for col in column_types['numeric']]
            )
        
        # Add categorical feature names (for one-hot encoding)
        if self.encoding_method == 'onehot':
            encoder = self.feature_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
            cat_feature_names = encoder.get_feature_names_out(column_types['categorical'])
            feature_names.extend(cat_feature_names)
        
        return feature_names
    
    def extract_interaction_features(self, 
                                     X: pd.DataFrame, 
                                     interaction_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create interaction features between selected columns
        
        Args:
            X (DataFrame): Input features
            interaction_columns (List[str], optional): Columns to create interactions
        
        Returns:
            DataFrame with additional interaction features
        """
        if interaction_columns is None:
            # Default to numeric columns
            interaction_columns = X.select_dtypes(include=['int64', 'float64']).columns
        
        X_interactions = X.copy()
        
        # Pairwise interactions
        for i in range(len(interaction_columns)):
            for j in range(i+1, len(interaction_columns)):
                col1, col2 = interaction_columns[i], interaction_columns[j]
                interaction_name = f'interaction_{col1}_{col2}'
                X_interactions[interaction_name] = X[col1] * X[col2]
        
        return X_interactions
    
    def polynomial_features(self, 
                             X: pd.DataFrame, 
                             degree: int = 2, 
                             include_bias: bool = False) -> pd.DataFrame:
        """
        Generate polynomial features
        
        Args:
            X (DataFrame): Input features
            degree (int): Polynomial degree
            include_bias (bool): Include bias term
        
        Returns:
            DataFrame with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        # Select numeric columns
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Create polynomial features
        poly = PolynomialFeatures(
            degree=degree, 
            include_bias=include_bias
        )
        
        # Transform numeric features
        poly_features = poly.fit_transform(X[numeric_cols])
        
        # Create DataFrame with polynomial features
        feature_names = poly.get_feature_names_out(numeric_cols)
        return pd.DataFrame(poly_features, columns=feature_names, index=X.index)

# Example usage
if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame({
        'area': [1500, 1800, 2000],
        'bedrooms': [3, 4, 2],
        'location': ['urban', 'suburban', 'rural'],
        'price': [250000, 300000, 200000]
    })
    
    # Initialize feature extractor
    extractor = FeatureExtractor(
        scaling_method='standard',
        encoding_method='onehot',
        feature_selection='f_score',
        selection_k=5,
        pca_components=None
    )
    
    # Separate features and target
    X = data.drop('price', axis=1)
    y = data['price']
    
    # Fit and transform features
    X_transformed = extractor.fit_transform(X, y)
    
    # Extract interaction features
    X_with_interactions = extractor.extract_interaction_features(X)
    
    # Generate polynomial features
    X_polynomial = extractor.polynomial_features(X, degree=2)