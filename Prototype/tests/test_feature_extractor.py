import unittest
import pandas as pd
import numpy as np
from ml_models.feature_extractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        """
        Create sample data and initialize the FeatureExtractor for testing.
        """
        self.data = pd.DataFrame({
            'area': [1500, 1800, 2000],
            'bedrooms': [3, 4, 2],
            'location': ['urban', 'suburban', 'rural'],
            'price': [250000, 300000, 200000]
        })
        self.features = self.data.drop('price', axis=1)
        self.target = self.data['price']
        self.extractor = FeatureExtractor(
            scaling_method='standard',
            encoding_method='onehot',
            feature_selection='f_score',
            selection_k=3,
            pca_components=None
        )

    def test_fit_transform(self):
        """
        Test the fit_transform method of FeatureExtractor.
        """
        transformed = self.extractor.fit_transform(self.features, self.target)
        self.assertEqual(
            transformed.shape[0], 
            self.features.shape[0],
            "Number of rows in the transformed data should match the input."
        )
        self.assertGreater(
            transformed.shape[1], 
            0,
            "Transformed data should have non-zero columns."
        )

    def test_get_feature_names(self):
        """
        Test the get_feature_names method of FeatureExtractor.
        """
        self.extractor.fit(self.features, self.target)
        feature_names = self.extractor.get_feature_names(self.features)
        self.assertIsInstance(
            feature_names, 
            list,
            "Feature names should be returned as a list."
        )
        self.assertGreater(
            len(feature_names), 
            0,
            "Feature names list should not be empty."
        )

    def test_extract_interaction_features(self):
        """
        Test the extract_interaction_features method.
        """
        interaction_features = self.extractor.extract_interaction_features(self.features)
        self.assertTrue(
            all(
                [f.startswith('interaction_') for f in interaction_features.columns if 'interaction_' in f]
            ),
            "Interaction features should have 'interaction_' prefix."
        )
        self.assertGreater(
            len(interaction_features.columns),
            len(self.features.columns),
            "Number of columns with interaction features should increase."
        )

    def test_polynomial_features(self):
        """
        Test the polynomial_features method.
        """
        poly_features = self.extractor.polynomial_features(self.features, degree=2)
        self.assertGreater(
            poly_features.shape[1],
            self.features.shape[1],
            "Polynomial features should increase the number of columns."
        )
        self.assertTrue(
            'area^2' in poly_features.columns,
            "Polynomial features should include squared terms."
        )

    def test_invalid_scaling_method(self):
        """
        Test invalid scaling method handling.
        """
        with self.assertRaises(KeyError):
            invalid_extractor = FeatureExtractor(scaling_method='invalid_method')
            invalid_extractor.fit_transform(self.features)

    def test_invalid_encoding_method(self):
        """
        Test invalid encoding method handling.
        """
        with self.assertRaises(KeyError):
            invalid_extractor = FeatureExtractor(encoding_method='invalid_method')
            invalid_extractor.fit_transform(self.features)

    def test_empty_feature_set(self):
        """
        Test handling of empty feature set.
        """
        empty_features = self.features.iloc[:, 0:0]
        transformed = self.extractor.fit_transform(empty_features, self.target)
        self.assertEqual(
            transformed.shape[1], 
            0,
            "Transformed output should have zero columns when input is empty."
        )


if __name__ == "__main__":
    unittest.main()
