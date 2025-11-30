import unittest
import numpy as np
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
from si.feature_selection.select_percentile import SelectPercentile

class TestSelectPercentile(unittest.TestCase):
    def setUp(self):
        """Set up a Dataset with known F-values for testing."""
        # Example dataset with 4 features
        self.X = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ])
        self.y = np.array([0, 1, 0, 1])
        self.features = ["f1", "f2", "f3", "f4"]
        self.dataset = Dataset(X=self.X, y=self.y, features=self.features)

        # Mock F-values for testing
        self.mock_F = np.array([1.2, 3.4, 5.6, 2.1])
        self.mock_p = np.array([0.1, 0.2, 0.3, 0.4])

    def test_init_valid_percentile(self):
        """Test that the constructor accepts valid percentiles."""
        selector = SelectPercentile(percentile=20, score_func=f_classification)
        self.assertEqual(selector.percentile, 20)
        self.assertEqual(selector.score_func, f_classification)

    def test_init_invalid_percentile(self):
        """Test that the constructor raises an error for invalid percentiles."""
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=120, score_func=f_classification)
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=-10, score_func=f_classification)

    def test_fit(self):
        """Test that the _fit method computes F and p values."""
        def mock_score_func(dataset):
            return self.mock_F, self.mock_p

        selector = SelectPercentile(percentile=50, score_func=mock_score_func)
        selector._fit(self.dataset)

        np.testing.assert_array_equal(selector.F, self.mock_F)
        np.testing.assert_array_equal(selector.p, self.mock_p) 

    def test_transform_ties(self):
        """Test that the _transform method handles ties correctly."""
        # Mock F-values with ties: [1.2, 3.4, 5.6, 5.6]
        self.mock_F = np.array([1.2, 3.4, 5.6, 5.6])
        self.mock_p = np.array([0.1, 0.2, 0.3, 0.3])

        def mock_score_func(dataset):
            return self.mock_F, self.mock_p

        selector = SelectPercentile(percentile=50, score_func=mock_score_func)
        selector._fit(self.dataset)

        # Transform the dataset
        transformed_dataset = selector._transform(self.dataset)

        # Check that the correct features are selected (top 2: f3, f4)
        self.assertEqual(len(transformed_dataset.features), 2)
        self.assertEqual(transformed_dataset.features, ["f3", "f4"])

    def test_transform_all_features(self):
        """Test that the _transform method selects all features if percentile=100."""
        def mock_score_func(dataset):
            return self.mock_F, self.mock_p

        selector = SelectPercentile(percentile=100, score_func=mock_score_func)
        selector._fit(self.dataset)

        # Transform the dataset
        transformed_dataset = selector._transform(self.dataset)

        # Check that all features are selected
        self.assertEqual(len(transformed_dataset.features), 4)
        self.assertEqual(transformed_dataset.features, self.features)

    def test_transform(self):
        """Test that the _transform method selects the correct features."""
        # Define a mock score function
        def mock_score_func(dataset):
            return self.mock_F, self.mock_p

        selector = SelectPercentile(percentile=50, score_func=mock_score_func)
        selector._fit(self.dataset)

        transformed_dataset = selector._transform(self.dataset)

        # Expect the top 2 features: f3 (5.6) and f2 (3.4)
        self.assertEqual(len(transformed_dataset.features), 2)

        # Cast to plain strings and ignore order
        selected = list(map(str, transformed_dataset.features))
        self.assertCountEqual(selected, ["f3", "f2"])