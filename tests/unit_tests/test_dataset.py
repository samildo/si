import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

class TestDatasetMethods(unittest.TestCase):
    def setUp(self):
        """Set up a Dataset with missing values for testing."""
        self.X = np.array([[1, 2],[3, np.nan],[4, 5],[np.nan, 6]])
        self.y = np.array([0, 1, 0, 1])
        self.dataset = Dataset(X=self.X, y=self.y)

    def test_dropna(self):
        """Test removing rows with NaN values."""
        original_shape = self.dataset.X.shape[0]
        self.dataset.dropna()

        # Check that rows with NaN are removed
        self.assertFalse(np.isnan(self.dataset.X).any())
        self.assertEqual(self.dataset.X.shape[0], original_shape - 2)  # 2 rows had NaN

    def test_fillna_with_value(self):
        """Test filling NaNs with a specified value."""
        self.dataset.fillna(0)

        # Check that no NaNs remain
        self.assertFalse(np.isnan(self.dataset.X).any())

        # Check that NaNs were replaced with 0
        expected_X = np.array([[1, 2],[3, 0],[4, 5],[0, 6]])
        np.testing.assert_array_equal(self.dataset.X, expected_X)

    def test_fillna_with_mean(self):
        """Test filling NaNs with the column-wise mean."""
        self.dataset.fillna("mean")

        # Check that no NaNs remain
        self.assertFalse(np.isnan(self.dataset.X).any())

        # Calculate expected means for each column
        expected_means = np.nanmean(self.X, axis=0)
        expected_X = np.copy(self.X)
        nan_rows, nan_cols = np.where(np.isnan(self.X))
        expected_X[nan_rows, nan_cols] = expected_means[nan_cols]

        # Check that NaNs were replaced with the mean
        np.testing.assert_array_almost_equal(self.dataset.X, expected_X)

    def test_fillna_with_median(self):
        """Test filling NaNs with the column-wise median."""
        self.dataset.fillna("median")

        # Check that no NaNs remain
        self.assertFalse(np.isnan(self.dataset.X).any())

        # Calculate expected medians for each column
        expected_medians = np.nanmedian(self.X, axis=0)
        expected_X = np.copy(self.X)
        nan_rows, nan_cols = np.where(np.isnan(self.X))
        expected_X[nan_rows, nan_cols] = expected_medians[nan_cols]

        # Check that NaNs were replaced with the median
        np.testing.assert_array_almost_equal(self.dataset.X, expected_X)

    def test_fillna_invalid_value(self):
        """Test that an invalid value raises an error."""
        with self.assertRaises(ValueError):
            self.dataset.fillna("invalid")

    def test_fillna_invalid_type(self):
        """Test that an invalid type raises an error."""
        with self.assertRaises(TypeError):
            self.dataset.fillna([1, 2, 3])  # Invalid type (list)

    def test_remove_by_index(self):
        """Test removing a sample by its index."""
        original_shape = self.dataset.X.shape[0]
        self.dataset.remove_by_index(1)  # Remove the second sample

        # Check that the sample was removed
        self.assertEqual(self.dataset.X.shape[0], original_shape - 1)
        self.assertEqual(self.dataset.y.shape[0], original_shape - 1)

    def test_remove_by_index_out_of_bounds(self):
        """Test that removing an out-of-bounds index raises an error."""
        with self.assertRaises(IndexError):
            self.dataset.remove_by_index(100)  # Invalid index
