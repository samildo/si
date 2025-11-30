from unittest import TestCase
import numpy as np
import os

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA


class TestPCA(TestCase):

    def setUp(self):
        # Use the iris dataset
        self.csv_file = os.path.join(DATASETS_PATH, "iris", "iris.csv")
        self.dataset = read_csv(
            filename=self.csv_file,
            features=True,
            label=True
        )  # X: (150, 4) typically, y: iris labels

    def test_init(self):
        """Test that the PCA object is initialized correctly."""
        pca = PCA(n_components=2)
        self.assertEqual(pca.n_components, 2)
        self.assertIsNone(pca.mean)
        self.assertIsNone(pca.components)
        self.assertIsNone(pca.explained_variance)
        self.assertFalse(pca.is_fitted)

    def test_fit(self):
        """Test that the _fit method computes the components and explained variance on iris."""
        pca = PCA(n_components=2)
        pca._fit(self.dataset)

        # Components and explained variance computed
        self.assertIsNotNone(pca.mean)
        self.assertIsNotNone(pca.components)
        self.assertIsNotNone(pca.explained_variance)
        self.assertTrue(pca.is_fitted)

        # Shape: (n_components, n_features) -> (2, 4) for iris
        self.assertEqual(pca.components.shape, (2, self.dataset.X.shape[1]))

        # explained_variance is an array of length n_components
        self.assertIsInstance(pca.explained_variance, np.ndarray)
        self.assertEqual(len(pca.explained_variance), 2)

    def test_transform(self):
        """Test that the _transform method reduces dimensionality of iris to n_components."""
        pca = PCA(n_components=2)
        pca._fit(self.dataset)
        transformed_dataset = pca._transform(self.dataset)

        # Shape: (n_samples, n_components)
        self.assertEqual(
            transformed_dataset.X.shape,
            (self.dataset.X.shape[0], 2)
        )

        # Features labeled as PC1, PC2
        self.assertEqual(transformed_dataset.features, ["PC1", "PC2"])

    def test_fit_transform_consistency(self):
        """Test that manual projection matches _transform result on iris."""
        pca = PCA(n_components=2)
        pca._fit(self.dataset)
        transformed_dataset = pca._transform(self.dataset)

        # Centered data
        X_centered = self.dataset.X - pca.mean

        # Manual projection
        X_reduced_manual = np.dot(X_centered, pca.components.T)

        np.testing.assert_array_almost_equal(
            transformed_dataset.X,
            X_reduced_manual
        )
