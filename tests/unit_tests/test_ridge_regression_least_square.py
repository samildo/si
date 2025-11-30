from unittest import TestCase
import os
import numpy as np
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares

class TestRidgeRegressionLeastSquares(TestCase):
    def setUp(self):
        """Set up test data using the CPU dataset."""
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_init(self):
        """Test the initialization of RidgeRegressionLeastSquares."""
        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        self.assertEqual(model.l2_penalty, 1.0)
        self.assertEqual(model.scale, True)
        self.assertIsNone(model.theta)
        self.assertIsNone(model.theta_zero)
        self.assertIsNone(model.mean)
        self.assertIsNone(model.std)

    def test_fit(self):
        """Test the fit method with CPU dataset."""
        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        fitted_model = model._fit(self.train_dataset)

        # Check that the model was fitted correctly
        self.assertIsInstance(fitted_model, RidgeRegressionLeastSquares)
        self.assertIsNotNone(model.theta)
        self.assertIsNotNone(model.theta_zero)
        self.assertIsNotNone(model.mean)
        self.assertIsNotNone(model.std)

        # Check that theta values are reasonable
        self.assertTrue(isinstance(model.theta, np.ndarray))
        self.assertTrue(isinstance(model.theta_zero, float))
        self.assertEqual(model.theta.shape[0], self.train_dataset.X.shape[1])

    def test_fit_without_scaling(self):
        """Test the fit method without scaling."""
        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=False)
        fitted_model = model._fit(self.train_dataset)

        # Check that the model was fitted correctly
        self.assertIsInstance(fitted_model, RidgeRegressionLeastSquares)
        self.assertIsNotNone(model.theta)
        self.assertIsNotNone(model.theta_zero)
        self.assertIsNone(model.mean)
        self.assertIsNone(model.std)

    def test_predict(self):
        """Test the predict method with CPU dataset."""
        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        model._fit(self.train_dataset)

        # Make predictions
        predictions = model._predict(self.test_dataset)

        # Check that predictions are of the correct shape
        self.assertEqual(predictions.shape, (len(self.test_dataset.y),))

        # Check that all predictions are finite numbers
        self.assertTrue(np.all(np.isfinite(predictions)))

        # Check that predictions are not all the same value (model is actually doing something)
        self.assertGreater(np.std(predictions), 1e-6)

    def test_score(self):
        """Test the score method with CPU dataset."""
        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        model._fit(self.train_dataset)

        # Calculate score
        score = model._score(self.test_dataset)

        # Check that score is a float
        self.assertIsInstance(score, float)

        # Check that score is non-negative
        self.assertGreaterEqual(score, 0)

        # Check that score is reasonable (not extremely large)
        self.assertLess(score, 1e10)

    def test_score_with_predictions(self):
        """Test the score method with precomputed predictions."""
        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        model._fit(self.train_dataset)

        # Make predictions
        predictions = model._predict(self.test_dataset)

        # Calculate score with precomputed predictions
        score = model._score(self.test_dataset, predictions)

        # Check that score is a float
        self.assertIsInstance(score, float)

        # Check that score is non-negative
        self.assertGreaterEqual(score, 0)

    def test_different_l2_penalty(self):
        """Test the model with different L2 penalty values."""
        for l2_penalty in [0.01, 0.1, 1.0, 10.0]:
            model = RidgeRegressionLeastSquares(l2_penalty=l2_penalty, scale=True)
            model._fit(self.train_dataset)

            # Check that model was fitted
            self.assertIsNotNone(model.theta)
            self.assertIsNotNone(model.theta_zero)

            # Make predictions
            predictions = model._predict(self.test_dataset)
            self.assertEqual(predictions.shape, (len(self.test_dataset.y),))
            self.assertTrue(np.all(np.isfinite(predictions)))

            # Calculate score
            score = model._score(self.test_dataset, predictions)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
