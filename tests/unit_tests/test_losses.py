import os
from unittest import TestCase
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from datasets import DATASETS_PATH
from si.neural_networks.losses import BinaryCrossEntropy, MeanSquaredError, CategoricalCrossEntropy
import numpy as np 


class TestLosses(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_mean_squared_error_loss(self):

        error = MeanSquaredError().loss(self.dataset.y, self.dataset.y)

        self.assertEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = MeanSquaredError().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

    def test_binary_cross_entropy_loss(self):

        error = BinaryCrossEntropy().loss(self.dataset.y, self.dataset.y)

        self.assertAlmostEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = BinaryCrossEntropy().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

#Ex 14 tests
    def test_categorical_cross_entropy_loss(self):
        """Tests the categorical cross entropy loss logic."""
        # 1. Prepare One-Hot labels from breast-bin (binary -> 2 columns)
        y_true = np.eye(2)[self.dataset.y.astype(int)]
        
        # 2. Test perfect prediction (loss should be 0)
        error = CategoricalCrossEntropy().loss(y_true, y_true)
        
        #Checks almost equal due to the 1e-15 clipping in the implementation
        self.assertAlmostEqual(error, 0, places=10)

    def test_categorical_cross_entropy_derivative(self):
        """Tests the categorical cross entropy derivative shape and logic."""
        # 1. Prepare One-Hot labels
        y_true = np.eye(2)[self.dataset.y.astype(int)]
        
        # 2. Compute derivative
        derivative_error = CategoricalCrossEntropy().derivative(y_true, y_true)

        # 3. Shape must be (num_samples, num_classes) -> (569, 2)
        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])
        self.assertEqual(derivative_error.shape[1], 2)

    def test_categorical_cross_entropy_value(self):
        """Tests categorical cross entropy with a known random distribution."""
        cce = CategoricalCrossEntropy()
        
        # Creates a simple 2-sample, 3-class test
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]])
        
        loss = cce.loss(y_true, y_pred)
        
        # Manual calculation: - (ln(0.5) + ln(0.8)) = 0.6931 + 0.2231 = 0.9162
        expected_loss = -(np.log(0.5) + np.log(0.8))
        self.assertAlmostEqual(loss, expected_loss, places=5)