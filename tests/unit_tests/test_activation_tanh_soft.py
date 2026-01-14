import os
import sys
import unittest
import numpy as np

# Setup to ensure 'si' is found
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from datasets import DATASETS_PATH
from si.io.data_file import read_data_file
from si.neural_networks.activation import TanhActivation, SoftmaxActivation
class TestActivationFunction(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    # --- Tanh Layer Tests ---

    def test_tanh_activation_function(self):
        """Test the activation_function logic of Tanh."""
        tanh_layer = TanhActivation()
        result = tanh_layer.activation_function(self.dataset.X)
        
        # Verify range: [-1, 1]
        self.assertTrue(np.all((result >= -1) & (result <= 1)))
        # Verify shape consistency
        self.assertEqual(result.shape, self.dataset.X.shape)

    def test_tanh_derivative(self):
        """Test the derivative logic of Tanh (1 - tanh^2)."""
        tanh_layer = TanhActivation()
        derivative = tanh_layer.derivative(self.dataset.X)
        
        # Verify range: [0, 1]
        self.assertTrue(np.all((derivative >= 0) & (derivative <= 1)))
        # Verify shape consistency
        self.assertEqual(derivative.shape, self.dataset.X.shape)

    # --- Softmax Layer Tests ---

    def test_softmax_activation_function(self):
        """Test the activation_function logic of Softmax."""
        softmax_layer = SoftmaxActivation()
        result = softmax_layer.activation_function(self.dataset.X)
        
        # Verify desired behaviour: Probabilities between 0 and 1
        self.assertTrue(np.all((result >= 0) & (result <= 1)))
        
        # Verify desired behaviour: Rows must sum to 1.0
        self.assertTrue(np.allclose(np.sum(result, axis=1), 1.0))
        
        # Verify shape consistency
        self.assertEqual(result.shape, self.dataset.X.shape)

    def test_softmax_derivative(self):
        """Test the derivative logic of Softmax (s * (1 - s))."""
        softmax_layer = SoftmaxActivation()
        derivative = softmax_layer.derivative(self.dataset.X)
        
        # Verify range: [0, 0.25]
        self.assertTrue(np.all((derivative >= 0) & (derivative <= 0.25)))
        # Verify shape consistency
        self.assertEqual(derivative.shape, self.dataset.X.shape)

if __name__ == '__main__':
    unittest.main()