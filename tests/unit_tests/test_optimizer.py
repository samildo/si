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
from si.neural_networks.optimizers import Adam

class TestAdam(unittest.TestCase):

    def setUp(self):
        # Using breast-bin data context for weight shapes
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        
        # Mocking 9 features to 1 output weight matrix
        self.w = np.random.randn(9, 1)
        self.grad = np.random.randn(9, 1)

    def test_initialization_and_t(self):
        """Individual Part: Test that m and v initialize on first update and t increments."""
        optimizer = Adam()
        self.assertEqual(optimizer.t, 0)
        self.assertIsNone(optimizer.m)
        
        optimizer.update(self.w, self.grad)
        
        self.assertEqual(optimizer.t, 1)
        self.assertIsNotNone(optimizer.m)
        self.assertEqual(optimizer.m.shape, self.w.shape)

    def test_weight_update_logic(self):
        """Individual Part: Verify that weights actually change after an update."""
        optimizer = Adam(learning_rate=0.1)
        original_w = self.w.copy()
        
        updated_w = optimizer.update(self.w, self.grad)
        
        # Ensure weights are no longer the same
        self.assertFalse(np.array_equal(original_w, updated_w))
        # Ensure shape is preserved
        self.assertEqual(updated_w.shape, original_w.shape)

    def test_numerical_stability(self):
        """Individual Part: Ensure epsilon handles zero gradients without crashing."""
        optimizer = Adam(epsilon=1e-8)
        zero_grad = np.zeros_like(self.w)
        
        # This should return the original weights without NaN errors
        updated_w = optimizer.update(self.w, zero_grad)
        self.assertFalse(np.isnan(updated_w).any())
        np.testing.assert_allclose(updated_w, self.w)

if __name__ == '__main__':
    unittest.main()