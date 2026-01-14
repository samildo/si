import unittest
import numpy as np

from si.neural_networks.layers import Dropout

class TestDropoutRandom(unittest.TestCase):

    def setUp(self):
        # Use random dimensions and data
        self.input_shape = (10, 10)
        self.input_data = np.random.randn(*self.input_shape)
        self.prob = 0.5
        self.layer = Dropout(probability=self.prob)
        # Mocking the input_shape attribute
        self.layer.input_shape = self.input_shape

    def test_forward_training_behavior(self):
        """Individual Part: Test training mode using random data."""
        # desired behaviour: applies mask and scaling
        output = self.layer.forward_propagation(self.input_data, training=True)
        
        # 1. Check scaling: non-zero elements should be input * (1 / (1 - prob))
        scaling_factor = 1 / (1 - self.prob)
        mask = self.layer.mask
        
        # Verify scaling on non-dropped neurons
        survived_indices = (mask == 1)
        if np.any(survived_indices):
            actual_scaling = output[survived_indices] / self.input_data[survived_indices]
            np.testing.assert_allclose(actual_scaling, scaling_factor)
        
        # 2. Check mask: some values must be 0
        self.assertTrue(np.any(output == 0))

    def test_forward_inference_behavior(self):
        """Test inference mode using random data."""
        # desired behaviour: returns received input
        output = self.layer.forward_propagation(self.input_data, training=False)
        np.testing.assert_array_equal(output, self.input_data)

    def test_backward_propagation(self):
        """Test backward pass using random data."""
        # desired behaviour: multiplies error by the mask
        self.layer.forward_propagation(self.input_data, training=True)
        
        output_error = np.random.randn(*self.input_shape)
        input_error = self.layer.backward_propagation(output_error)
        
        expected_error = output_error * self.layer.mask
        np.testing.assert_array_equal(input_error, expected_error)

    def test_output_shape(self):
        """Individual Part: Verify shape remains the same."""
        self.assertEqual(self.layer.output_shape(), self.input_shape)

    def test_parameters(self):
        """Individual Part: Verify 0 parameters."""
        self.assertEqual(self.layer.parameters(), 0)

if __name__ == '__main__':
    unittest.main()