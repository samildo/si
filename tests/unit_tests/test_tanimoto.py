import unittest
import numpy as np
from si.statistics.tanimoto import tanimoto_similarity 

class TestTanimotoSimilarity(unittest.TestCase):
    def test_tanimoto_similarity(self):
        """Test the Tanimoto similarity function with simple binary vectors."""
        # Test data (binary)
        x = np.array([1, 1, 0, 0])
        y = np.array([[1, 1, 0, 0],  # Identical to x
                       [1, 0, 0, 0],  # One bit different
                       [0, 0, 0, 0]]) # Completely different

        # Execute your function
        our_similarity = tanimoto_similarity(x, y)

        # Expected values calculated manually
        # For x and y[0]: (1*1 + 1*1 + 0*0 + 0*0) / (2 + 2 - 2) = 2/2 = 1.0
        # For x and y[1]: (1*1 + 1*0 + 0*0 + 0*0) / (2 + 1 - 1) = 1/2 = 0.5
        # For x and y[2]: (1*0 + 1*0 + 0*0 + 0*0) / (2 + 0 - 0) = 0/2 = 0.0
        expected_similarity = np.array([1.0, 0.5, 0.0])

        # Verification
        np.testing.assert_allclose(our_similarity, expected_similarity)