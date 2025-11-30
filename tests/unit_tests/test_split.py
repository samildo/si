from unittest import TestCase

from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv
import numpy as np

from si.model_selection.split import train_test_split, stratified_train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

    def test_stratified_train_test_split(self):
        """Test the stratified_train_test_split function."""
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)

        # Calculate expected test size
        test_samples_size = int(len(self.dataset.y) * 0.2)

        # Check that the sizes are correct
        self.assertEqual(len(test.y), test_samples_size)
        self.assertEqual(len(train.y), len(self.dataset.y) - test_samples_size)

        # Check that the features and label are preserved
        self.assertListEqual(list(train.features), list(self.dataset.features))
        self.assertListEqual(list(test.features), list(self.dataset.features))
        self.assertEqual(train.label, self.dataset.label)
        self.assertEqual(test.label, self.dataset.label)

        # Check that the class distribution is preserved
        unique_labels, original_counts = np.unique(self.dataset.y, return_counts=True)
        train_labels, train_counts = np.unique(train.y, return_counts=True)
        test_labels, test_counts = np.unique(test.y, return_counts=True)

        # Verify that all classes are represented in both train and test sets
        self.assertTrue(np.array_equal(np.sort(unique_labels), np.sort(train_labels)))
        self.assertTrue(np.array_equal(np.sort(unique_labels), np.sort(test_labels)))

        # Check that the ratio of classes is approximately preserved
        for i, label in enumerate(unique_labels):
            original_ratio = original_counts[i] / len(self.dataset.y)
            train_ratio = train_counts[i] / len(train.y)
            test_ratio = test_counts[i] / len(test.y)

            # Allow for small differences due to rounding
            self.assertAlmostEqual(original_ratio, train_ratio, delta=0.05)
            self.assertAlmostEqual(original_ratio, test_ratio, delta=0.05)

    def test_stratified_split_reproducibility(self):
        """Test that the stratified split is reproducible with the same random_state."""
        train1, test1 = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
        train2, test2 = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)

        # Check that the results are identical with the same random_state
        np.testing.assert_array_equal(train1.y, train2.y)
        np.testing.assert_array_equal(test1.y, test2.y)
