import os
import sys
import numpy as np
from unittest import TestCase

# Setup to ensure 'si' is found
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from datasets import DATASETS_PATH
from si.io.data_file import read_data_file
from si.models.logistic_regression import LogisticRegression
from si.model_selection.randomized_search import randomized_search_cv

class TestRandomizedSearch(TestCase):

    def setUp(self):
        # 1. Use the breast-bin.csv dataset
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_protocol_exercise_11(self):
        # 2. Create a LogisticRegression model
        lg = LogisticRegression()

        # 3. Define hyperparameter distributions
        hp_grid = {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': np.linspace(1000, 2000, 200).astype(int) #cannot be a float (astype)
        }

        # 4. Perform randomized search (n_iter=10, cv=3)
        results = randomized_search_cv(
            model=lg,
            dataset=self.dataset,
            hyperparameter_grid=hp_grid,
            cv=3,
            n_iter=10
        )

        # 5. Output and Verification
        print("\n--- Results ---")
        for i, score in enumerate(results['scores']):
            print(f"Combination {i+1}: Score = {score:.4f}")
        
        print(f"\nBest Score: {results['best_score']:.4f}")
        print(f"Best Hyperparameters: {results['best_hyperparameters']}")

        # Assertions to ensure individual parts worked
        self.assertEqual(len(results['scores']), 10)
        self.assertIsNotNone(results['best_hyperparameters'])
        self.assertGreater(results['best_score'], 0.90) # Accuracy is usually > 95%