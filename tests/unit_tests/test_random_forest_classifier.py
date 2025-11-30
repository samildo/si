from unittest import TestCase
import os

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.random_forest_classifier import RandomForestClassifier


class TestRandomForestClassifier(TestCase):

    def setUp(self):
        # 1. Use the iris.csv dataset
        self.csv_file = os.path.join(DATASETS_PATH, "iris", "iris.csv")
        self.dataset = read_csv(
            filename=self.csv_file,
            features=True,
            label=True
        )
        # 2. Split the data into train and test sets
        self.train_dataset, self.test_dataset = train_test_split(
            self.dataset,
            test_size=0.2,
            random_state=42
        )

    def test_fit(self):
        model = RandomForestClassifier(
            n_estimators=10,
            max_features=None,
            min_sample_split=2,
            max_depth=10,
            mode="gini",
            seed=42
        )

        model._fit(self.train_dataset)

        # hyperparameters kept
        self.assertEqual(model.n_estimators, 10)
        self.assertEqual(model.min_sample_split, 2)
        self.assertEqual(model.max_depth, 10)

        # trees were actually built
        self.assertEqual(len(model.trees), model.n_estimators)
        feat_idx, tree = model.trees[0]
        self.assertIsNotNone(feat_idx)
        self.assertGreater(len(feat_idx), 0)
        self.assertIsNotNone(tree)

    def test_predict(self):
        model = RandomForestClassifier(
            n_estimators=10,
            max_depth=10,
            seed=42
        )
        model._fit(self.train_dataset)

        # 3. Predict labels on the test set
        predictions = model._predict(self.test_dataset)

        # shape check like in the reference tests
        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])

    def test_score(self):
        # 3. Create the RandomForestClassifier model
        model = RandomForestClassifier(
            n_estimators=100,
            max_features=None,
            min_sample_split=2,
            max_depth=10,   # ensure an int, not None, for DecisionTree
            mode="gini",
            seed=42
        )

        # 4. Train the model
        model._fit(self.train_dataset)

        # get score directly
        score = model._score(self.test_dataset)

        print("RandomForestClassifier Iris test score:", score)

        #compare to rounded expected value or threshold
        # start with a safe threshold, then tighten once you see the actual score
        self.assertGreaterEqual(round(score, 2), 0.90)
