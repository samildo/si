from unittest import TestCase
import os

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.knn_regressor import KNNRegressor
from si.statistics.euclidean_distance import euclidean_distance


class TestKNNRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, "cpu", "cpu.csv")
        self.dataset = read_csv(
            filename=self.csv_file,
            features=True,
            label=True
        )
        # Train/test split
        self.train_dataset, self.test_dataset = train_test_split(
            self.dataset,
            test_size=0.2,
            random_state=42
        )

    def test_fit(self):
        """
        Test if KNNRegressor correctly stores the training dataset.
        """
        knn = KNNRegressor(k=3, distance=euclidean_distance)
        knn._fit(self.train_dataset)

        # The training dataset should be stored
        self.assertIsNotNone(knn.dataset)
        self.assertEqual(knn.dataset.X.shape, self.train_dataset.X.shape)
        self.assertEqual(knn.dataset.y.shape, self.train_dataset.y.shape)

    def test_predict(self):
        """
        Test if predictions have the correct shape.
        """
        knn = KNNRegressor(k=3, distance=euclidean_distance)
        knn._fit(self.train_dataset)

        y_pred = knn._predict(self.test_dataset)

        # Same number of predictions as test samples
        self.assertEqual(y_pred.shape[0], self.test_dataset.X.shape[0])

    def test_score(self):
        """
        Test the RMSE score on the test set.
        """
        knn = KNNRegressor(k=3, distance=euclidean_distance)
        knn._fit(self.train_dataset)

        error = knn._score(self.test_dataset)
        print("KNNRegressor RMSE on test set:", error)

        self.assertGreaterEqual(error, 0.0)
