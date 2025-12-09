from typing import Callable, Optional

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse     
from si.statistics.euclidean_distance import euclidean_distance  


class KNNRegressor(Model):
    """
    K-Nearest Neighbors Regressor.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to consider.
    distance : Callable
        Function that computes distances from one point to a set of points.
        Signature: distance(x: np.ndarray, y: np.ndarray) -> np.ndarray
    """

    def __init__(
        self,
        k: int = 3,
        distance: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean_distance,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset: Optional[Dataset] = None

    def _fit(self, dataset: Dataset) -> "KNNRegressor":
        """
        Stores the training dataset.
        """
        self.dataset = dataset
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts labels for the test dataset using KNN regression.
        """
        if self.dataset is None:
            raise RuntimeError("The model has not been fitted yet.")

        X_train = self.dataset.X
        y_train = self.dataset.y
        X_test = dataset.X

        n_test = X_test.shape[0]
        y_pred = np.zeros(n_test, dtype=float) #array of zeros same size as the sample size

        for i in range(n_test): #apply along axis. 
            x = X_test[i]

            # 1. Distances from x to all training points (vectorized)
            distances = self.distance(x, X_train)  # shape: (n_train,)

            # 2. Indices of k closest neighbors
            nn_idx = np.argsort(distances)[: self.k]

            # 3. Corresponding target values
            nn_y = y_train[nn_idx]

            # 4. Mean of neighbor targets
            y_pred[i] = float(np.mean(nn_y))

        # 5. Return predictions for all test samples
        return y_pred

    def _score(self, dataset: Dataset) -> float:
        """
        Returns RMSE between predictions and true values.
        """
        if self.dataset is None:
            raise RuntimeError("The model has not been fitted yet.")

        y_pred = self._predict(dataset)
        return rmse(dataset.y, y_pred)
