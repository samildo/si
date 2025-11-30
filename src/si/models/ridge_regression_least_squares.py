import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from typing import Optional

class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression using the least squares method with L2 regularization.

    Parameters
    ----------
    l2_penalty : float, optional
        L2 regularization parameter (default is 1.0).
    scale : bool, optional
        Whether to scale the data (default is True).
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta = None  # Coefficients for features
        self.theta_zero = None  # Intercept term
        self.mean = None  # Mean of features (for scaling)
        self.std = None  # Standard deviation of features (for scaling)

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fit the Ridge Regression model.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.

        Returns
        -------
        RidgeRegressionLeastSquares
            The fitted model.
        """
        X = dataset.X
        y = dataset.y

        # Step 1: Scale the data if required
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean) / self.std

        # Step 2: Add intercept term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # Step 3: Compute the penalty matrix
        penalty_matrix = self.l2_penalty * np.eye(X.shape[1])
        penalty_matrix[0, 0] = 0  # Do not penalize the intercept term

        # Step 4: Compute the model parameters
        XTX = X.T @ X
        XTX_penalty = XTX + penalty_matrix
        XTX_penalty_inv = np.linalg.inv(XTX_penalty)
        XTy = X.T @ y
        thetas = XTX_penalty_inv @ XTy

        # Step 5: Extract theta_zero and theta
        self.theta_zero = thetas[0]
        self.theta = thetas[1:]

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the dependent variable (y) using the estimated coefficients.

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict.

        Returns
        -------
        np.ndarray
            The predicted y values.
        """
        X = dataset.X

        # Step 1: Scale the data if required
        if self.scale:
            X = (X - self.mean) / self.std

        # Step 2: Add intercept term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # Step 3: Compute the predicted y
        thetas = np.r_[self.theta_zero, self.theta]
        y_pred = X @ thetas

        return y_pred

    def _score(self, dataset: Dataset, predictions: Optional[np.ndarray] = None) -> float:
        """
        Calculate the MSE score between the true and predicted y values.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing the true y values.
        predictions : np.ndarray, optional
            The predicted y values. If None, they are computed using _predict.

        Returns
        -------
        float
            The MSE score.
        """
        if predictions is None:
            predictions = self._predict(dataset)

        mse = np.mean((dataset.y - predictions) ** 2)
        return mse


