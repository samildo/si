import numpy as np


from si.base.transformer import Transformer
from si.data.dataset import Dataset


class PCA(Transformer):
    """
    Principal Component Analysis (PCA) for dimensionality reduction.

    PCA transforms the data to a lower-dimensional space by projecting it onto the principal components,
    which are the directions of maximum variance in the data.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep. Must be less than or equal to the number of features in the dataset.

    Attributes
    ----------
    mean : numpy.ndarray
        Mean of the training data, used for centering.
    components : numpy.ndarray
        Principal components (eigenvectors) of the covariance matrix, stored as columns.
        Shape: (n_components, n_features).
    explained_variance : numpy.ndarray
        Explained variance ratio for each principal component.
    is_fitted : bool
        Flag indicating whether the model has been fitted to data.
    """
    def __init__(self, n_components,**kwargs):
        """
        Initialize the PCA transformer.

        Parameters
        ----------
        n_components : int
            Number of principal components to keep.
        **kwargs : dict
            Additional keyword arguments for the base Transformer class.
        """
        super().__init__(**kwargs)

        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance= None
        self.is_fitted = False ##why is this here ?

    def _fit(self, dataset: Dataset) -> 'PCA':
        """
        Fit the PCA model using eigenvalue decomposition of the covariance matrix.

        Parameters
        ----------
        dataset : Dataset
            Dataset object containing the data to fit.

        Returns
        -------
        PCA
            The fitted PCA object.
        """
        # 1. Center the data
        self.mean = np.mean(dataset.X, axis=0)
        X_centered = dataset.X - self.mean

        # 2. Calculate the covariance matrix and perform eigenvalue decomposition
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 3. Infer the principal components (eigenvectors corresponding to the top n eigenvalues)
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components eigenvectors
        self.components = sorted_eigenvectors[:, :self.n_components].T

        # 4. Infer the explained variance
        total_variance = np.sum(eigenvalues)
        self.explained_variance = sorted_eigenvalues[:self.n_components] / total_variance

        self.is_fitted= True
        return self
    
    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the data using the principal components.

        Parameters
        ----------
        dataset : Dataset
            Input dataset to transform. The data will be centered using the mean from the training data.

        Returns
        -------
        Dataset
            Transformed dataset with features projected onto the principal components.
            The new features are labeled as "PC1", "PC2", etc.
        """
        # 1. Center the data
        X_centered = dataset.X - self.mean
        
        # 2. Project data onto principal components
        X_reduced = np.dot(X_centered, self.components.T)
        return Dataset(X_reduced, y= dataset.y, features=[f"PC{i+1}" for i in range(self.n_components)], label= dataset.label)
