from typing import List, Optional, Tuple, Literal, Union

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier

Mode = Literal["gini", "entropy"]


class RandomForestClassifier(Model):
    """
    Random Forest Classifier.

    Parameters
    ----------
    n_estimators : int, optional
        Number of decision trees to use (default is 100).
    max_features : int, optional
        Maximum number of features to use per tree (default is None,
        which uses sqrt(n_features)).
    min_sample_split : int, optional
        Minimum samples allowed in a split (default is 2).
    max_depth : int, optional
        Maximum depth of the trees (default is None).
    mode : {'gini', 'entropy'}, optional
        Impurity calculation mode (default is 'gini').
    seed : int, optional
        Random seed for reproducibility (default is 42).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Optional[int] = None,
        min_sample_split: int = 2,
        max_depth: Optional[int] = None,
        mode: Mode = "gini",
        seed: int = 42,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        # list of (feature_indices, trained_tree)
        self.trees: List[Tuple[np.ndarray, DecisionTreeClassifier]] = []

    def _fit(self, dataset: Dataset) -> "RandomForestClassifier":
        """
        Train the decision trees of the random forest.
        """
        # 1. Sets the random seed
        rng = np.random.RandomState(self.seed)

        n_samples, n_features = dataset.X.shape

        # 2. Define self.max_features to be int(sqrt(n_features)) if None
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        # Choose an effective max_depth to pass to the trees
        # If self.max_depth is None, use a reasonable default (e.g., 10)
        effective_max_depth = self.max_depth if self.max_depth is not None else 10

        self.trees = []

        for _ in range(self.n_estimators):
            # 3. Create a bootstrap dataset:
            #    pick n_samples with replacement
            bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = dataset.X[bootstrap_indices]
            y_bootstrap = dataset.y[bootstrap_indices]

            #    pick self.max_features random features without replacement
            feature_indices = rng.choice(
                n_features, size=self.max_features, replace=False
            )
            X_bootstrap_sub = X_bootstrap[:, feature_indices]

            bootstrap_dataset = Dataset(X=X_bootstrap_sub, y=y_bootstrap)

            # 4. Create and train a decision tree
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=effective_max_depth,  # <- never None now
                mode=self.mode,
                seed=self.seed,
            )
            tree._fit(bootstrap_dataset)

            # 5. Append (features_used, tree)
            self.trees.append((feature_indices, tree))

        # 7. Return itself
        return self


    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the labels using the ensemble of decision trees.
        """
        if not self.trees:
            raise RuntimeError("The model has not been fitted yet.")

        n_samples = dataset.X.shape[0]
        # store raw labels (strings) instead of ints
        predictions = np.empty((n_samples, self.n_estimators), dtype=object)

        # 1. Get predictions for each tree using respective features
        for i, (feature_indices, tree) in enumerate(self.trees):
            X_sub = dataset.X[:, feature_indices] #this is selecting the random features to use in one tree 
            tree_preds = tree._predict(Dataset(X=X_sub, y=None)) #da classe decision tree classifier #usar o método público 
            # no astype(int) here
            predictions[:, i] = tree_preds  #all samples for tree i 

        # 2. Most common predicted class for each sample (mode over strings)
        y_pred = np.empty(n_samples, dtype=object) #empty 1D array for final predictions 
        for i in range(n_samples): #apply along axis como no knn classifier 
            # count occurrences of each label
            values, counts = np.unique(predictions[i, :], return_counts=True)
            # pick the label with the highest count
            y_pred[i] = values[counts.argmax()]

        # 3. Return predictions as a numpy array of labels
        return y_pred


    def _score(
        self,
        dataset: Dataset,
        predictions: Optional[Union[np.ndarray, list]] = None,
    ) -> float:
        """
        Compute the accuracy between predicted and real labels.
        """
        # 1. Get predictions if not provided
        if predictions is None:
            predictions = self._predict(dataset)

        # 2. Compute accuracy
        return accuracy(dataset.y, predictions)
