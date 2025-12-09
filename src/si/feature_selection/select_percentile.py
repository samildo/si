import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics import f_classification


class SelectPercentile(Transformer):
    """
    Select features based on a percentile of the highest scoring features using a scoring function.

    This transformer selects features based on their scores (e.g., F-values or p-values)
    computed using a scoring function (e.g., ANOVA F-value). The top `percentile` of features
    with the highest scores are selected.

    Parameters
    ----------
    score_func : Callable, optional
        The scoring function to compute feature scores. Default is `f_classif` (ANOVA F-value).
    percentile : int, optional
        The percentile of features to select. For example, `percentile=40` selects the top 40%
        of features with the highest scores. Default is 10.

    Attributes
    ----------
    F : np.ndarray
        The F-values for each feature, computed by `score_func`.
    p : np.ndarray
        The p-values for each feature, computed by `score_func`.
    """

    
    def __init__(self, percentile: float, score_func: callable = f_classification, **kwargs):
        super().__init__(**kwargs)
        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be a float between 0 and 100")
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None
        return

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Fit the transformer by computing the F and p values for each feature.

        Parameters
        ----------
        dataset : Dataset
            The input dataset containing features (X) and labels (y).

        Returns
        -------
        SelectPercentile
            The fitted transformer.
        """
        self.F, self.p = self.score_func(dataset)
        return self
    
    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset by selecting the top `percentile` of features based on their F-values.

        The method selects features with F-values greater than the threshold determined by the
        specified percentile. If there are ties at the threshold, the method includes enough tied
        features to meet the required number of selected features.

        Parameters
        ----------
        dataset : Dataset
            The input dataset to transform.

        Returns
        -------
        Dataset
            The transformed dataset with only the selected features.
        """
        threshold = np.percentile(self.F, 100 - self.percentile) # Find cutoff F-value at specified percentile
        
        mask = self.F >= threshold

        if mask.sum() > int(len(self.F) * self.percentile / 100): #if it has too many features (ties)
            sorted_indices = np.argsort(-self.F)  # Sort feature indices by descending F-value
            num_features = int(len(self.F) * self.percentile / 100) # Exact number of features to select
            selected_indices = sorted_indices[:num_features] # Pick top features by index
            mask = np.zeros_like(self.F, dtype=bool)  # Create empty mask
            mask[selected_indices] = True # Mark top indices as selected

        selected_features = np.array(dataset.features)[mask]  # Extract feature names for selected features

        # Return new Dataset with only selected features and original labels
        return Dataset(X=dataset.X[:, mask], y=dataset.y, features=list(selected_features), label=dataset.label)