import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier(Model):
    """
    Ensemble classifier that uses a set of models to generate predictions,
    which are then used to train a final model (meta-learner).
    """
    def __init__(self, models: list, final_model: Model):
        """
        Initialize the StackingClassifier.

        Parameters
        ----------
        models : list
            The initial set of models (base learners).
        final_model : Model
            The model to make the final predictions (meta-learner).
        """
        super().__init__()
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Trains the ensemble models and the final model.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : StackingClassifier
            The trained model.
        """
        # 1. Train the initial set of models
        for model in self.models:
            model.fit(dataset)

        # 2. Get predictions from the initial set of models
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # Stack predictions horizontally (n_samples, n_models)
        meta_features = np.stack(predictions, axis=1)

        # 3. Train the final model with the predictions of the initial set
        self.final_model.fit(Dataset(meta_features, dataset.y))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the labels using the ensemble models.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        predictions : np.ndarray
            The final predictions.
        """
        # 1. Get predictions from the initial set of models
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # Stack predictions
        meta_features = np.stack(predictions, axis=1)

        # 2. Get the final predictions using the final model
        final_predictions = self.final_model.predict(Dataset(meta_features, dataset.y))

        return final_predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray = None) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            The test data.
        predictions: np.ndarray
            Predictions

        Returns
        -------
        score : float
            Mean accuracy
        """
        if predictions is None:
            predictions = self.predict(dataset)
            
        return accuracy(dataset.y, predictions)