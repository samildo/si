import numpy as np
from typing import Dict, Any, Callable
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation

def randomized_search_cv(model,
                         dataset: Dataset,
                         hyperparameter_grid: Dict[str, Any],
                         scoring: Callable = None,
                         cv: int = 5,
                         n_iter: int = 10) -> Dict[str, Any]:
    """
    Implements a randomized search cross-validation strategy for hyperparameter optimization.

    Parameters
    ----------
    model
        The model to validate.
    dataset : Dataset
        The validation dataset.
    hyperparameter_grid : Dict[str, Any]
        Dictionary with hyperparameter names as keys and distributions/values as values.
    scoring : Callable
        The scoring function to use.
    cv : int
        Number of folds for cross-validation.
    n_iter : int
        Number of hyperparameter random combinations to test.

    Returns
    -------
    results : Dict[str, Any]
        Dictionary containing scores, hyperparameters, best hyperparameters, and best score.
    """
    # 1. Check if the provided hyperparameters are valid
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter '{parameter}'.")

    results = {
        'hyperparameters': [],
        'scores': [],
        'best_hyperparameters': None,
        'best_score': -np.inf
    }

    # 2. Generate random combinations
    #n_iter iterations to pick random samples from the provided distributions
    for _ in range(n_iter): #6. Repeat steps 3, 4 and 5 for all hyperparameter combinations.
        combination = {}
        for parameter, values in hyperparameter_grid.items():
            # Randomly select one value from the list/array of possible values
            combination[parameter] = np.random.choice(values)

        # 3. Set the model hyperparameters
        for parameter, value in combination.items():
            setattr(model, parameter, value)

        # 4. Cross validate the model
        # Assumes k_fold_cross_validation returns a list of scores
        scores = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # 5. Save the mean of the scores and hyperparameters
        mean_score = np.mean(scores)
        results['scores'].append(mean_score)
        results['hyperparameters'].append(combination)

        # 7. Update best score and best hyperparameters
        if mean_score > results['best_score']:
            results['best_score'] = mean_score
            results['best_hyperparameters'] = combination

    return results