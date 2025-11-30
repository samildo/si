import numpy as np
from typing import Union, List

def tanimoto_similarity(x: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> np.ndarray:
    """
    Computes the Tanimoto similarity between a single binary sample and multiple binary samples.

    Parameters
    ----------
    x : 1D array or list
        A single binary sample.
    y : 2D array or list of lists
        Multiple binary samples, where each row is a sample.

    Returns
    -------
    np.ndarray
        An array containing the Tanimoto similarities between x and each sample in y.
    """

    # Ensure x is 1D and y is 2D
    if x.ndim != 1:
        raise ValueError("x must be a 1D array or list.")
    if y.ndim != 2:
        raise ValueError("y must be a 2D array or list of lists.")

    # Compute dot products and norms
    dot_products = np.dot(y, x)
    x_norm_sq = np.sum(x ** 2)
    y_norm_sq = np.sum(y ** 2, axis=1)

    # Compute Tanimoto similarity
    similarities = dot_products / (x_norm_sq + y_norm_sq - dot_products)

    return similarities