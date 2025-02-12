import numpy as np

def relative_error_within_boundaries(x: np.array, low: np.array, high: np.array) -> np.array: # type: ignore
    """
    Calculate the relative error of values within specified boundaries.

    Args:
        x (np.array): The array of values.
        low (np.array): The lower boundary array.
        high (np.array): The upper boundary array.

    Returns:
        np.array: The array of relative errors.
    """
    return np.abs(error_within_boundaries(x, low, high))/x

def error_within_boundaries(x: np.array, low: np.array, high: np.array) -> np.array: # type: ignore
    """
    Calculate the error of values within specified boundaries.

    Args:
        x (np.array): The array of values.
        low (np.array): The lower boundary array.
        high (np.array): The upper boundary array.

    Returns:
        np.array: The array of errors.
    """
    nearest_boundary = np.where(x < low, low, np.where(x > high, high, x))
    return x - nearest_boundary