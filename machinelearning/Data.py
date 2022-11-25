from collections.abc import Callable

import numpy as np

from machinelearning.utils import validate_data, validate_features_dimension


def _generate_y(x, f, mean_sd: tuple[float, float]):
    y: np.ndarray = f(x)
    residuals = np.random.normal(*mean_sd, y.shape[0]).reshape((-1, 1))
    return y + residuals


def _add_outliers(x, y, outliers):
    outliers = tuple(arr.reshape((-1, 1)) if arr.ndim == 1 else arr for arr in outliers)
    x = np.concatenate((x, outliers[0]), axis=0)
    y = np.concatenate((y, outliers[1]), axis=0)
    return x, y


def generate_functional_data(
    x: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    mean_sd: tuple[float, float],
    outliers=(None, None),
) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim == 1:
        x = x.reshape((-1, 1))

    validate_features_dimension(x)

    y = _generate_y(x, f, mean_sd)

    if all(outliers) and len(outliers) == 2:
        x, y = _add_outliers(x, y, outliers)

    validate_data(x, y)
    return x, y
