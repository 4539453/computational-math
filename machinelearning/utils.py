import numpy as np


def _validate_targets_dimension(y: np.ndarray):
    if y.ndim != 2:
        raise ValueError("Must be a 2D array.")
    if len(y[0, :]) != 1:
        raise ValueError("Oh, several targets. Interesting...")


def _valdate_dimensions_consistency(x: np.ndarray, y: np.ndarray):
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples.")


def validate_features_dimension(x: np.ndarray):
    if x.ndim != 2:
        raise ValueError("Must be a 2D array.")
    # if len(x[0, :]) != 1:
    #     raise ValueError("Stop! You're not prepared for multi features!")


def validate_data(x: np.ndarray, y: np.ndarray):
    validate_features_dimension(x)
    _validate_targets_dimension(y)
    _valdate_dimensions_consistency(x, y)
