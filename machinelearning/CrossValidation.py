from copy import copy

import numpy as np
from sklearn.model_selection import KFold

from machinelearning.Bootstrap import Bootstrap
from machinelearning.utils import validate_data


def kfold_cross_validation(
    bs: Bootstrap, x: np.ndarray, y: np.ndarray, f=None, n_splits=10
) -> dict:
    """Simple implementation of KFold CV directly for Bootstrap class."""
    validate_data(x, y)

    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf = KFold(n_splits=n_splits, shuffle=True)

    scores = {
        "test_r2": np.zeros(n_splits),
        "test_adj_r2": np.zeros(n_splits),
        "test_mse": np.zeros(n_splits),
        "test_bs_mse": np.zeros(n_splits),
        "test_bias_y_hat": np.zeros(n_splits),
        "test_var_y_hat": np.zeros(n_splits),
        "test_cond_num": np.zeros(n_splits),
        "train_r2": np.zeros(n_splits),
        "train_adj_r2": np.zeros(n_splits),
        "train_mse": np.zeros(n_splits),
        "train_bs_mse": np.zeros(n_splits),
        "train_bias_y_hat": np.zeros(n_splits),
        "train_var_y_hat": np.zeros(n_splits),
        "train_cond_num": np.zeros(n_splits),
    }

    for i, idx in enumerate(kf.split(x)):
        x_train, x_test = x[idx[0]], x[idx[1]]
        y_train, y_test = y[idx[0]], y[idx[1]]

        bs.fit(x_train, y_train)
        estimator = copy(bs.estimator).fit(x_train, y_train)

        scores["test_r2"][i] = estimator.score(x_test, y_test)
        scores["test_adj_r2"][i] = estimator.adj_r2_score(x_test, y_test)
        scores["test_mse"][i] = estimator.mse_score(x_test, y_test)
        scores["test_bs_mse"][i] = bs.mse_score(x_test, y_test)
        scores["test_bias_y_hat"][i] = bs.square_bias_score(x_test, f)
        scores["test_var_y_hat"][i] = bs.y_hat_variance_score(x_test)
        scores["test_cond_num"][i] = np.linalg.cond(estimator.transform(x_train))

        scores["train_r2"][i] = estimator.score(x_train, y_train)
        scores["train_adj_r2"][i] = estimator.adj_r2_score(x_train, y_train)
        scores["train_mse"][i] = estimator.mse_score(x_train, y_train)
        scores["train_bs_mse"][i] = bs.mse_score(x_train, y_train)
        scores["train_bias_y_hat"][i] = bs.square_bias_score(x_train, f)
        scores["train_var_y_hat"][i] = bs.y_hat_variance_score(x_train)
        scores["train_cond_num"][i] = np.linalg.cond(estimator.transform(x_train))

    return {k: np.mean(v) for k, v in scores.items()}
