from copy import copy

import numpy as np
from sklearn.metrics import r2_score

from machinelearning.LinearRegression import LinearRegression
from machinelearning.Sampler import Sampler
from machinelearning.utils import validate_data, validate_features_dimension


class Bootstrap:
    def __init__(self, estimator, sampler: Sampler, n_bootstraps: int = 1000):
        self.__estimator = estimator
        self.__sampler = sampler
        self.__n_bootstraps = n_bootstraps
        self.__w_hats = None
        self.__estimator_w_hat = None

    @property
    def estimator(self):
        return self.__estimator

    @estimator.setter
    def estimator(self, obj):
        if not isinstance(obj, LinearRegression):
            wrong_estimator_type_message = (
                "\nEstimator should be instance of LinearRegression"
                f"your estimator is {type(obj)}\n\n"
            )
            raise TypeError(wrong_estimator_type_message)
        self.__estimator = obj

    @property
    def w_hats(self):
        if self.__w_hats is None:
            raise ValueError("w_hats not generated yet")
        return self.__w_hats

    @property
    def estimator_w_hat(self):
        if self.__estimator_w_hat is None:
            raise ValueError("Estimator's w_hat is not calculated yet")
        return self.__estimator_w_hat

    def fit(self, x: np.ndarray, y: np.ndarray):
        validate_data(x, y)
        estimator = copy(self.estimator)
        self.__estimator_w_hat = estimator.fit(x, y).w_hat
        self.__sampler.fit(x, y)
        self.__w_hats = np.zeros((estimator.n_ivs + 1, self.__n_bootstraps))

        for i in range(self.__n_bootstraps):
            # bootstrap sample
            x_boot, y_boot = next(self.__sampler)

            # fit model
            estimator.fit(x_boot, y_boot)

            # save w_hat
            self.w_hats[:, i] = estimator.w_hat.flatten()

        return self

    def predict(self, x: np.ndarray):
        validate_features_dimension(x)
        return self.estimator.transform(x) @ self.w_hats

    def score(self, x: np.ndarray, y: np.ndarray):
        validate_data(x, y)
        y_tiled = np.tile(y, (self.__n_bootstraps))
        return r2_score(y_tiled, self.predict(x), multioutput="raw_values")

    # TODO: move scores to estimator class
    # differet scores for different estimators
    def mse_score(self, x: np.ndarray, y: np.ndarray):
        validate_data(x, y)
        return np.mean(np.mean((self.predict(x) - y) ** 2, axis=1))

    def y_hat_variance_score(self, x: np.ndarray):
        validate_features_dimension(x)
        """Calculate the mean variance of the bootstrap predictions of a model."""
        return np.mean(np.var(self.predict(x), axis=1))

    def square_bias_score(self, x: np.ndarray, f=None):
        """Calculate the mean bias^2 of the bootstrap predictions of a model.

        >>> y_hat = np.array([[1, 2], [3, 4]])
        >>> f_x = np.array([0.5, 0.5])
        >>> get_square_bias(y_hat,  f_x)
        5
        """
        validate_features_dimension(x)
        if f is None:
            f_x = self.estimator.transform(x) @ self.estimator_w_hat
        else:
            if not callable(f):
                raise TypeError("f should be callable")
            f_x = f(x)

        f_hat_mean = np.mean(self.predict(x), axis=1)
        return np.mean((f_hat_mean - f_x.flatten()) ** 2)

    def get_w_hat(self):
        return np.mean(self.w_hats, axis=1)

    def get_w_hat_std(self):
        return np.std(self.w_hats, axis=1)

    def get_w_hat_ci(self, alpha: float = 0.05):
        return np.quantile(self.w_hats, [alpha / 2, 1 - alpha / 2], axis=1)

    def get_w_hat_bias(self):
        return np.mean(self.w_hats, axis=1) - self.estimator_w_hat

    def get_w_hat_bias_std(self):
        return np.std(self.w_hats, axis=1)

    def get_w_hat_bias_ci(self, alpha: float = 0.05):
        return (
            np.quantile(self.w_hats, [alpha / 2, 1 - alpha / 2], axis=1)
            - self.estimator_w_hat
        )

    def get_w_hat_mse(self):
        return np.mean((self.w_hats - self.estimator_w_hat) ** 2, axis=1)

    def get_w_hat_mse_std(self):
        return np.std((self.w_hats - self.estimator_w_hat) ** 2, axis=1)

    def get_w_hat_mse_ci(self, alpha: float = 0.05):
        return np.quantile(
            (self.w_hats - self.estimator_w_hat) ** 2,
            [alpha / 2, 1 - alpha / 2],
            axis=1,
        )
