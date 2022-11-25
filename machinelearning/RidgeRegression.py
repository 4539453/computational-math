import numpy as np

from machinelearning.LinearRegression import LinearRegression
from machinelearning.utils import validate_data


class RidgeRegression(LinearRegression):
    def __init__(self, transformer=None, alpha: float = 1.0):
        super().__init__(transformer)
        self.__alpha = alpha

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a number")
        self.__alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """Fit ridge regression model."""
        validate_data(X, y)
        # without intercept
        X = self.transform(X)[:, 1:]
        w_hat = np.linalg.inv(X.T @ X + self.alpha * np.eye(X.shape[1])) @ X.T @ y
        intercept = np.mean(y)
        self._w_hat = np.insert(w_hat, 0, intercept, axis=0)
        self._set_n_ivs(X)
        return self
