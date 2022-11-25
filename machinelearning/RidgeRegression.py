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

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        validate_data(x, y)
        # data without intercept
        X = self.transform(x)[:, 1:]
        _, n_ivs = X.shape
        w_hat = np.linalg.inv(X.T @ X + self.alpha * np.eye(n_ivs)) @ X.T @ y
        intercept = np.mean(y)
        self._w_hat = np.insert(w_hat, 0, intercept, axis=0)
        return self


class SVDRidgeRegression(RidgeRegression):
    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        validate_data(x, y)
        # data without intercept
        X = self.transform(x)[:, 1:]
        # `V` is V.T in the SVD decomposition
        U, s, V = np.linalg.svd(X, full_matrices=False)
        _, n_ivs = X.shape
        w_hat = V.T @ np.linalg.inv(np.diag(s**2) + self.alpha * np.eye(n_ivs)) @ np.diag(s) @ U.T @ y
        intercept = np.mean(y)
        self._w_hat = np.insert(w_hat, 0, intercept, axis=0)
        return self

    def fit_predict(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        validate_data(x, y)
        # data without intercept
        X = self.transform(x)[:, 1:]
        U, s, V = np.linalg.svd(X, full_matrices=False)
        _, n_ivs = X.shape
        y_hat = U @ np.diag(s) @ np.linalg.inv(np.diag(s**2) + self.alpha * np.eye(n_ivs)) @ np.diag(s) @ U.T @ y
        y_hat = y_hat + np.mean(y)
        return y_hat
