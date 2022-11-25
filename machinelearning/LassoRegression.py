import numpy as np
from sklearn.linear_model import Lasso

from machinelearning.LinearRegression import LinearRegression
from machinelearning.utils import validate_data


class LassoRegression(LinearRegression):
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

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LassoRegression":
        validate_data(x, y)
        X = self.transform(x)
        self._w_hat = Lasso(alpha=self.alpha).fit(X, y).coef_.reshape((-1, 1))

        return self
