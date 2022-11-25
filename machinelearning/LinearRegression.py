import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from machinelearning.utils import validate_data, validate_features_dimension


class LinearRegression(RegressorMixin):
    """Linear regression model/estimator.

    Attributes
    ----------
    __transformer : sklearn.preprocessing._data.PolynomialFeatures

    __w_hat : np.ndarray

    __n_ivs : int
        Number of independent variables.
    """

    def __init__(self, transformer=None):
        self._set_transformer(transformer)
        self._w_hat = None
        self._n_ivs = None

    @property
    def w_hat(self):
        if self._w_hat is None:
            raise ValueError("Model is not fitted yet")
        return self._w_hat

    @property
    def n_ivs(self):
        if self._n_ivs is None:
            raise ValueError("Model is not fitted yet")
        return self._n_ivs

    def _set_n_ivs(self, x: np.ndarray):
        if self._transformer is None:
            self._n_ivs = int(x.shape[1])
        else:
            # FIXME: do we count the intercept?
            self._n_ivs = int(self._transformer.n_output_features_) - 1
            # assert self.__n_ivs == self.__transformer.degree

    def _set_transformer(self, obj):
        if obj is None:
            pass
        if not isinstance(obj, PolynomialFeatures):
            wrong_transformer_type_message = (
                "\nTransformer should be instance of PolynomialFeatures"
                f"your transformer is {type(obj)}"
            )
            raise TypeError(wrong_transformer_type_message)
        self._transformer = obj

    def transform(self, x):
        if self._transformer is None:
            pass
        else:
            x = self._transformer.fit_transform(x)
        return x

    def fit(self, x: np.ndarray, y: np.ndarray):
        validate_data(x, y)
        x = self.transform(x)
        self._w_hat = np.linalg.pinv(x) @ y

        # FIXME: maybe not good idea put it here
        self._set_n_ivs(x)
        return self

    def predict(self, x: np.ndarray):
        validate_features_dimension(x)
        x = self.transform(x)
        return x @ self.w_hat

    def score(self, x: np.ndarray, y: np.ndarray):
        validate_data(x, y)
        return r2_score(y, self.predict(x))

    # FIXME: this is not a good way to do it
    # maybe some restrictions exists?
    # n - p - 1 < 0 ... ?
    def adj_r2_score(self, x: np.ndarray, y: np.ndarray):
        validate_data(x, y)
        r2 = self.score(x, y)
        adj_r2 = 1 - (1 - r2) * (y.shape[0] - 1) / (y.shape[0] - self.n_ivs - 1)

        return adj_r2

    def mse_score(self, x: np.ndarray, y: np.ndarray):
        validate_data(x, y)
        return np.mean((self.predict(x) - y) ** 2)
