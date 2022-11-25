import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from machinelearning.utils import validate_data, validate_features_dimension


class LinearRegression(RegressorMixin):
    """Linear regression model/estimator.

    Attributes
    ----------
    _transformer : sklearn.preprocessing._data.PolynomialFeatures

    _w_hat : np.ndarray
    """

    def __init__(self, transformer=None):
        self._set_transformer(transformer)
        self._w_hat = None

    @property
    def w_hat(self):
        if self._w_hat is None:
            raise ValueError("Model is not fitted yet")
        return self._w_hat

    def _set_transformer(self, obj):
        if obj is None:
            obj = PolynomialFeatures(degree=1, include_bias=True)
        elif not isinstance(obj, PolynomialFeatures):
            wrong_transformer_type_message = (
                "\nTransformer should be instance of PolynomialFeatures"
                f"your transformer is {type(obj)}"
            )
            raise TypeError(wrong_transformer_type_message)
        self._transformer = obj

    def get_n_ivs(self) -> int:
        return int(self._transformer.n_output_features_) - 1

    def transform(self, x):
        return self._transformer.fit_transform(x)

    def fit(self, x: np.ndarray, y: np.ndarray):
        validate_data(x, y)
        x = self.transform(x)
        self._w_hat = np.linalg.pinv(x) @ y

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
        n_samples, _ = y.shape
        n_ivs = self.get_n_ivs()
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_ivs - 1)

        return adj_r2

    def mse_score(self, x: np.ndarray, y: np.ndarray):
        validate_data(x, y)
        return np.mean((self.predict(x) - y) ** 2)
