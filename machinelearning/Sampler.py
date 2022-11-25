from copy import copy

import numpy as np

from machinelearning.LinearRegression import LinearRegression
from machinelearning.utils import validate_data


class Sampler:
    def __init__(self, type=None, n_samples=None, estimator=None, f=None):
        self.__set_type(type)

        if type == "pair":
            self.__set_n_samples(n_samples)
        elif type == "residuals":
            self.__set_estimator(estimator)
        elif type == "cheating":
            self.__set_f(f)

        self.__x = None
        self.__y = None

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

    def __set_type(self, value):
        allowed_types_str = ("pair", "residuals", "cheating")
        if isinstance(value, str):
            if value not in allowed_types_str:
                raise ValueError(f"Sampler type {value} is not allowed")
        else:
            raise TypeError("\nType should be str")
        self.__type = value

    def __set_estimator(self, obj) -> None:
        if not isinstance(obj, LinearRegression):
            wrong_estimator_type_message = (
                "\nEstimator should be instance of LinearRegression"
                f"your estimator is {type(obj)}\n\n"
            )
            raise TypeError(wrong_estimator_type_message)
        self.__estimator = obj

    def __set_n_samples(self, value) -> None:
        if value is None:
            pass
        elif not isinstance(value, int):
            wrong_n_samples_type_message = (
                "\nN_samples should be integer" f"your n_samples is {type(value)}\n\n"
            )
            raise TypeError(wrong_n_samples_type_message)
        self.__n_samples = value

    def __set_f(self, value) -> None:
        if not callable(value):
            wrong_f_type_message = (
                "\nF should be callable" f"your f is {type(value)}\n\n"
            )
            raise TypeError(wrong_f_type_message)
        self.__f = value

    @property
    def x(self):
        if self.__x is None:
            raise ValueError("Sampler is not fitted")
        return self.__x

    @property
    def y(self):
        if self.__y is None:
            raise ValueError("Sampler is not fitted")
        return self.__y

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        validate_data(x, y)
        self.__x = x
        self.__y = y

        if self.__type == "residuals":
            estimator = copy(self.__estimator)
            self.__y_hat = estimator.fit(x, y).predict(x)
            self.__residuals = y - self.__y_hat

        if self.__type == "cheating":
            self.__y_hat = self.__f(x)
            self.__residuals = y - self.__y_hat

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        if self.__type == "pair":
            idx = self.get_sample_idx()
            return self.x[idx], self.y[idx]

        elif self.__type in ["residuals", "cheating"]:
            np.random.shuffle(self.__residuals)
            return self.x, self.__y_hat + self.__residuals

        else:
            raise ValueError("Wrong sampler type")

    def get_sample_idx(self) -> np.ndarray:
        if self.__n_samples is not None:
            idx = np.random.choice(self.y.shape[0], size=self.__n_samples)
        else:
            idx = np.random.choice(self.y.shape[0], size=self.y.shape[0])
        return idx
