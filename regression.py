# %%
from collections.abc import Callable, Iterator
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# sns.set_theme(context="notebook", style="ticks")
# plt.style.use('bmh')
plt.style.use("ggplot")

np.random.seed(42)

# %% [markdown]
# ## Model

# %%

from copy import copy

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample


def _validate_features_dimension(x: np.ndarray):
    if x.ndim != 2:
        raise ValueError("Must be a 2D array.")
    if len(x[0, :]) != 1:
        raise ValueError("Stop! You're not prepared for multi features!")


def _validate_targets_dimension(y: np.ndarray):
    if y.ndim != 2:
        raise ValueError("Must be a 2D array.")
    if len(y[0, :]) != 1:
        raise ValueError("Oh, several targets. Interesting...")


class LinearRegression(RegressorMixin):
    def __init__(self, transformer):
        self.__set_transformer(transformer)
        self.__w_hat = None

    @property
    def w_hat(self):
        if self.__w_hat is None:
            raise ValueError("Model is not fitted yet")
        return self.__w_hat

    @property
    def degree(self):
        return self.__transformer.degree

    def __set_transformer(self, obj):
        if not isinstance(obj, PolynomialFeatures):
            wrong_transformer_type_message = (
                "\nTransformer should be instance of PolynomialFeatures"
                f"your transformer is {type(obj)}"
            )
            raise TypeError(wrong_transformer_type_message)
        self.__transformer = obj

    def transform(self, x):
        return np.asarray(self.__transformer.fit_transform(x))

    def fit(self, x: np.ndarray, y: np.ndarray):
        _validate_features_dimension(x)
        _validate_targets_dimension(y)
        x = self.transform(x)
        self.__w_hat = np.linalg.pinv(x) @ y

        return self

    def predict(self, x: np.ndarray):
        _validate_features_dimension(x)
        x = self.transform(x)
        return x @ self.w_hat

    def score(self, x: np.ndarray, y: np.ndarray):
        _validate_features_dimension(x)
        _validate_targets_dimension(y)
        return r2_score(y, self.predict(x))

    # FIXME: this is not a good way to do it
    # maybe some restrictions exists?
    # n - p - 1 < 0 ... ?
    def adj_r2_score(self, x: np.ndarray, y: np.ndarray):
        _validate_features_dimension(x)
        _validate_targets_dimension(y)
        r2 = self.score(x, y)
        adj_r2 = 1 - (1 - r2) * (y.shape[0] - 1) / (y.shape[0] - self.degree - 1)

        return adj_r2

    def mse_score(self, x: np.ndarray, y: np.ndarray):
        _validate_features_dimension(x)
        _validate_targets_dimension(y)
        return np.mean((self.predict(x) - y) ** 2)


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
        _validate_features_dimension(x)
        _validate_targets_dimension(y)
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


class Bootstrap(RegressorMixin):
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
        _validate_features_dimension(x)
        _validate_targets_dimension(y)
        estimator = copy(self.estimator)
        self.__estimator_w_hat = estimator.fit(x, y).w_hat
        self.__sampler.fit(x, y)
        self.__w_hats = np.zeros((estimator.degree + 1, self.__n_bootstraps))

        for i in range(self.__n_bootstraps):
            # bootstrap sample
            x_boot, y_boot = next(self.__sampler)

            # fit model
            estimator.fit(x_boot, y_boot)

            # save w_hat
            self.w_hats[:, i] = estimator.w_hat.flatten()

        return self

    def predict(self, x: np.ndarray):
        _validate_features_dimension(x)
        return self.estimator.transform(x) @ self.w_hats

    def score(self, x: np.ndarray, y: np.ndarray):
        _validate_features_dimension(x)
        _validate_targets_dimension(y)
        y_tiled = np.tile(y, (self.__n_bootstraps))
        return r2_score(y_tiled, self.predict(x), multioutput="raw_values")

    # TODO: move scores to estimator class
    # differet scores for different estimators
    def mse_score(self, x: np.ndarray, y: np.ndarray):
        _validate_features_dimension(x)
        _validate_targets_dimension(y)
        return np.mean(np.mean((self.predict(x) - y) ** 2, axis=1))

    def y_hat_variance_score(self, x: np.ndarray):
        _validate_features_dimension(x)
        """Calculate the mean variance of the bootstrap predictions of a model."""
        return np.mean(np.var(self.predict(x), axis=1))

    def square_bias_score(self, x: np.ndarray, f=None):
        """Calculate the mean bias^2 of the bootstrap predictions of a model.

        >>> y_hat = np.array([[1, 2], [3, 4]])
        >>> f_x = np.array([0.5, 0.5])
        >>> get_square_bias(y_hat,  f_x)
        5
        """
        _validate_features_dimension(x)
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


# %%

from sklearn.model_selection import KFold


def kfold_cross_validation(
    bs: Bootstrap, x: np.ndarray, y: np.ndarray, f=None, n_splits=10
) -> dict:
    """Simple implementation of KFold CV directly for Bootstrap class."""
    _validate_features_dimension(x)
    _validate_targets_dimension(y)

    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf = KFold(n_splits=n_splits, shuffle=True)

    scores = {
        "train_r2": np.zeros(n_splits),
        "train_adj_r2": np.zeros(n_splits),
        "train_mse": np.zeros(n_splits),
        "train_bs_mse": np.zeros(n_splits),
        "train_bias_y_hat": np.zeros(n_splits),
        "train_var_y_hat": np.zeros(n_splits),
        "train_cond_num": np.zeros(n_splits),
        "test_r2": np.zeros(n_splits),
        "test_adj_r2": np.zeros(n_splits),
        "test_mse": np.zeros(n_splits),
        "test_bs_mse": np.zeros(n_splits),
        "test_bias_y_hat": np.zeros(n_splits),
        "test_var_y_hat": np.zeros(n_splits),
        "test_cond_num": np.zeros(n_splits),
    }

    for i, idx in enumerate(kf.split(x)):
        x_train, x_test = x[idx[0]], x[idx[1]]
        y_train, y_test = y[idx[0]], y[idx[1]]

        bs.fit(x_train, y_train)
        estimator = copy(bs.estimator).fit(x_train, y_train)

        scores["train_r2"][i] = estimator.score(x_train, y_train)
        scores["train_adj_r2"][i] = estimator.adj_r2_score(x_train, y_train)
        scores["train_mse"][i] = estimator.mse_score(x_train, y_train)
        scores["train_bs_mse"][i] = bs.mse_score(x_train, y_train)
        scores["train_bias_y_hat"][i] = bs.square_bias_score(x_train, f)
        scores["train_var_y_hat"][i] = bs.y_hat_variance_score(x_train)
        scores["train_cond_num"][i] = np.linalg.cond(estimator.transform(x_train))

        scores["test_r2"][i] = estimator.score(x_test, y_test)
        scores["test_adj_r2"][i] = estimator.adj_r2_score(x_test, y_test)
        scores["test_mse"][i] = estimator.mse_score(x_test, y_test)
        scores["test_bs_mse"][i] = bs.mse_score(x_test, y_test)
        scores["test_bias_y_hat"][i] = bs.square_bias_score(x_test, f)
        scores["test_var_y_hat"][i] = bs.y_hat_variance_score(x_test)
        scores["test_cond_num"][i] = np.linalg.cond(estimator.transform(x_train))

    return {k: np.mean(v) for k, v in scores.items()}


# %% [markdown]
# # Data generation

# %%
def generate_x_y_data(
    x: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    mean_sd: tuple[float, float],
    outliers=(None, None),
) -> tuple[np.ndarray, np.ndarray]:
    def generate_y(x, f, mean_sd: tuple[float, float]):
        y: np.ndarray = f(x)
        residuals = np.random.normal(*mean_sd, y.shape[0]).reshape((-1, 1))
        return y + residuals

    def add_outliers(x, y, outliers):
        outliers = tuple(
            arr.reshape((-1, 1)) if arr.ndim == 1 else arr for arr in outliers
        )
        x = np.concatenate((x, outliers[0]), axis=0)
        y = np.concatenate((y, outliers[1]), axis=0)
        return x, y

    if x.ndim == 1:
        x = x.reshape((-1, 1))

    _validate_features_dimension(x)

    y = generate_y(x, f, mean_sd)

    if all(outliers) and len(outliers) == 2:
        x, y = add_outliers(x, y, outliers)

    return x, y


# %%
data_params_1 = {
    "x": 2 - 3 * np.random.normal(0, 1, 50),
    "f": lambda x: x - 2 * (x**2) + 0.5 * (x**3),
    "mean_sd": (0, 3),
}

data_params_2 = {
    "x": np.linspace(-10, 10, 70),
    "f": lambda x: 1 + 2 * (x**2),
    "mean_sd": (0, 3),
}

x, y = generate_x_y_data(**data_params_1)
f = data_params_1["f"]

# degrees = [1, 2, 10]
# degrees = range(1, 11)
degrees = range(2, 7)
pf = PolynomialFeatures()
lr = LinearRegression(transformer=pf)
bs_sampler = Sampler(type="cheating", f=f)
bs = Bootstrap(lr, bs_sampler, n_bootstraps=100)
n_cv_splits = 4


cv_scores: dict[str, np.ndarray] = {}
y_hats: list[np.ndarray] = []
for i, degree in enumerate(degrees):
    # set degree of polynomial features
    pf.degree = degree
    # calculate cv scores
    score = kfold_cross_validation(bs, x, y, f, n_cv_splits)
    # estimate w_hat on all data
    y_hats.append(lr.fit(x, y).predict(x).flatten())
    # fill cv_scores dict
    for k, v in score.items():
        if k not in cv_scores.keys():
            cv_scores[k] = np.zeros(len(degrees))
        cv_scores[k][i] = v

# print(pd.DataFrame(cv_scores.values(), index=list(cv_scores.keys()), columns=degrees))
pd.DataFrame(cv_scores, index=degrees)

# %%
# plt.scatter(x, y, s=15, c="black", marker="o", label="data")
# plt.legend(loc="upper left", prop={"size": 10})
# plt.title("Generated data")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()


# %%
def plot(cv_scores: dict[str, np.ndarray], degrees: range, x, y, y_hats):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # plot (x, y) data
    ax1.scatter(x, y, s=15, c="black", marker="o", label="data")

    def plot_y_hats(ax, y_hats, degrees):
        """Plot fitted polynomials."""
        for i, degree in enumerate(degrees):
            data: np.ndarray = np.concatenate([x, y_hats[i].reshape((-1, 1))], axis=1)
            data = data[np.argsort(data[:, 0], axis=0), :]
            ax.plot(data[:, 0], data[:, 1], label=f"degree {degree}")

        ax.legend(loc="upper left", prop={"size": 10})
        ax.set_title("Fitted polynomial regressions")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plot_y_hats(ax1, y_hats, degrees)

    def plot_bias_variance_tradeoff(ax, cv_scores, degrees):
        labels = ["bias_y_hat", "var_y_hat", "bs_mse"]
        labels = [f"test_{lb}" for lb in labels]
        scores = np.array([cv_scores[lb] for lb in labels])
        normalized_scores = scores / scores.max()
        for i, lb in enumerate(labels):
            ax.plot(degrees, normalized_scores[i], "-o", label=lb)

        ax.legend(loc="upper right", prop={"size": 10})
        ax.set_title("Bias-variance tradeoff")
        ax.set_xlabel("degrees")
        ax.set_ylabel("percents(%)")

    plot_bias_variance_tradeoff(ax2, cv_scores, degrees)

    plt.show(fig)


plot(cv_scores, degrees, x, y, y_hats)


# %%
# def plot(x: np.ndarray, y: np.ndarray, label: str) -> None:
#     # sort the values of x before line plot
#     data: np.ndarray = np.concatenate([x, y], axis=1)
#     data = data[np.argsort(data[:, 0], axis=0), :]

#     plt.plot(data[:, 0], data[:, 1], color=next(colors_cycle), label=label)


# %%
# degrees: list[int] = [1, 2, 3, 4, 15, 20]
# models_residuals: list[np.ndarray] = []

# for degree in degrees:
#     # generate X (i poly features for x)
#     polynomial_features: PolynomialFeatures = PolynomialFeatures(degree=degree)
#     X: np.ndarray = polynomial_features.fit_transform(data[:, 0])

#     # train-test split

#     # check condition number
#     condition_number = np.linalg.cond(X)

#     # X^*
#     try:
#         # https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
#         X_pseudo_inverse: np.ndarray = np.linalg.pinv(X)
#     except LinAlgError as e:
#         print(f"SVD computation does not converge for {str(X)}")
#         raise e

#     # w_hat = X^* * y
#     w_hat: np.ndarray = X_pseudo_inverse @ y
#     # y_hat = X * w
#     y_hat: np.ndarray = X @ w_hat

#     # metrics
#     # adjusted coefficient of determination
#     r2: np.float64 = np.float64(r2_score(y, y_hat))
#     adj_r2: np.float64 = 1 - (1 - r2) * (y_hat.shape[0] - 1) / (
#         y_hat.shape[0] - degree - 1
#     )

#     # predictions variance
#     var_y_hat: np.float64 = np.var(y_hat)

#     # bias^2 = SSE - var(y_hat) - var(y)
#     SSE: np.float64 = np.mean((y - y_hat) ** 2)
#     sqared_bias: np.float64 = SSE - var_y_hat - np.var(y)

#     print(
#         f" \
#         polynomial degree: {degree},\n \
#         condition_number:  {condition_number:.2f},\n \
#         var_y_hat:         {var_y_hat:.2f},\n \
#         sqared_bias:       {sqared_bias:.2f},\n \
#         SSE:               {SSE:.2f},\n \
#         R2:                {r2:.2f},\n \
#         adj. R2:           {adj_r2:.8f}\n \
#         w_hat: {np.squeeze(w_hat).round(2)}\n"
#     )

#     # plot
#     plot(x, y_hat, f"poly degree: {degree}")

#     # metadata
#     models_residuals.append(np.squeeze(y - y_hat))

# plt.scatter(x, y, s=15, c="black", marker="o", label="data")
# plt.legend(loc="upper left", prop={"size": 10})
# plt.title("Fitted polynomial regressions")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# %% [markdown]
# ## QQPlot of residuals

# %%
# for i, degree in enumerate(degrees):
#     fig = sm.qqplot(
#         models_residuals[i], line="45", fit=True, label=f"poly degree: {degree}"
#     )
#     plt.legend(loc="upper left", prop={"size": 10})
#     plt.show(fig)
