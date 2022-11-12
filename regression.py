# %%
from collections.abc import Callable, Iterator
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

sns.set_theme(context="notebook", style="ticks")
np.random.seed(42)

colors_cycle: Iterator = cycle("bgrcmykw")


# %% [markdown]
# # Data generation

# %%
def generate_x_y_data(f: Callable[[np.ndarray], np.ndarray], n_dots: int = 25):
    x: np.ndarray = 2 - 3 * np.random.normal(0, 1, n_dots)
    y: np.ndarray = f(x) + np.random.normal(-3, 3, n_dots)

    # outliers
    # outliers_x = np.array([1, 3, 4])
    # outliers_y = np.array([-30, 20, 40])
    # x = np.concatenate((x, outliers_x))
    # y = np.concatenate((y, outliers_y))

    # return np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1)
    return (x[:, np.newaxis], y[:, np.newaxis])


f = lambda x: x - 2 * (x**2) + 0.5 * (x**3)
x, y = generate_x_y_data(f=f, n_dots=20)

# %%
plt.scatter(x, y, s=15, c="black", marker="o", label="data")
plt.legend(loc="upper left", prop={"size": 10})
plt.title("Generated data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# %% [markdown]
# # Polynomial regression

# %% [markdown]
# ## Pseudocode
#
# ```
# for i in range polynom_degree:
#     generate X (i poly features for x)
#     check condition number
#     calculate:
#           X^*
#           w = X^* * y
#           y_pred = X * w
#
#     calculate metrics:
#         adjusted coefficient of determination
#
#         predictions variance
#         bias^2
#
#
#     plot:
#         (x, y_pred), (x, y)
#
# plot/print metrics
# ```

# %% [markdown]
# ## Model

# %%
def plot(x: np.ndarray, y: np.ndarray, label: str) -> None:
    # sort the values of x before line plot
    data: np.ndarray = np.concatenate([x, y], axis=1)
    data = data[np.argsort(data[:, 0], axis=0), :]

    plt.plot(data[:, 0], data[:, 1], color=next(colors_cycle), label=label)


# %%
degrees: list[int] = [1, 2, 3, 4, 15, 20]
models_residuals: list[np.ndarray] = []

for degree in degrees:
    # generate X (i poly features for x)
    polynomial_features: PolynomialFeatures = PolynomialFeatures(degree=degree)
    X: np.ndarray = polynomial_features.fit_transform(data[:, 0])

    # train-test split

    # check condition number
    condition_number = np.linalg.cond(X)

    # X^*
    try:
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
        X_pseudo_inverse: np.ndarray = np.linalg.pinv(X)
    except LinAlgError as e:
        print(f"SVD computation does not converge for {str(X)}")
        raise e

    # w_hat = X^* * y
    w_hat: np.ndarray = X_pseudo_inverse @ y
    # y_hat = X * w
    y_hat: np.ndarray = X @ w_hat

    # metrics
    # adjusted coefficient of determination
    r2: np.float64 = np.float64(r2_score(y, y_hat))
    adj_r2: np.float64 = 1 - (1 - r2) * (y_hat.shape[0] - 1) / (
        y_hat.shape[0] - degree - 1
    )

    # predictions variance
    var_y_hat: np.float64 = np.var(y_hat)

    # bias^2 = SSE - var(y_hat) - var(y)
    SSE: np.float64 = np.mean((y - y_hat) ** 2)
    sqared_bias: np.float64 = SSE - var_y_hat - np.var(y)

    print(
        f" \
        polynomial degree: {degree},\n \
        condition_number:  {condition_number:.2f},\n \
        var_y_hat:         {var_y_hat:.2f},\n \
        sqared_bias:       {sqared_bias:.2f},\n \
        SSE:               {SSE:.2f},\n \
        R2:                {r2:.2f},\n \
        adj. R2:           {adj_r2:.8f}\n \
        w_hat: {np.squeeze(w_hat).round(2)}\n"
    )

    # plot
    plot(x, y_hat, f"poly degree: {degree}")

    # metadata
    models_residuals.append(np.squeeze(y - y_hat))

plt.scatter(x, y, s=15, c="black", marker="o", label="data")
plt.legend(loc="upper left", prop={"size": 10})
plt.title("Fitted polynomial regressions")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %% [markdown]
# ## QQPlot of residuals

# %%
for i, degree in enumerate(degrees):
    fig = sm.qqplot(
        models_residuals[i], line="45", fit=True, label=f"poly degree: {degree}"
    )
    plt.legend(loc="upper left", prop={"size": 10})
    plt.show(fig)

# %%

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample


class LinearRegression(RegressorMixin):
    def __init__(self, transformer=None):
        self.transformer = transformer

    @property
    def transformer(self):
        return self.transformer

    @transformer.setter
    def transformer(self, transformer):
        if not isinstance(transformer, PolynomialFeatures):
            wrong_transformer_type_message = (
                "\nTransformer should be instance of PolynomialFeatures"
                "your transformer is {}".format(type(transformer))
            )
            raise TypeError(wrong_transformer_type_message)
        self.transformer = transformer

    def transform(self, X):
        if self.transformer is None:
            return X
        return self.transformer.transform(X)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self.transform(X)
        self.w_hat = np.linalg.pinv(X) @ y

        return self

    def predict(self, X: np.ndarray):
        X = self.transform(X)
        return X @ self.w_hat

    def adj_r2_score(self, X: np.ndarray, y: np.ndarray):
        r2 = self.score(X, y)
        adj_r2 = 1 - (1 - r2) * (y.shape[0] - 1) / (y.shape[0] - X.shape[1] - 2)

        return adj_r2


class Bootstrap():
    def __init__(self, estimator, n_bootstraps: int = 1000):
        self.estimator = estimator
        self.n_bootstraps = n_bootstraps

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

        self.w_hats = np.zeros((self.X.shape[1], self.n_bootstraps))

        for i in range(self.n_bootstraps):
            # bootstrap sample
            X_boot, y_boot = resample(self.X, self.y)

            # fit model
            self.estimator.fit(X_boot, y_boot)

            # save w_hat
            self.w_hats[:, i] = self.estimator.w_hat.flatten()

        return self

    def predict(self, X: np.ndarray):
        return X @ self.w_hats

    def score(self, X: np.ndarray, y: np.ndarray):
        y_tiled = np.tile(y, (self.n_bootstraps))
        return r2_score(y_tiled, self.predict(X), multioutput="raw_values")

    def mse_score(self, X: np.ndarray, y: np.ndarray):
        return np.mean((self.predict(X) - y) ** 2)

    def y_hat_variance_score(self, X: np.ndarray):
        """Calculate the mean variance of the bootstrap predictions of a model.
        """
        return np.mean(np.var(self.predict(X), axis=1))

    def get_w_hats(self):
        return self.w_hats

    def get_w_hat(self):
        return np.mean(self.w_hats, axis=1)

    def get_w_hat_std(self):
        return np.std(self.w_hats, axis=1)

    def get_w_hat_ci(self, alpha: float = 0.05):
        return np.quantile(self.w_hats, [alpha / 2, 1 - alpha / 2], axis=1)

    def get_w_hat_bias(self):
        return np.mean(self.w_hats, axis=1) - self.estimator.w_hat

    def get_w_hat_bias_std(self):
        return np.std(self.w_hats, axis=1)

    def get_w_hat_bias_ci(self, alpha: float = 0.05):
        return (
            np.quantile(self.w_hats, [alpha / 2, 1 - alpha / 2], axis=1)
            - self.estimator.w_hat
        )

    def get_w_hat_mse(self):
        return np.mean((self.w_hats - self.estimator.w_hat) ** 2, axis=1)

    def get_w_hat_mse_std(self):
        return np.std((self.w_hats - self.estimator.w_hat) ** 2, axis=1)

    def get_w_hat_mse_ci(self, alpha: float = 0.05):
        return np.quantile(
            (self.w_hats - self.estimator.w_hat) ** 2,
            [alpha / 2, 1 - alpha / 2],
            axis=1,
        )







# def bootstrap_pipeline(degree: int, n_bootstraps: int = 1000):
#     return Pipeline(
#         [
#             ("poly", PolynomialFeatures(degree=degree)),
#             ("bootstrap", Bootstrap(LinearRegression(), n_bootstraps)),
#         ]
#     )

# %%

from typing import Type

from sklearn.model_selection import KFold


# def _check_pipline_contain(estimator: Type, position: int, pipeline: Pipeline):
#     if not isinstance(pipeline[position], estimator):
#         wrong_estimator_type_message = (
#             "\n Your pipeline should contain an "
#             f"estimator of type {estimator} in position {position}."
#         )
#         raise TypeError(wrong_estimator_type_message)


# https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b611bf873bd5836748647221480071a87/sklearn/model_selection/_validation.py#L49
def kfold_cross_validation(
    pipe: Pipeline, x: np.ndarray, y: np.ndarray, f: Callable, n_splits=10
) -> dict:
    """Simple implementation of KFold CV directly for Bootstrap class."""

    def get_mse(y_hat, y):
        return np.mean((y_hat - y) ** 2)

    
    def get_square_bias(y_hat: np.ndarray, f_x: np.ndarray):
        """Calculate the mean bias^2 of the bootstrap predictions of a model.

        >>> y_hat = np.array([[1, 2], [3, 4]])
        >>> f_x = np.array([0.5, 0.5])
        >>> get_square_bias(y_hat,  f_x)
        5
        """
        return np.mean((np.mean(y_hat, axis=1) - f_x) ** 2)

    _check_pipline_contain(Bootstrap, -1, pipe)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = {
        "train_r2": np.zeros((1, n_splits)),
        "train_adj_r2": np.zeros((1, n_splits)),
        "train_mse": np.zeros((1, n_splits)),
        "train_bias_y_hat": np.zeros((1, n_splits)),
        "train_var_y_hat": np.zeros((1, n_splits)),
        "test_r2": np.zeros((1, n_splits)),
        "test_adj_r2": np.zeros((1, n_splits)),
        "test_mse": np.zeros((1, n_splits)),
        "test_bias_y_hat": np.zeros((1, n_splits)),
        "test_var_y_hat": np.zeros((1, n_splits)),
    }

    for i, idx in enumerate(kf.split(x)):
        x_train, x_test = x[idx[0]], x[idx[1]]
        y_train, y_test = y[idx[0]], y[idx[1]]

        pipe.fit(x_train, y_train)

        y_hat_train: np.ndarray = pipe.predict(x_train)
        y_hat_test: np.ndarray = pipe[-1].estimator.predict(x_train)

        scores["train_r2"][:, i] = pipe[-1].estimator.score(x_train, y_train)
        scores["train_adj_r2"][:, i] = pipe[-1].estimator.adj_r2_score(x_train, y_train)
        scores["train_sse"][:, i] = get_mse(y_hat_train, y_train)
        scores["train_bias_y_hat"][:, i] = get_square_bias(y_hat_train, f(x_train))
        scores["train_var_y_hat"][:, i] = get_variance(y_hat_train)
        scores["test_r2"][:, i] = pipe[-1].estimator.score(x_test, y_test)
        scores["test_adj_r2"][:, i] = pipe[-1].estimator.adj_r2_score(x_test, y_test)
        scores["test_sse"][:, i] = get_mse(y_hat_test, y_test)
        scores["test_bias_y_hat"][:, i] = get_square_bias(y_hat_test, f(x_test))
        scores["test_var_y_hat"][:, i] = get_variance(y_hat_test)

    return {k: np.mean(v, axis=1) for k, v in scores.items()}


pipe = Pipeline(
    steps=[
        ("polynomial_features", PolynomialFeatures()),
        ("bootstrap", Bootstrap(estimator=LinearRegression())),
    ]
)

# %%
pipe.set_params(polynomial_features__degree=3)

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# pipe.fit(X_train, y_train)
kfold_cross_validation(pipe, x, y, f, n_splits=10)

# print(f"R2: {np.mean(pipe.score(X_test, y_test)):8f}")
# print(f"w_hat: {pipe['bootstrap'].get_w_hats()}")
# print(f"w_hat: {pipe.get_params()}")
