# %%
from collections.abc import Callable, Iterator
from copy import copy
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from scipy.stats import alpha
import seaborn as sns
import statsmodels.api as sm
from pandas.plotting import PlotAccessor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample

from machinelearning.Bootstrap import Bootstrap
from machinelearning.CrossValidation import kfold_cross_validation
from machinelearning.Data import generate_functional_data
from machinelearning.LassoRegression import LassoRegression
from machinelearning.LinearRegression import LinearRegression
from machinelearning.RidgeRegression import RidgeRegression, SVDRidgeRegression
from machinelearning.Sampler import Sampler
from machinelearning.utils import validate_data, validate_features_dimension

# sns.set_theme(context="notebook", style="ticks")
# plt.style.use('bmh')
plt.style.use("ggplot")

np.random.seed(42)


# %% [markdown]
# # Data generation

# %%
data_params_1 = {
    "x": 2 - 3 * np.random.normal(0, 1, 40),
    "f": lambda x: x - 2 * (x**2) + 0.5 * (x**3),
    "mean_sd": (0, 15),
}

data_params_2 = {
    "x": np.linspace(-10, 10, 70),
    "f": lambda x: 1 + 2 * (x**2),
    "mean_sd": (0, 3),
}

x, y = generate_functional_data(**data_params_1)
f = data_params_1["f"]
w_true = np.array([0.0, 1.0, -2.0, 0.5]).reshape((-1, 1))


# %% [markdown]
# ## Model


# %%
def plot(
    cv_scores: dict[str, np.ndarray],
    degrees: range,
    x,
    y,
    y_hats,
    title: str = "Regression",
):
    gridsize = (3, 2)
    fig = plt.figure(figsize=(12, 12))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid(gridsize, (2, 0))
    ax3 = plt.subplot2grid(gridsize, (2, 1))
    # fig, ((ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 12))
    # plot (x, y) data
    ax1.scatter(x, y, s=15, c="black", marker="o", label="data")

    def plot_y_hats(ax, y_hats, degrees):
        """Plot fitted polynomials."""
        for i, degree in enumerate(degrees):
            data: np.ndarray = np.concatenate([x, y_hats[i].reshape((-1, 1))], axis=1)
            data = data[np.argsort(data[:, 0], axis=0), :]
            ax.plot(data[:, 0], data[:, 1], label=f"degree {degree}")

        ax.legend(loc="upper left", prop={"size": 10})
        ax.set_title("Fitted data")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plot_y_hats(ax1, y_hats, degrees)

    def plot_f_hat_bias_variance_tradeoff(ax, cv_scores, degrees):
        labels = ["bias_y_hat", "var_y_hat", "bs_mse"]
        labels = [f"test_{lb}" for lb in labels]
        scores = np.array([cv_scores[lb] for lb in labels])
        normalized_scores = scores / scores.max()
        for i, lb in enumerate(labels):
            ax.plot(degrees, normalized_scores[i], "-o", label=lb)

        ax.legend(loc="upper right", prop={"size": 10})
        ax.set_title("f_hat bias-variance tradeoff")
        ax.set_xlabel("degrees")
        ax.set_ylabel("percents(%)")

    plot_f_hat_bias_variance_tradeoff(ax2, cv_scores, degrees)

    def plot_w_hat_bias_variance_tradeoff(ax, cv_scores, degrees):
        labels = ["w_hat_bias", "w_hat_var"]
        labels = [f"test_bs_{lb}" for lb in labels]
        scores = np.array([cv_scores[lb] for lb in labels])
        normalized_scores = scores / scores.max()
        for i, lb in enumerate(labels):
            ax.plot(degrees, normalized_scores[i], "-o", label=lb)

        ax.legend(loc="upper right", prop={"size": 10})
        ax.set_title("w_hat bias-variance tradeoff")
        ax.set_xlabel("degrees")
        ax.set_ylabel("percents(%)")

    plot_w_hat_bias_variance_tradeoff(ax3, cv_scores, degrees)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show(fig)


# %%
# degrees = [1, 2, 10]
# degrees = range(1, 11)
degrees = range(2, 7)
pf = PolynomialFeatures()
estimator = LinearRegression(transformer=pf)
bs_sampler = Sampler(type="cheating", f=f)
bs = Bootstrap(estimator, bs_sampler, n_bootstraps=100)
n_cv_splits = 4


cv_scores: dict[str, np.ndarray] = {}
y_hats: list[np.ndarray] = []
for i, degree in enumerate(degrees):
    # set degree of polynomial features
    pf.degree = degree
    # calculate cv scores
    score = kfold_cross_validation(bs, x, y, f, w_true, n_cv_splits)
    # estimate w_hat on all data
    y_hats.append(estimator.fit(x, y).predict(x).flatten())
    # fill cv_scores dict
    for k, v in score.items():
        if k not in cv_scores.keys():
            cv_scores[k] = np.zeros(len(degrees))
        cv_scores[k][i] = v

# print(pd.DataFrame(cv_scores.values(), index=list(cv_scores.keys()), columns=degrees))
pd.DataFrame(cv_scores, index=degrees)

# %%
plot(cv_scores, degrees, x, y, y_hats, title="Polynomial regression")

# %%
# degrees = [1, 2, 10]
# degrees = range(1, 11)
degrees = range(2, 7)
pf = PolynomialFeatures()
estimator = RidgeRegression(transformer=pf, alpha=10e3)
bs_sampler = Sampler(type="cheating", f=f)
bs = Bootstrap(estimator, bs_sampler, n_bootstraps=100)
n_cv_splits = 4


cv_scores: dict[str, np.ndarray] = {}
y_hats: list[np.ndarray] = []
for i, degree in enumerate(degrees):
    # set degree of polynomial features
    pf.degree = degree
    # calculate cv scores
    score = kfold_cross_validation(bs, x, y, f, w_true, n_cv_splits)
    # estimate w_hat on all data
    y_hats.append(estimator.fit(x, y).predict(x).flatten())
    # fill cv_scores dict
    for k, v in score.items():
        if k not in cv_scores.keys():
            cv_scores[k] = np.zeros(len(degrees))
        cv_scores[k][i] = v

# print(pd.DataFrame(cv_scores.values(), index=list(cv_scores.keys()), columns=degrees))
pd.DataFrame(cv_scores, index=degrees)

# %%
plot(cv_scores, degrees, x, y, y_hats, title="Polynomial ridge regression")

# %%
# degrees = [1, 2, 10]
# degrees = range(1, 11)
degrees = range(2, 7)
pf = PolynomialFeatures()
estimator = SVDRidgeRegression(transformer=pf, alpha=10e3)
bs_sampler = Sampler(type="cheating", f=f)
bs = Bootstrap(estimator, bs_sampler, n_bootstraps=100)
n_cv_splits = 4


cv_scores: dict[str, np.ndarray] = {}
y_hats: list[np.ndarray] = []
for i, degree in enumerate(degrees):
    # set degree of polynomial features
    pf.degree = degree
    # calculate cv scores
    score = kfold_cross_validation(bs, x, y, f, w_true, n_cv_splits)
    # estimate w_hat on all data
    y_hats.append(estimator.fit_predict(x, y).flatten())
    # fill cv_scores dict
    for k, v in score.items():
        if k not in cv_scores.keys():
            cv_scores[k] = np.zeros(len(degrees))
        cv_scores[k][i] = v

# print(pd.DataFrame(cv_scores.values(), index=list(cv_scores.keys()), columns=degrees))
pd.DataFrame(cv_scores, index=degrees)

# %%
plot(cv_scores, degrees, x, y, y_hats, title="SVD Ridge regression")


# %%
# degrees = [1, 2, 10]
# degrees = range(1, 11)
degrees = range(2, 7)
pf = PolynomialFeatures()
estimator = LassoRegression(transformer=pf, alpha=10e3)
bs_sampler = Sampler(type="cheating", f=f)
bs = Bootstrap(estimator, bs_sampler, n_bootstraps=100)
n_cv_splits = 4


cv_scores: dict[str, np.ndarray] = {}
y_hats: list[np.ndarray] = []
for i, degree in enumerate(degrees):
    # set degree of polynomial features
    pf.degree = degree
    # calculate cv scores
    score = kfold_cross_validation(bs, x, y, f, w_true, n_cv_splits)
    # estimate w_hat on all data
    y_hats.append(estimator.fit(x, y).predict(x).flatten())
    # fill cv_scores dict
    for k, v in score.items():
        if k not in cv_scores.keys():
            cv_scores[k] = np.zeros(len(degrees))
        cv_scores[k][i] = v

# print(pd.DataFrame(cv_scores.values(), index=list(cv_scores.keys()), columns=degrees))
pd.DataFrame(cv_scores, index=degrees)

# %%
plot(cv_scores, degrees, x, y, y_hats, title="Polynomial lasso regression")
