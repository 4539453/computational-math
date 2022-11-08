# %%
from collections.abc import Iterator
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

cycol: Iterator = cycle("bgrcmykw")


# %% [markdown]
# # Data generation

# %%
def generate_x_y_data(n_dots: int = 25) -> tuple[np.ndarray, np.ndarray]:
    x: np.ndarray = 2 - 3 * np.random.normal(0, 1, n_dots)
    y: np.ndarray = x - 2 * (x**2) + 0.5 * (x**3) + np.random.normal(-4, 4, n_dots)

    # outliers
    # outliers_x = np.array([1, 3, 4])
    # outliers_y = np.array([-30, 20, 40])
    # x = np.concatenate((x, outliers_x))
    # y = np.concatenate((y, outliers_y))

    # transform to vector [rows, columns (new axis from every column)]
    x: np.ndarray = x[:, np.newaxis]
    y: np.ndarray = y[:, np.newaxis]
    return x, y


x, y = generate_x_y_data(20)

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

    plt.plot(data[:, 0], data[:, 1], color=next(cycol), label=label)




# %%
degrees: list[int] = [1, 2, 3, 4, 15, 20]
models_residuals: list[np.ndarray] = []

for degree in degrees:
    # generate X (i poly features for x)
    polynomial_features: PolynomialFeatures = PolynomialFeatures(degree=degree)
    X: np.ndarray = polynomial_features.fit_transform(x)

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
