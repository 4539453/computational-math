# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/4539453/computational-math/blob/main/sphere_flight.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="utedRmTaGFlJ"
# # Simulating Sphere Flight

# %% [markdown] id="AvQia9jkw8ud"
# ## System

from copy import copy
# %% id="ECL7vbRZ1b8m"
from types import SimpleNamespace

import pandas as pd


class SettableNamespace(SimpleNamespace):
    """Collection of parameters"""

    def __init__(self, namespace=None, **kwargs):
        super().__init__()
        if namespace:
            self.__dict__.update(namespace.__dict__)
        self.__dict__.update(kwargs)

    def get(self, name, default=None):
        try:
            return self.__getattribute__(name, default)
        except AttributeError:
            return default

    def set(self, **variables):
        # new = copy(self)
        # new.__dict__.update(variables)
        # return new
        return self.__dict__.update(variables)


class System(SettableNamespace):
    pass


class Params(SettableNamespace):
    pass


def State(**variables):
    """Values of state variables"""
    return pd.Series(variables, name="state")


def Vector(x, y):
    return pd.Series(dict(x=x, y=y))


def show(obj):
    """Display a Series or Namespace as a DataFrame"""
    if isinstance(obj, pd.Series):
        df = pd.DataFrame(obj)
        return df
    elif hasattr(obj, "__dict__"):
        return pd.DataFrame(pd.Series(obj.__dict__), columns=["value"])
    else:
        return obj


# %% [markdown] id="1SZXh8bbxRg3"
# ## Plot

# %% id="hddnB4__w5jt"
import matplotlib.pyplot as plt


def decorate(**options):
    """decorate(title='Title',
                xlabel='x',
                ylabel='y')
    https://matplotlib.org/api/axes_api.html
    """
    ax = plt.gca()
    ax.set(**options)
    ax.axis("equal")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    plt.tight_layout()


def make_series(x, y):
    if isinstance(y, pd.Series):
        y = y.values
    series = pd.Series(y, index=x)
    # series.index.name = 'index'
    return series


# %% [markdown] id="0ShJtUX-xEVA"
# ## IVP wrapper

# %% id="aUGWGrGlveiE"
from numpy import linspace
from scipy.integrate import solve_ivp


def TimeFrame(*args, **kwargs):
    """Maps from time to State"""
    return pd.DataFrame(*args, **kwargs)


def run_solve_ivp(system, slope_func, **options):

    t_0 = getattr(system, "t_0", 0)

    events = options.get("events", [])

    # if there's one event, put it in a list
    try:
        iter(events)
    except TypeError:
        events = [events]

    for event in events:
        # make events terminal if not specified
        if not hasattr(event, "terminal"):
            event.terminal = True

    # run the solver
    bunch = solve_ivp(
        slope_func, [t_0, system.t_end], system.init, args=[system], **options
    )

    # separate the results from the details
    y = bunch.pop("y")
    t = bunch.pop("t")

    columns = system.init.index

    # results at equally-spaced points
    if options.get("dense_output"):
        try:
            num = system.num
        except AttributeError:
            num = 69
        t_final = t[-1]
        t_array = linspace(t_0, t_final, num)
        y_array = bunch.sol(t_array)
        results = TimeFrame(y_array.T, index=t_array, columns=columns)
    else:
        results = TimeFrame(y.T, index=t, columns=columns)

    return results, bunch


# %% [markdown] id="CsE_FYa5xdb9"
# ## Some math

# %% id="XaA_cBhnPhIp"
import numpy as np
from numpy import pi


def pol2cart(alpha: float, radius: float) -> Vector:
    x = radius * np.cos(alpha)
    y = radius * np.sin(alpha)
    return Vector(x, y)


def sphere_mass(diameter: float, rho: float) -> float:
    volume = 4 / 3 * pi * (diameter / 2) ** 3
    return volume * rho


def vector_norm(v: Vector) -> float:
    return np.linalg.norm(v)


def vector_hat(v: Vector) -> Vector:
    mag = vector_norm(v)
    if mag == 0:
        return v
    else:
        return v / mag


# %% [markdown] id="IGWUcnwfxnSN"
# ## Simulation

# %% id="Ob-qx0zcIcWM"
from numpy import deg2rad, pi


def make_system(params: Params) -> System:
    theta = deg2rad(params.angle)
    vx, vy = pol2cart(theta, params.speed)

    init = State(x=params.x, y=params.y, vx=vx, vy=vy)
    middel = pi * (params.diameter / 2) ** 2
    mass = sphere_mass(params.diameter, params.rho)
    return System(params, init=init, middel=middel, mass=mass)


def acceleration(V: Vector, system: System) -> Vector:
    mass, g = system.mass, system.g

    a_drag = drag_force(V, system) / mass
    a_grav = g * Vector(0, -1)
    return a_drag + a_grav


def drag_force(V: Vector, system: System) -> Vector:
    C_d, rho_air, middel = system.C_d, system.rho_air, system.middel

    magnitude = C_d * rho_air * middel * vector_norm(V) ** 2 / 2
    direction = -vector_hat(V)
    return direction * magnitude


# %% id="MSGoi6QhvK5e"
from typing import Tuple


def slope_func(t, state: np.ndarray, system: System) -> Tuple:
    x, y, vx, vy = state
    V = Vector(vx, vy)

    A = acceleration(V, system)
    return V.x, V.y, A.x, A.y


def event(t, state, system):
    x, y, vx, vy = state
    return y


# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="WBEgl4AA4gGH" outputId="2706cfed-179b-469d-e092-fa457c9ee5ed"
params = Params(
    x=0,  # [m]
    y=1,  # [m]
    angle=15,  # [degree]
    speed=40,  # [m / s]
    diameter=5e-2,  # [m]
    rho=27,  # body density [kg/m**3]
    C_d=0.3,  # coef
    rho_air=1.2,  # [kg/m**3]
    g=9.8,  # [m/s**2]
    t_end=10,  # [s]
    num=69,  # number of points
)

system = make_system(params)

results, details = run_solve_ivp(system, slope_func, events=event, dense_output=True)
details.message

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="8VNamwvgoo-f" outputId="75db0ace-9c0d-40ed-d74e-9813863a46ac"
results

# %% colab={"base_uri": "https://localhost:8080/"} id="qATqhMDQBCo5" outputId="97bc3e08-4546-4b5d-b165-5435ebf9d159"
x_dist = results.iloc[-1].x
x_dist

# %% colab={"base_uri": "https://localhost:8080/", "height": 294} id="3BwcDchKFUry" outputId="704b57d8-9a6b-46b7-c504-b9000e18613f"
results.x.plot()
results.y.plot()

decorate(xlabel="Time (s)", ylabel="Position (m)")

# %% colab={"base_uri": "https://localhost:8080/", "height": 297} id="Gy1HtFciGajr" outputId="27748d52-0641-4781-b260-ac0488f925cb"
make_series(results.x, results.y).plot(label="trajectory")

decorate(xlabel="x position (m)", ylabel="y position (m)")

# %% [markdown] id="2vQuJHrB2Azw"
# ## Dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="FIdNE8RwzmZ8" outputId="8d0775c7-0923-4ea5-eed4-f20a7d672c42" magic_args="echo skipping" language="script"
# params = Params(
#   x = 0,            # [m]
#   y = 0.5,          # [m]
#   angle = 85,       # [degree]
#   speed = 10,       # [m / s]
#
#   diameter = 5e-2,  # [m]
#   rho = 160,        # body density [kg/m**3]
#
#   C_d = 0.3,        # coef
#   rho_air = 1.8,    # [kg/m**3] https://en.wikipedia.org/wiki/Density_of_air#Dry_air
#   g = 9.8,          # [m/s**2]
#
#   t_end = 15,       # [s]
# )
#
# show(params)

# %% colab={"base_uri": "https://localhost:8080/"} id="Pl7f1mIf44Kt" outputId="370b5a96-b10f-47e5-a7d6-c7fc15492863" magic_args="echo skipping" language="script"
# system = make_system(params)
# results, details = run_solve_ivp(system,
#                                  slope_func,
#                                  events=event)
#
# results

# %% colab={"base_uri": "https://localhost:8080/"} id="lLdmd0y75Xib" outputId="33716aa4-74ea-4a09-c822-09153753f63b" magic_args="echo skipping" language="script"
# from tqdm import tqdm
#
# def make_dataset(params: Params) -> pd.DataFrame:
#   df =  pd.DataFrame(columns=params.__dict__.keys())
#   df = fill_dataset(df, params)
#   return df
#
# def fill_dataset(df: pd.DataFrame, params: Params) -> pd.DataFrame:
#   for _ in tqdm(change_params(params)):
#     system = make_system(params)
#     results, details = run_solve_ivp(system,
#                                      slope_func,
#                                      events=event)
#     result_row = concate(params, results)
#     df = df.append(result_row, ignore_index=True)
#   return df
#
# def change_params(params: Params) -> Params:
#   for angle in range(5, 85, 8):
#     params.angle = angle
#
#     for diameter in range(5, 15, 1):
#       params.diameter = diameter * 1e-2
#
#       for speed in range(10, 50, 5):
#         params.speed = speed
#
#         for y in np.arange(5e-1, 1, 5e-2):
#           params.y = y
#
#           yield params
#
# def concate(params: Params, results: pd.DataFrame) -> dict:
#   res = params.__dict__
#   res['x_dist'] = results.iloc[-1].x
#   res['t'] = results.index[-1]
#   return res
#
#
# dataset = make_dataset(params)
# dataset = dataset.drop(columns='t_end')
#
# dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="DKNWw02LSsXv" outputId="89b423b4-c147-4bbd-d336-fa251d6f8f80" magic_args="echo skipping" language="script"
# from google.colab import files
#
# dataset.to_csv('quasi_experiment_of_sphere_flight.csv', encoding = 'utf-8')
# files.download('quasi_experiment_of_sphere_flight.csv')

# %% [markdown] id="Tx2TM-SqLNpS"
# ## Models

# %% [markdown] id="wqgB_pVpLRbf"
# ### Data preporation

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="70KUmAo7WcnH" outputId="2a0a2b80-b32a-4bbd-97d0-fc7ee95d229d"
from random import seed, uniform

import pandas as pd

seed(7)


def make_noise_data(s: pd.Series, err_presentage: float) -> pd.Series:
    return s + s.apply(lambda x: uniform(-x * err_presentage, x * err_presentage))


data = pd.read_csv(
    "https://gist.githubusercontent.com/4539453/\
406368d0ed2d9d407f4cff1b6b9ef90d/raw/628b65fbc502f0b3ad69df758f4f0e93dcebe015/\
quasi_experiment_of_sphere_flight.csv",
    index_col=0,
)

data.x_dist = make_noise_data(data.x_dist, 5e-2)
data.speed = make_noise_data(data.speed, 2e-2)
data.angle = make_noise_data(data.angle, 2e-2)
data.y = make_noise_data(data.y, 2e-2)

data = data.round(2)
data = data.drop(columns=["t"])
# data = data.drop(columns=['t', 'C_d'])
data = data.sample(frac=1, random_state=1).reset_index(drop=True)

data.tail()

# %% [markdown] id="svPguDxARz8b"
# ### Parameter estimation for DEs

from typing import Any, Callable, List

# %% colab={"base_uri": "https://localhost:8080/"} id="k4B95n7B2ZRO" outputId="fb0d2796-7293-44dd-e8d5-3be5e1e461fb"
# # %%script echo skipping
from scipy.optimize import least_squares, leastsq


def init_systems(
    make_system: Callable[[Params], System],
    data: pd.DataFrame,
) -> List[System]:
    systems = []
    for i in range(0, data.shape[0]):
        params = Params(**data.iloc[i].to_dict())
        params.t_end = 15
        systems.append(make_system(params))
    return systems


def calc_deltas(guess: np.ndarray, systems: Any) -> List[float]:
    deltas = []
    for system in systems:
        system.C_d = guess[0]
        results, details = run_solve_ivp(
            system, slope_func, events=event, dense_output=True
        )
        delta = results.iloc[-1].x - system.x_dist
        deltas.append(delta)
    return deltas


systems = init_systems(make_system, data[:1])
guess = [0.1]

guess, covs = leastsq(calc_deltas, guess, args=systems)
guess[0]

# %% [markdown] id="ixFqXfU9-D84"
# ### Regression... boooooring...

# %% id="RWRmrsQL8jVX"
