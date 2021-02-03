from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt


# Finds one local min that matches global
from ForTesting import test_convergence_1d


def grad_descent_v1(func, deriv, start=None, callback=None):
    if start is None:
        np.random.seed(179)
        start = np.random.randn()

    lr = 1e-3
    epsilon = 1e-5

    estimate = start
    while deriv(estimate) > epsilon:
        estimate -= lr * deriv(estimate)
        callback(estimate, func(estimate))

    return estimate


# Testing
test_cases = {
    "square": {
        "func": lambda x: x * x,
        "deriv": lambda x: 2 * x,
        "start": 2,
        "answer": 0.0
    },
    "module": {
        "func": lambda x: abs(x),
        "deriv": lambda x: 1 if x > 0 else -1,
        "start": 2,
        "answer": 0.0
    },
    "third_power": {
        "func": lambda x: abs((x - 1) ** 3),
        "deriv": lambda x: 3 * (x - 1) ** 2 * np.sign(x - 1),
        "start": -1,
        "answer": 1.0
    },
    "ln_x2_1": {
        "func": lambda x: np.log((x + 1) ** 2 + 1),
        "deriv": lambda x: 2 * (x + 1) / (x ** 2 + 1),
        "start": 1,
        "answer": -1.0
    }
}

tol = 1e-2
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Grad descent ver 1", fontweight="bold", fontsize=20)
grid = np.linspace(-2, 2, 100)

is_correct, debug_log = test_convergence_1d(
    grad_descent_v1, test_cases, tol,
    axes, grid
)
plt.show()

if not is_correct:
    print("Incorrect")
    for log_entry in debug_log:
        print(log_entry)
