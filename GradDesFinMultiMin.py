from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt

# Finds global local min out of local mins
from ForTesting import test_convergence_1d


def grad_descent_v2(func, deriv, low=None, high=None, callback=None):
    if low is None:
        low = -4
    if high is None:
        high = 4

    N = 30000
    eps = 0.002

    best_estimate = low

    for estimate in range(low, high):
        for i in range(N):
            estimate_last = estimate
            estimate -= eps * deriv(estimate)
            callback(estimate, func(estimate))
            if abs(estimate_last - estimate) < eps ** 2:
                break
            if func(best_estimate) > func(estimate):
                best_estimate = estimate

    return best_estimate


test_cases = {
    "poly1": {
        "func": lambda x: x ** 4 + 3 * x ** 3 + x ** 2 - 1.5 * x + 1,
        "deriv": lambda x: 4 * x ** 3 + 9 * x ** 2 + 2 * x - 1.5,
        "low": -3, "high": 3, "answer": -1.88
    },
    "poly2": {
        "func": lambda x: x ** 4 + 3 * x ** 3 + x ** 2 - 2 * x + 1.0,
        "deriv": lambda x: 4 * x ** 3 + 9 * x ** 2 + 2 * x - 2.0,
        "low": -3, "high": 3, "answer": 0.352
    },
    "another poly": {
        "func": lambda x: x ** 6 + x ** 4 - 10 * x ** 2,
        "deriv": lambda x: 6 * x ** 5 + 4 * x ** 3 - 20 * x,
        "low": 0, "high": 2, "answer": 1 / 3 * np.sqrt((np.sqrt(31) - 1) * 3)
    },
    "another yet poly": {
        "func": lambda x: x ** 6 + x ** 4 - 10 * x ** 2 - x,
        "deriv": lambda x: 6 * x ** 5 + 4 * x ** 3 - 20 * x - 1,
        "low": -2, "high": 2, "answer": 1.24829
    },
    "and another yet poly": {
        "func": lambda x: x ** 20 + x ** 2 - 20 * x + 10,
        "deriv": lambda x: 20 * x ** 19 + 2 * x - 20,
        "low": -1, "high": 2, "answer": 0.994502
    },
    "|x|/x^2 - x + sqrt(-x) + (even polynom)": {
        "func": lambda x: 5 * np.abs(x) / x ** 2 - 0.5 * x + 0.1 * np.sqrt(-x) + 0.01 * x ** 2,
        "deriv": lambda x: -0.5 - 0.05 / np.sqrt(-x) + 0.02 * x + 5 / (x * np.abs(x)) - (10 * np.abs(x)) / x ** 3,
        "low": -5, "high": -2, "answer": -2.91701
    },
}

tol = 1e-2  # желаемая точность

fig, axes = plt.subplots(2, 4, figsize=(24, 8))
fig.suptitle("Grad Descent ver 2", fontweight="bold", fontsize=20)
grid = np.linspace(-3, 3, 100)

is_correct, debug_log = test_convergence_1d(
    grad_descent_v2, test_cases, tol,
    axes, grid
)

if not is_correct:
    print("Incorrect")
    for log_entry in debug_log:
        print(log_entry)
