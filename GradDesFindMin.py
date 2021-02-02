from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt


# Finds one local min that matches global
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
def plot_convergence_1d(func, x_steps, y_steps, ax, grid=None, title=""):
    ax.set_title(title, fontsize=16, fontweight='bold')

    if grid is None:
        grid = np.linspace(np.min(x_steps), np.max(x_steps), 100)

    fgrid = [func(item) for item in grid]
    ax.plot(grid, fgrid)
    yrange = np.max(fgrid) - np.min(fgrid)

    arrow_kwargs = dict(linestyle="--", color="grey", alpha=0.4)

    for i, _ in enumerate(x_steps):
        if i + 1 < len(x_steps):
            ax.arrow(
                x_steps[i], y_steps[i],
                x_steps[i + 1] - x_steps[i],
                y_steps[i + 1] - y_steps[i],
                **arrow_kwargs
            )

    n = len(x_steps)
    color_list = [(i / n, 0, 0, 1 - i / n) for i in range(n)]
