import numpy as np
import matplotlib.pyplot as plt


def numerical_derivative_2d(func, epsilon):
    def grad_func(x):
        return np.array(
            [(func([x[0] + epsilon, x[1]]) - func(x)) / epsilon,
             (func([x[0], x[1] + epsilon]) - func(x)) / epsilon])

    return grad_func


def grad_descent_2d(func, low, high, callback=None):
    lr = 1
    eps = 1e-5
    estimates = []
    best = low
    deriv = numerical_derivative_2d(func, 1e-5)

    for j in range(low, high):
        x = np.array(([j, j]), dtype=float)
        for _ in range(100000):
            x_pred = x
            x -= lr * deriv(x)
            callback(x, func(x))
            if abs(func(x) - func(x_pred)) < abs(eps):
                break
        estimates.append(x)
        best = estimates[0]

        for e in estimates:
            if func(e) < func(best):
                best = e

    return best


# Testing
def plot_convergence_2d(func, steps, ax,
                        xlim, ylim, cmap="viridis", title=""):
    ax.set_title(title, fontsize=20, fontweight="bold")

    xrange = np.linspace(*xlim, 100)
    yrange = np.linspace(*ylim, 100)

    grid = np.meshgrid(xrange, yrange)

    X, Y = grid
    fvalues = func(np.dstack(grid).reshape(-1, 2)).reshape((xrange.size, yrange.size))

    ax.pcolormesh(xrange, yrange, fvalues, cmap=cmap, alpha=0.8)
    CS = ax.countour(xrange, yrange, fvalues)
    ax.clabel(CS, CS.levels, inline=True)

    arrow_kwargs = dict(linestyle="--", color="black", alpha=0.8)
    for i, _ in enumerate(steps):
        if i + 1 < len(steps):
            ax.arrow(
                *steps[i],
                *(steps[i + 1] - steps[i]),
                **arrow_kwargs
            )
