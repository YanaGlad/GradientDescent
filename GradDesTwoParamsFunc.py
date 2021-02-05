from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from ForTesting import LoggingCallback


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
def plot_convergence_2d(func, steps, ax, xlim, ylim, cmap="viridis", title=""):

    ax.set_title(title, fontsize=20, fontweight="bold")

    xrange = np.linspace(*xlim, 100)
    yrange = np.linspace(*ylim, 100)
    grid = np.meshgrid(xrange, yrange)
    X, Y = grid
    fvalues = func(
        np.dstack(grid).reshape(-1, 2)
    ).reshape((xrange.size, yrange.size))
    ax.pcolormesh(xrange, yrange, fvalues, cmap=cmap, alpha=0.8)
    CS = ax.contour(xrange, yrange, fvalues)
    ax.clabel(CS, CS.levels, inline=True)

    arrow_kwargs = dict(linestyle="--", color="black", alpha=0.8)
    for i, _ in enumerate(steps):
        if i + 1 < len(steps):
            ax.arrow(
                *steps[i],
                *(steps[i+1] - steps[i]),
                **arrow_kwargs
            )

    n = len(steps)
    color_list = [(i / n, 0, 0, 1 - i / n) for i in range(n)]
    ax.scatter(steps[:, 0], steps[:, 1], c=color_list, zorder=10)
    ax.scatter(steps[-1, 0], steps[-1, 1],
               color="red", label=f"estimate = {np.round(steps[-1], 2)}")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")
    ax.legend(fontsize=16)
    plt.show()


def test_convergence_2d(grad_descent_2d, test_cases, tol, axes=None):
    right_flag = True
    debug_log = []
    for i, key in enumerate(test_cases.keys()):

        answer = test_cases[key]["answer"]
        test_input = deepcopy(test_cases[key])
        del test_input["answer"]

        callback = LoggingCallback()  # Не забываем про логирование
        res_point = grad_descent_2d(**test_input, callback=callback)

        if axes is not None:
            ax = axes[np.unravel_index(i, shape=axes.shape)]
            plot_convergence_2d(
                np.vectorize(test_input["func"], signature="(n)->()"),
                np.vstack(callback.x_steps),
                ax=ax,
                xlim=(test_input["low"], test_input["high"]),
                ylim=(test_input["low"], test_input["high"]),
                title=key
            )

        if np.linalg.norm(answer - res_point, ord=1) > tol:
            debug_log.append(
                f"Test '{key}':\n"
                f"\t- answer: {answer}\n"
                f"\t- counted: {res_point}"
            )
            right_flag = False
    return right_flag, debug_log


test_cases = {
    "concentric_circles": {
        "func": lambda x: (
                -1 / ((x[0] - 1) ** 2 + (x[1] - 1.5) ** 2 + 1)
                * np.cos(2 * (x[0] - 1) ** 2 + 2 * (x[1] - 1.5) ** 2)
        ),
        "low": -5,
        "high": 5,
        "answer": np.array([1, 1.5])
    }
}
tol = 1e-3

fig, axes = plt.subplots(figsize=(10, 10), squeeze=False)
fig.suptitle("Grad desc", fontsize = 25, fontweight = "bold")

is_correct, debug_log = test_convergence_2d(grad_descent_2d, test_cases, tol, axes)

if not is_correct:
    print("Wrong")
    for log_entry in debug_log:
        print(log_entry)
