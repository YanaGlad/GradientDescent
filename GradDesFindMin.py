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
    ax.scatter(x_steps, y_steps, c=color_list)
    ax.scatter(x_steps[-1], y_steps[-1], c="red")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")


class LoggingCallback:
    def __init__(self):
        self.x_steps = []
        self.y_steps = []

    def __call__(self, x, y):
        self.x_steps.append(x)
        self.y_steps.append(y)


def test_convergence_1d(grad_descent, test_cases, tol=1e-2, axes=None, grid=None):
    right_flag = True
    debug_log = []

    for i, key in enumerate(test_cases.keys()):
        answer = test_cases[key]["answer"]
        test_input = deepcopy(test_cases[key])
        del test_input["answer"]

        callback = LoggingCallback()
        res_point = grad_descent(**test_input, callback=callback)

        if axes is not None:
            ax = axes[np.unravel_index(i, shape=axes.shape)]
            x_steps = np.array(callback.x_steps)
            y_steps = np.array(callback.y_steps)
            plot_convergence_1d(
                test_input["func"], x_steps, y_steps, ax, grid, key
            )
            ax.axvline(answer, 0, linestyle="--", c="red",
                       label=f"true answer = {answer}")

            ax.axvline(x_steps[-1], 0, linestyle="--", c="xkcd:tangerine",
                       label=f"estimate = {np.round(x_steps[-1], 3)}")
            ax.legend(fontsize=16)

            if abs(answer - res_point) > tol or np.isnan(res_point):
                debug_log.append(
                    f"Test  'key' :\n"
                    f"\t - answer: {answer}\n"
                    f"\t - counted: {res_point}"
                )
                right_flag = False

        return right_flag, debug_log


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
