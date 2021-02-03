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

#Testing