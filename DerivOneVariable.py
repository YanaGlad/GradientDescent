from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt


# Returns a func that counts a derivative for func in some point
def numerical_derivative_1d(func, epsilon):
    def deriv_func(x):
        return (func(x + epsilon) - func(x)) / epsilon
    return deriv_func

# Testing funcs
def polynom_to_prime(x):
    return 20 * x ** 5 + x ** 3 - 5 * x ** 2 + 2 * x + 2.0

def primed_poly(x):
    return 100 * x ** 4 + 3 * x ** 2 - 10 * x + 2.0


approx_deriv = numerical_derivative_1d(polynom_to_prime, 1e-5)

grid = np.linspace(-2, 2, 100)
right_flag = True
tol = 0.05
debug_print = []

for x in grid:
    estimation_error = abs(primed_poly(x) - approx_deriv(x))
    if estimation_error > tol:
        debug_print.append((estimation_error, primed_poly(x), approx_deriv(x)))
        right_flag = False

if not right_flag:
    print("Something went wrong...")
    primed_poly(debug_print)
    plt.plot(grid, primed_poly(grid), label='True derivative')
    plt.plot(grid, approx_deriv(grid), label='Counted derivative')
    plt.legend()

print(str(right_flag))
