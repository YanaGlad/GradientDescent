from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt

# Finds global local min out of local mins
def grad_descent_v2(func, deriv, low=None, high=None, callback=None):

    if low is None:
        low = -4
    if high is None:
        high = 4

    N = 30000
    eps = 0.002

    best_estimate = low

    for estimate in range(low, high):
        

    return best_estimate