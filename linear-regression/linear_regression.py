import numpy as np


def cost(x, theta, y):
    return 1/(2*len(x)) * np.sum(np.square(np.dot(x, theta) - y))


def gradient_descent(x, theta, y, alpha, iterations):
    j_vals = np.zeros((iterations, 1))
    new_theta = theta

    for i in range(0, iterations):
        new_theta -= alpha/x.shape[0] * np.dot(x.T, (np.dot(x, theta) - y))
        j_vals[i] = cost(x, new_theta, y)

    return j_vals, new_theta


def predict(x, theta):
    return np.dot(x, theta)[0, 0]
