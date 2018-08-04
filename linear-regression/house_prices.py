import pandas as pd
import numpy as np
import linear_regression as linear


def predict_house_price(x, mu, sigma, theta):
    # Apply normalization with the values we got initially
    x_norm = (x - mu.values)/sigma.values
    # Add intercept term at first column
    x_norm = np.append(np.ones((x_norm.shape[0], 1)), x_norm, axis=1)

    return linear.predict(x_norm, theta)


def start():
    # ****** Multivariate Linear Regression ******
    # Prepare dataset
    data = pd.read_csv('datasets/house_prices.csv', header=0)
    x = data[['Size', 'Bedrooms']]
    y = data[['Price']]

    # Normalize x, since features differ by many orders of magnitude
    x_norm, mu, sigma = linear.normalize(x)
    
    # Add intercept (bias) term
    x_norm.insert(0, 'Intercept', 1)
    x_norm = x_norm.values
    y = y.values
    theta = np.zeros((x_norm.shape[1], 1))
    
    print('Cost with theta [0; 0; 0]: {0}'.format(linear.cost(x_norm, theta, y)))
    
    alpha = 0.01
    iterations = 500

    # Run gradient descent to minimize the error
    new_theta, j_vals = linear.gradient_descent(x_norm, theta, y, alpha, iterations)
    linear.plot_cost(j_vals)

    # Our cost is millions of times lower!
    print('\nNew theta: [{0}; {1}; {2}]'.format(new_theta[0][0], new_theta[1][0], new_theta[2][0]))
    print('Final cost: ', linear.cost(x_norm, new_theta, y))
    print('\nA house of 3000sqft and 3 bedrooms costs around ', predict_house_price(np.array([[3000, 3]]), mu, sigma, new_theta))


if __name__ == '__main__':
    start()
