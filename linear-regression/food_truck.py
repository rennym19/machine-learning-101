import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
from linear_regression import *


def predict_profit(population, params, print_msg=True):
    prediction = predict(np.array([[1, population/10000]]), params)
    if print_msg:
        print('For a population of {0}, we predict a profit of {1}'.format(population, prediction * 10000))
    return prediction


def start():
    # Changing plots' style
    style.use('ggplot')

    # Get and Visualize dataset
    data = pd.read_csv('datasets/food_truck.csv', delimiter=',', header=0)
    x = data[['Population']]
    y = data[['Profits']]

    plt.scatter(x['Population'], y['Profits'], color='red', marker='.')
    plt.title('Change in Profits in relation to Population')
    plt.xlabel('Population in 10,000s')
    plt.ylabel('Profits in $ 10,000s')
    plt.show()

    # Add intercept (bias) column to x and check initial cost with all params = 0
    x.insert(0, 'Intercept', 1)
    x = np.array(x)
    y = np.array(y)
    theta = np.zeros((x.shape[1], 1))

    print('Cost with theta [0; 0]: {0}'.format(cost(x, theta, y)))
    print('Cost with theta [-1; 2]: {0}'.format(cost(x, np.array([[-1], [2]]), y)))

    # Setting hyperparameters before running gradient descent
    alpha = 0.01
    iterations = 1500

    # Run gradient descent to minimize the error
    j_vals, new_theta = gradient_descent(x, theta, y, alpha, iterations)
    print('\nNew theta: [{0}; {1}]\n'.format(new_theta[0][0], new_theta[1][0]))

    # Plotting cost history
    print('Displaying cost history...')
    plt.plot(j_vals)
    plt.title('Cost History over 1500 iterations of GD')
    plt.legend('J')
    plt.xlabel('N° Iterations')
    plt.ylabel('Cost')
    plt.show()

    # Plotting Line of Best Fit
    print('Displaying line of best fit...')
    plt.scatter(x[:, 1], y[:, 0], color='red', marker='.')
    plt.plot(x[:, 1], np.dot(x, theta))
    plt.xlabel('Population in 10,000s')
    plt.ylabel('Profits in $ 10,000s')
    plt.show()

    # Making predictions
    user_choice = input('Do you want to make a prediction? (y/n)')
    while user_choice == 'y':
        predict_profit(int(input('Enter population: ')), new_theta)
        user_choice = input('Do you want to make another prediction? (y/n)')


if __name__ == '__main__':
    start()
