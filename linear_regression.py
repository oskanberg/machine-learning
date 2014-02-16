
import csv
from numpy import loadtxt, zeros, ones, array, delete, insert
from scipy.optimize import fmin
from matplotlib import pyplot as plt

def get_data(input_file_name):
    with open(input_file_name, 'rbU') as f:
        reader = csv.reader(f)
        data = [ map(float, t) for t in reader ]
    return data

def squared_loss(theta, x, y):
    # theta.X
    predictions = x.dot(theta)
    # (y - theta.x) ^ 2
    total_loss = (y - predictions.flatten()) ** 2
    # sum((y-theta.x) ^ 2 )
    return total_loss.sum()

def f(theta, x):
    return theta[0] * 1 + theta[1] * x

def linear(input_file_name):
    data = array(get_data(input_file_name))

    # the stock price will always be the last element
    y = data[:, -1]
    # X is everything else
    x = delete(data, -1, 1)
    # prepend ones for theta-0
    x = insert(x, 0, 1, axis=1)

    # initialise theta (num of x rows) zeros
    theta = ones(shape=(x.shape[1], 1))

    # minimise the squared loss function wrt theta
    minimised = fmin(squared_loss, theta, args=(x, y))

    # return MSE
    return squared_loss(theta, x, y) / x.shape[0]
    
    #plot the results
    #plt.scatter(x[:, 1], y)
    #f_x = [ i for i in xrange(1500000) ]
    #f_y = [ f(minimised, i) for i in f_x ]
    #plt.plot(f_x, f_y)
    #plt.show()


if __name__ == '__main__':
    print linear('stock_price.csv')