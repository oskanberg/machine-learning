
import math
from feature_expansion import FeatureExpander
from numpy import zeros, ones, array, delete, insert, dot, reshape, matrix
from numpy.random import shuffle
from itertools import combinations
from scipy.optimize import fmin_bfgs
from matplotlib import pyplot as plt

thetas = []

def format_data(data):
    # the stock price will always be the last element
    y = data[:, -1]
    # X is everything else
    x = delete(data, -1, 1)
    # prepend ones for theta-0
    x = insert(x, 0, 1, axis=1)
    return x, y

def squared_loss(theta, x, y):
    # theta.X
    predictions = dot(x, theta)
    # (y - theta.x) ^ 2
    losses = (y - predictions.flatten()) ** 2
    # sum((y-theta.x) ^ 2
    return losses.sum()

def gradient(theta, x, y):
    predictions = dot(x, theta)
    error = (y - predictions.flatten())
    gradient = dot(x.T, error) / x.shape[0]
    return gradient

def learn(x, y):
    # initialise theta (num of x columns) zeros
    theta = ones(x.shape[1])
    # minimise the squared loss function wrt theta
    #result = fmin_bfgs(squared_loss, theta, fprime=gradient, args=(x,y))
    result = fmin_bfgs(squared_loss, theta, args=(x,y))
    return result

def linear(data):
    data = array(data)
    # shuffle the data
    shuffle(data)
    # split as evenly as possible
    split_point = math.floor(data.shape[0] / 2)
    split0 = data[0:split_point,:]
    split1 = data[split_point:,:]

    x0, y0 = format_data(split0)
    x1, y1 = format_data(split1)

    minimised0 = learn(x0, y0)
    mse0 = squared_loss(minimised0, x1, y1) / split0.shape[0]

    minimised1 = learn(x1,y1)
    mse1 = squared_loss(minimised1, x0, y0) / split1.shape[0]
    global thetas
    thetas = [minimised0, minimised1]
    #print thetas
    # return average MSE
    return (mse0 + mse1) / 2.0


MSEs = {}
fe = FeatureExpander('stock_price.csv')
fe.normalise_data()
for feature in fe.get_all_features():
    if 'price_yesterday' in feature.__name__ or 'all_price_delta' in feature.__name__:
        fe.expand(fe.generate_feature_raised_to_power(feature, 1))
        fe.expand(fe.generate_feature_raised_to_power(feature, 2))
        fe.write_to_file('feature_expansion_tmp.csv')
        print feature

data = fe.get_data()

MSEs['a'] = linear(data)
print MSEs


fig = plt.figure()
#fe.reset()
#fe.normalise_data()
#
#x, y = format_data(array(data))
#
##print thetas
#
#plt.plot(array(fe.get_data())[10:,:])
#plt.plot(dot(x, thetas[0]))
#plt.plot(dot(x, thetas[1]))
#plt.show()
fig.subplots_adjust(bottom=0.5)
plt.bar(range(len(MSEs)), MSEs.values(), align='center')
plt.xticks(range(len(MSEs)), MSEs.keys())
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()