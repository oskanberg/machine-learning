
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

def regularised_squared_loss(theta, x, y, lmbda):
    # theta.X
    predictions = dot(x, theta)
    # (y - theta.x) ^ 2
    losses = (y - predictions.flatten()) ** 2
    # sum((y-theta.x) ^ 2
    loss = lmbda * losses.sum()
    complexity = (1 - lmbda) * theta.sum()
    return loss - complexity

def gradient(theta, x, y):
    predictions = dot(x, theta)
    error = (y - predictions.flatten())
    gradient = dot(x.T, error)
    return gradient

def sign(theta):
    new_theta = []
    for t in theta:
        if t < 0:
            new_theta.append(-1)
        elif t == 0:
            new_theta.append(0)
        else:
            new_theta.append(1)
    return array(new_theta)

def regularised_gradient(theta, x, y, lmbda):
    predictions = dot(x, theta)
    error = (y - predictions.flatten())
    first_part = (-2 * lmbda )* dot(x.T, error)
    second_part = (1 - lmbda) * sign(theta)
    return first_part + second_part

def learn(x, y):
    # initialise theta (num of x columns) zeros
    theta = ones(x.shape[1])
    # minimise the squared loss function wrt theta
    #result = fmin_bfgs(squared_loss, theta, fprime=gradient, args=(x,y))
    result = fmin_bfgs(squared_loss, theta, args=(x,y))
    return result

def regularised_learn(x, y, lmbda):
    # initialise theta (num of x columns) zeros
    theta = ones(x.shape[1])
    # minimise the squared loss function wrt theta
    #result = fmin_bfgs(squared_loss, theta, fprime=gradient, args=(x,y))
    result = fmin_bfgs(regularised_squared_loss, theta, fprime=regularised_gradient, args=(x,y,lmbda))
    return result

def linear_inner(data):
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

def regularised_linear_inner(data):
    data = array(data)
    # shuffle the data
    shuffle(data)
    # split as evenly as possible
    split_amount = math.floor(data.shape[0] / 3)
    split0 = data[0:split_amount,:]
    split1 = data[split_amount:split_amount * 2,:]
    split2 = data[split_amount * 2:split_amount * 3,:]

    x0, y0 = format_data(split0)
    x1, y1 = format_data(split1)
    x2, y2 = format_data(split2)

    lambda_record = []
    for lmbda in [t / 100.0 for t in xrange(100)]:
        minimised0 = regularised_learn(x0, y0, lmbda)
        minimised1 = regularised_learn(x1, y1, lmbda)
        mse0 = squared_loss(minimised0, x1, y1) / split0.shape[0]
        lambda_record.append([mse0, lmbda, minimised0])
        mse1 = squared_loss(minimised1, x0, y0) / split1.shape[0]
        lambda_record.append([mse1, lmbda, minimised1])

    best = (100000000, None, None)
    for record in lambda_record:
        if record[0] < best[0]:
            best = record
    print 'best:', best
    #print lambda_record


def linear(InputFileName):
    fe = FeatureExpander(InputFileName)
    print linear_inner(fe.get_data())

def reglinear(InputFileName):
    fe = FeatureExpander(InputFileName)
    print regularised_linear_inner(fe.get_data())

MSEs = {}
fe = FeatureExpander('stock_price.csv')
fe.normalise_data()
for feature in fe.get_all_features():
    if 'two_day_price' in feature.__name__:
        fe.expand(fe.generate_feature_raised_to_power(feature, 1))
        #fe.expand(fe.generate_feature_raised_to_power(feature, 2))
        fe.write_to_file('feature_expansion_tmp.csv')
        print feature

data = fe.get_data()

MSEs['a'] = regularised_linear_inner(data)
print MSEs

#fig = plt.figure()
##fe.reset()
##fe.normalise_data()
##
##x, y = format_data(array(data))
##
###print thetas
##
##plt.plot(array(fe.get_data())[10:,:])
##plt.plot(dot(x, thetas[0]))
##plt.plot(dot(x, thetas[1]))
##plt.show()
#fig.subplots_adjust(bottom=0.5)
#plt.bar(range(len(MSEs)), MSEs.values(), align='center')
#plt.xticks(range(len(MSEs)), MSEs.keys())
#locs, labels = plt.xticks()
#plt.setp(labels, rotation=90)
#plt.show()