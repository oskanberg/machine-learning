
import csv, math
from feature_expansion import FeatureExpander
from numpy import array, dot, delete, power, zeros, exp, add, log, insert
from numpy.random import shuffle
from scipy.optimize import fmin_bfgs

#import warnings
#warnings.filterwarnings('error')

def format_data(data):
    # the classifications will always be the last element
    y = data[:, -1]
    # X is everything else
    x = delete(data, -1, 1)
    # prepend ones for theta-0
    x = insert(x, 0, 1, axis=1)
    return x, y

def probability_of_class(theta_index, thetas, data_point):
    normaliser = log_sum_exp(data_point, thetas)
    # exp(x) creates overflow for x > 709
    if normaliser > 709:
        # just return smallest thing possible
        from sys import float_info
        return float_info.min
    else:
        numerator = exp(dot(data_point, thetas[theta_index]))
        denominator = exp(normaliser)
        return  numerator / denominator

def get_predicted_classifications(thetas, x, y):
    thetas = [thetas[i:i+x.shape[1]] for i in range(0, len(thetas), x.shape[1])]
    predicted_classifications = []
    for data_point, classification in zip(x, y):
        best = { 'probability' : 0, 'theta_index' : -1 }
        for theta_index, theta in enumerate(thetas):
            if probability_of_class(theta_index, thetas, data_point) > best['probability']:
                best['probability'] = probability_of_class(theta_index, thetas, data_point)
                best['theta_index'] = theta_index
        predicted_classifications.append(best['theta_index'])
    return predicted_classifications

def gradient(thetas, x, y):
    # fmin_bfgs flattens this, so reshape it
    thetas = [thetas[i:i+x.shape[1]] for i in range(0, len(thetas), x.shape[1])]
    gradient = [array([0.0 for dimension in theta]) for theta in thetas]
    for theta_index, theta in enumerate(thetas):
        for data_point, classification in zip(x, y):
            indicator = 1 if int(classification) == theta_index else 0
            tmp = indicator - probability_of_class(theta_index, thetas, data_point)
            gradient[theta_index] += dot(data_point, tmp)
    return array(gradient).flatten()

def log_sum_exp(data_point, thetas):
    result = 0
    # x.theta
    x_dot_theta = [dot(data_point, theta) for theta in thetas]
    max_val = max(x_dot_theta)
    # exp(x.theta - max)
    exp_x_dot_theta = [exp(lp - max_val) for lp in x_dot_theta]
    # max + log(sum(exp(x.theta - max)))
    return max_val + log(sum(exp_x_dot_theta))

def l(thetas, x, y):
    # fmin_bfgs flattens this, so reshape it
    thetas = [thetas[i:i+x.shape[1]] for i in range(0, len(thetas), x.shape[1])]
    result = 0
    for data_point, classification in zip(x, y):
        classification = int(classification)
        # thetac = the theta for this datapoint's class
        theta = thetas[classification]
        # sum(x.thetac - log_sum_exp(x.theta0, x.theta1, ...))
        result += dot(data_point, theta) - log_sum_exp(data_point, thetas)
    # return negative so we can minimise
    return -result

def get_accuracy(theta, x, y):
    win = 0
    for i in zip(get_predicted_classifications(theta, x, y), map(int,y)):
        #print i
        if i[0] == i[1]:
            win += 1
    return (win / float(len(y))) * 100

def get_random_guess_accuracy(x, y):
    import random
    win = 0
    random_theta = [random.randint(-1000, 1000) for classification in set(y)]
    for i in zip(get_predicted_classifications(random_theta, x, y)[0], y):
        if i[0] == i[1]:
            win += 1
    return win / float(len(y))

def logistic(data):
    # get rid of price
    # data = delete(data, -1, 1)
    # give it a good mix
    data = array(data)
    shuffle(data)
    #split as evenly as possible
    split_point = math.floor(data.shape[0] / 2)
    split0 = data[0:split_point,:]
    split1 = data[split_point:,:]

    # number of unique elements
    classes = len(set(format_data(data)[1]))
    thetas = [[0 for i in xrange(data.shape[1])] for j in xrange(classes)]

    x0, y0 = format_data(split0)

    print 'minimising split 0'
    minimised0 = fmin_bfgs(l, thetas, fprime=gradient, args=(x0, y0)).ravel().tolist()

    x1, y1 = format_data(split1)
    print 'minimising split 1'
    minimised1 = fmin_bfgs(l, thetas, fprime=gradient, args=(x1, y1)).ravel().tolist()

    print minimised0,minimised1
    mse0 = get_accuracy(minimised0, x1, y1)
    mse1 = get_accuracy(minimised1, x0, y0)
    print 'minimised0 accuracy:', mse0
    #print 'random guess accuracy', get_random_guess_accuracy(x1, y1)
    print 'minimised1 accuracy:', mse1
    #print 'random guess accuracy', get_random_guess_accuracy(x0, y0)
    return (mse0 + mse1) / 2.0


fe = FeatureExpander('stock_price.csv', True)
for feature in fe.get_all_features():
    fe.expand(feature)

fe.classify(FeatureExpander._classification_price_change)
fe.write_to_file('feature_expansion_tmp.csv')

result = logistic(fe.get_data())

print result