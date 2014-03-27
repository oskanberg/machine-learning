
import csv, math
from feature_expansion import FeatureExpander
from numpy import array, dot, delete, power, zeros, exp, add, log, insert
from scipy.optimize import fmin_bfgs

import warnings
warnings.filterwarnings('error')


def format_data(data):
    # the stock price will always be the last element
    y = data[:, -1]
    # X is everything else
    x = delete(data, -1, 1)
    # prepend ones for theta-0
    x = insert(x, 0, 1, axis=1)
    return x, y

def log_sum_exp(data_point, thetas):
    result = 0
    max_val = max(data_point)
    data_point = data_point - max_val
    # sum j (e^x.thetaj)
    for theta in thetas:
        try:
            result += exp(dot(data_point, theta))
        except Warning:
            print 'balls'
            print data_point, theta
            raw_input()
    if result == 0:
        # avoid log(0)
        import sys
        result = sys.float_info.min
    return max_val + log(result)

def l(thetas, x, y, ind_theta_len):
    #reconstruct
    thetas = [thetas[i:i+ind_theta_len] for i in range(0, len(thetas), ind_theta_len)]
    result = 0
    for index, row in enumerate(x):
        theta_index = int(y[index])
        theta = thetas[theta_index]
        result += dot(row, theta) - log_sum_exp(row, thetas)
    return -result

with open('stock_price.csv', 'rbU') as f:
    reader = csv.reader(f)
    data = array([ map(float, t) for t in reader ])

fe = FeatureExpander('stock_price.csv')
for index, feature in enumerate(fe.get_all_features()):
    print index, feature.__name__
    fe.expand(feature)
fe.normalise_data()
fe.classify(FeatureExpander._classification_price_change)
fe.write_to_file('feature_expansion_tmp.csv')
data = array(fe.get_data())

# get rid of price
data = delete(data, -1, 1)

x, y = format_data(data)
classes = 4

x0 = [[1 for i in xrange(x.shape[1])] for j in xrange(classes + 1)]
#print x0
print fmin_bfgs(l, x0, args=(x, y, x.shape[1]))