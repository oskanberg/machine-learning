from scipy.optimize import fmin
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    data = []
    x = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y

def f(x0, x):
    prediction = 0
    prediction += x0[0] * x
    prediction += x0[1] * x ** 2
    prediction += x0[1] * x ** 3
    prediction += x0[1] * (x - 5)
    prediction += x0[1] * (x - 10)
    prediction += x0[1] * (x - 15)
    return prediction

def squaredLoss(x0, x, y):
    assert len(x0) == len(x)
    total_error = 0
    # for each sample
    for sample_index in xrange(len(y) - 1):
        # calculate prediction
        prediction = 0
        for term_index in xrange(len(x) - 1):
            prediction += x0[term_index] * x[term_index][sample_index]
        total_error += (y[sample_index] - prediction) ** 2
    return total_error

def subtract_mean(l):
    mean = sum(l) / len(l)
    for i in xrange(len(l) - 1):
        l[i] = l[i] - mean
    return l

x_raw, y_raw = read_data('ageData.csv')

x = [
    subtract_mean(x_raw),
    subtract_mean([i ** 2 for i in x_raw]),
    subtract_mean([i ** 3 for i in x_raw]),
    subtract_mean([max(i - 5, 0) for i in x_raw]),
    subtract_mean([max(i - 10, 0) for i in x_raw]),
    subtract_mean([max(i - 15, 0) for i in x_raw]),
]
y = subtract_mean(y_raw)

# initialise theta
x0 = [ 0 for i in x ]
print "initial fitness:\t", squaredLoss(x0, x, y)
minimised = fmin(squaredLoss, x0, (x, y), xtol=0.01, disp=False)
print "new fitness:\t\t", squaredLoss(minimised, x, y)
print list(minimised)

plt.scatter(x[0], y)
f_x = np.linspace(-15, 15, 100)
f_y = [ f(minimised, i) for i in f_x ]
plt.plot(f_x, f_y)
plt.show()
