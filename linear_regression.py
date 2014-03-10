
import csv, math
from numpy import zeros, ones, matrix, array, delete, insert, std, mean, dot, reshape
from numpy.random import shuffle
from itertools import combinations
from scipy.optimize import fmin_bfgs
from matplotlib import pyplot as plt

p = True

def get_data(input_file_path):
    with open(input_file_path, 'rbU') as f:
        reader = csv.reader(f)
        data = [ map(float, t) for t in reader ]
    return data

def format_data(data):
    # the stock price will always be the last element
    y = data[:, -1]
    # X is everything else
    x = delete(data, -1, 1)
    # prepend ones for theta-0
    x = insert(x, 0, 1, axis=1)
    return x, y

def squared_loss(theta, data):
    x, y = format_data(data)
    # theta.X
    predictions = dot(x, theta)
    # (y - theta.x) ^ 2
    losses = (y - predictions.flatten()) ** 2
    # sum((y-theta.x) ^ 2 )
    return losses.sum()

def gradient(theta, data):
    x, y = format_data(data)
    predictions = dot(x, theta)
    error = predictions - y
    gradient = dot(x.T, error) / x.shape[0]
    return gradient

def learn(data):
    x, y = format_data(data)
    # initialise theta (num of x columns) zeros
    theta = zeros(x.shape[1])
    # minimise the squared loss function wrt theta
    result = fmin_bfgs(squared_loss, theta, fprime=gradient, args=(data,))
    return result

def linear(input_file_name):
    data = array(get_data(input_file_name))
    # shuffle the data
    shuffle(data)
    #split as evenly as possible
    split_point = math.floor(data.shape[0] / 2)
    split0 = data[0:split_point,:]
    split1 = data[split_point:,:]
    # learn theta on each set
    minimised0 = learn(split0)
    minimised1 = learn(split1)
    print 'split0: %s\nsplit1: %s' % (str(minimised0), str(minimised1))
    # calculate MSEs
    mse0 = squared_loss(minimised0, split0) / split0.shape[0]
    mse1 = squared_loss(minimised1, split1) / split1.shape[0]
    print 'split0 MSE: %s\nsplit1 MSE: %s' % (str(mse0), str(mse1))
    # return average MSE
    return (mse0 + mse1) / 2.0


##
# Class for feature expansion:
# adds elements to each data point and
# writes new data to a separate file
##
class FeatureExpander(object):

    def __init__(self, data):
        self.data = data

    def _feature_nothing(self, last_ten_days):
        return 0

    def _feature_price_yesterday(self, last_ten_days):
        return last_ten_days[-1][-1]

    def _feature_all_price_delta(self, last_ten_days):
        start_price = last_ten_days[0][-1]
        final_price = last_ten_days[-1][-1]
        return final_price - start_price

    def _feature_two_day_price_delta(self, last_ten_days):
        return last_ten_days[-1][-1] - last_ten_days[-2][-1]

    def _feature_all_stock_delta(self, last_ten_days):
        start_stock = last_ten_days[0][0]
        final_stock = last_ten_days[-1][0]
        return (final_stock - start_stock)

    def _feature_price_volitility(self, last_ten_days):
        prices = [ day[-1] for day in last_ten_days ]
        # return the standard deviation of the prices
        return std(prices)

    def _feature_total_value_of_trades(self, last_ten_days):
        # volume of trades * price of trades
        total_values = [ day[0] * day[-1] for day in last_ten_days ]
        return sum(total_values)

    def _feature_sum_of_price(self, last_ten_days):
        prices = [ day[-1] for day in last_ten_days ]
        return sum(prices)

    def _feature_mean_price(self, last_ten_days):
        prices = [ day[-1] for day in last_ten_days ]
        return mean(prices)

    def _feature_mean_value(self, last_ten_days):
        values = [ day[0] for day in last_ten_days ]
        return mean(values)

    def generate_feature_raised_to_power(self, feature, power):
        # return callable raising feature to power
        return lambda obj, last_ten_days: feature(obj, last_ten_days) ** power

    def feature_generator(self, new_feature_callable):
        for index in xrange(10, len(self.data)):
            row = self.data[index]
            # data points for last 10 days (excl. today)
            last_ten_days = self.data[index - 10:index]
            # add new feature value
            row.append(new_feature_callable(self, last_ten_days))
            # swap last two elements to make sure price is always last
            row[-1], row[-2] = row[-2], row[-1]
            yield row

    def expand(self, feature_callable):
        new_data = []
        for row in self.feature_generator(feature_callable):
            new_data.append(row)
        self.data = new_data

    def normalise_data(self):
        column_no = len(self.data[0])
        row_no = len(self.data)
        for column in xrange(column_no):
            column_sum = sum((x[column] for x in self.data))
            column_mean = column_sum / row_no
            for row in xrange(row_no):
                self.data[row][column] -= column_mean

    def write_to_file(self, file_path):
        # we automatically don't train on today's price
        # but we need to remove today's stock value
        for index, row in enumerate(self.data):
            self.data[index].pop(0)
        with open(file_path, 'w') as f:
            for row in self.data:
                row = map(str, row)
                csvs = ','.join(row)
                f.write(csvs)
                f.write('\n')

if __name__ == '__main__':
    expanded_file = 'feature_expansion_tmp.csv'

    features = [
        #[ FeatureExpander._feature_price_volitility,],
        #[ FeatureExpander._feature_all_price_delta,],
        [ FeatureExpander._feature_price_yesterday, FeatureExpander._feature_all_price_delta, FeatureExpander._feature_two_day_price_delta,],
        [ FeatureExpander._feature_price_yesterday ],
        #[ FeatureExpander._feature_all_stock_delta,],
        #[ FeatureExpander._feature_price_yesterday,],
        #[ FeatureExpander._feature_sum_of_price,],
        #[ FeatureExpander._feature_total_value_of_trades,],
        #[ FeatureExpander._feature_mean_price,],
        #[ FeatureExpander._feature_mean_value ],
    ]

    MSEs = {}

    for feature_set in features:
        fe = FeatureExpander(get_data('stock_price.csv'))
        fe.normalise_data()
        for feature in feature_set:
            for i in range(0, 2):
                raised = fe.generate_feature_raised_to_power(feature, i)
                fe.expand(raised)
        fe.write_to_file(expanded_file)
        MSEs[''.join([ f.__name__ for f in feature_set ])] = linear(expanded_file)

    print MSEs
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.5)
    plt.bar(range(len(MSEs)), MSEs.values(), align='center')
    plt.xticks(range(len(MSEs)), MSEs.keys())
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.show()