
import csv
from numpy import zeros, ones, array, delete, insert, std
from scipy.optimize import minimize
from matplotlib import pyplot as plt

p = True

def get_data(input_file_path):
    with open(input_file_path, 'rbU') as f:
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
    result = minimize(squared_loss, theta, args=(x, y))
    minimised = result.x
    print 'Minimised theta: %s' % str(minimised)

    global p
    if p:
        # only plot y once
        plt.plot(y, 'r-')
        p = False
    predictions = x.dot(minimised).flatten()
    plt.plot(predictions)
    plt.draw()

    # return MSE
    return squared_loss(theta, x, y) / x.shape[0]

##
# Class for feature expansion:
# adds elements to each data point and 
# writes new data to a separate file
##
class FeatureExpander(object):

    def __init__(self, data):
        self.data = data

    def feature_price_yesterday(self, last_ten_days):
        return last_ten_days[-1][-1]

    def feature_price_delta(self, last_ten_days):
        start_price = last_ten_days[0][-1]
        final_price = last_ten_days[-1][-1]
        return final_price - start_price

    def feature_price_delta_squared(self, last_ten_days):
        start_price = last_ten_days[0][-1]
        final_price = last_ten_days[-1][-1]
        return (final_price - start_price) ** 2

    def feature_stock_delta(self, last_ten_days):
        start_stock = last_ten_days[0][0]
        final_stock = last_ten_days[-1][0]
        return (final_stock - start_stock)

    def feature_stock_delta_squared(self, last_ten_days):
        start_stock = last_ten_days[0][0]
        final_stock = last_ten_days[-1][0]
        return (final_stock - start_stock) ** 2

    def feature_price_volitility(self, last_ten_days):
        prices = [ day[-1] for day in last_ten_days ]
        # return the standard deviation of the prices
        return std(prices)

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

    def write_to_file(self, file_path):
        with open(file_path, 'w') as f:
            for row in self.data:
                row = map(str, row)
                csvs = ','.join(row)
                f.write(csvs)
                f.write('\n')

if __name__ == '__main__':
    # engineer new features
    data = get_data('stock_price.csv')
    fe = FeatureExpander(data)

    expanded_file = 'feature_expansion_tmp.csv'

    fe.expand(FeatureExpander.feature_stock_delta)
    #fe.expand(FeatureExpander.feature_stock_delta_squared)
    fe.expand(FeatureExpander.feature_price_delta)
    #fe.expand(FeatureExpander.feature_price_delta_squared)
    #fe.expand(FeatureExpander.feature_price_volitility)
    fe.expand(FeatureExpander.feature_price_yesterday)
    fe.write_to_file(expanded_file)

    print linear(expanded_file)
    plt.show()

