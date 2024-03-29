
from copy import deepcopy
import csv, math
from numpy import mean, std

##
# feature expansion module:
# adds elements to each data point and
# writes new data to a separate file
##
class FeatureExpander(object):

    def __init__(self, file_path, normalise=None):
        if normalise == None:
            normalise = False
        self.original = file_path
        with open(file_path, 'rbU') as f:
            reader = csv.reader(f)
            data = [ map(float, t) for t in reader ]
            self.original_data = deepcopy(data)
            if normalise:
                print 'normalising'
                #self.normalise_data()
            self.data = deepcopy(self.original_data[10:])

    def reset(self):
        self.__init__(self.original)

    def _feature_price_yesterday(self, last_ten_days):
        return last_ten_days[-1][-1]

    def _feature_volume_yesterday(self, last_ten_days):
        return last_ten_days[0][-1]

    def _feature_all_price_delta(self, last_ten_days):
        start_price = last_ten_days[0][-1]
        final_price = last_ten_days[-1][-1]
        return final_price - start_price

    def _feature_two_day_price_delta(self, last_ten_days):
        return last_ten_days[-1][-1] - last_ten_days[-2][-1]

    def _feature_all_stock_delta(self, last_ten_days):
        start_stock = last_ten_days[0][0]
        final_stock = last_ten_days[-1][0]
        return (final_stock - start_stock) / 1000000.0

    def _feature_price_volatility(self, last_ten_days):
        prices = [ day[-1] for day in last_ten_days ]
        # return the standard deviation of the prices
        return std(prices)

    def _feature_total_value_of_trades(self, last_ten_days):
        # volume of trades * price of trades
        total_values = [ day[0] * day[-1] for day in last_ten_days ]
        return sum(total_values) / 100000000.0

    def _feature_mean_price(self, last_ten_days):
        prices = [ day[-1] for day in last_ten_days ]
        return mean(prices)

    #def _feature_mean_value(self, last_ten_days):
    #    values = [ day[0] for day in last_ten_days ]
    #    return mean(values)

    def _classification_price_change(self, today_and_yesterday):
        if today_and_yesterday[0][-1] == 0:
            # avoid division by zero
            import sys
            today_and_yesterday[0][-1] = sys.float_info.min
        percent_change = (today_and_yesterday[1][-1] / float(today_and_yesterday[0][-1]) * 100) - 100
        if math.fabs(percent_change) <= 5:
            #no change
            return 0
        elif percent_change > 5 and percent_change <= 10:
            # up
            return 1
        elif percent_change <= -5 and percent_change > -10:
            # down
            return 2
        elif percent_change > 10:
            # sharp up
            return 3
        elif percent_change <= -10:
            # sharp down
            return 4
        else:
            print percent_change
            raise Exception('class not classified properly')

    def get_data(self):
        data_to_return = self.data
        # we automatically don't train on today's price
        # but we need to remove today's stock value
        for index, row in enumerate(data_to_return):
            data_to_return[index].pop(0)
        return data_to_return

    def get_all_features(self):
        attributes = dir(self)
        return [ getattr(self, attribute) for attribute in attributes if attribute.startswith('_feature_') ]

    def generate_feature_raised_to_power(self, feature, power):
        # return callable raising feature to power
        return lambda last_ten_days: feature(last_ten_days) ** power

    def classification_generator(self, new_feature_callable):
        for index in xrange(0, len(self.data)):
            row = self.data[index]
            # data points for yesterday and today
            today_and_yesterday = self.original_data[index + 8:index+10]
            today_and_yesterday = today_and_yesterday[:]
            # replace price with class
            row[-1] = new_feature_callable(self, today_and_yesterday)
            #row[-1], row[-2] = row[-2], row[-1]
            yield row

    def feature_generator(self, new_feature_callable):
        for index in xrange(0, len(self.data)):
            row = self.data[index]
            # data points for last 10 days (excl. today)
            # original_data is 10 longer than data
            last_ten_days = self.original_data[index:index + 10][:]
            # add new feature value
            row.append(new_feature_callable(last_ten_days))
            # swap last two elements to make sure price is always last
            row[-1], row[-2] = row[-2], row[-1]
            yield row

    def expand(self, feature_callable):
        new_data = []
        for row in self.feature_generator(feature_callable):
            new_data.append(row)
        self.data = new_data

    def classify(self, feature_callable):
        new_data = []
        for row in self.classification_generator(feature_callable):
            new_data.append(row)
        self.data = new_data

    def normalise_data(self):
        column_no = len(self.original_data[0])
        row_no = len(self.original_data)
        for column in xrange(column_no):
            column_data = [x[column] for x in self.original_data]
            column_mean = mean(column_data)
            column_std_dev = std(column_data)
            for row in xrange(row_no):
                self.original_data[row][column] = (self.original_data[row][column] - column_mean) / column_std_dev

    def write_to_file(self, file_path):
        with open(file_path, 'w') as f:
            for row in self.data:
                row = map(str, row)
                csvs = ','.join(row)
                f.write(csvs)
                f.write('\n')