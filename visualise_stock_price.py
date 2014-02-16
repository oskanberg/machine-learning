
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filename = 'stock_price.csv'
data = []

with open(filename, 'rbU') as f:
    reader = csv.reader(f)
    data = list(reader)

# cast each data item to float from string
data = [map(float, t) for t in data]

day_data = [ day for day in xrange(len(data)) ]
price_data = [ day[1] for day in data ]
volume_data = [ day[0] for day in data ]
delta_price_data = [ 0 ]
for day in xrange(1, len(price_data)):
    delta_price_data.append(price_data[day] - price_data[day - 1])
assert len(delta_price_data) == len(price_data)


fig = plt.figure()
axes = Axes3D(fig)

axes.scatter(day_data, price_data, delta_price_data)
plt.show()

#plt.plot(delta_price_data)
#plt.show()