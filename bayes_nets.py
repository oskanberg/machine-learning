
import csv, sys, random
from collections import deque
from itertools import product
from numpy import mean
from numpy.random import beta as beta_distribution
from matplotlib import pyplot as plt

def get_csv_rows(filename, cast_filetype=None):
    with open(filename, 'rbU') as f:
        reader = csv.reader(f)
        if cast_filetype:
            return [ map(cast_filetype, t) for t in reader ]
        else:
            return [ t for t in reader ]

def get_mean_conditional_probability(subject, conditions, data):
    #print 'getting', subject, 'wrt.', conditions
    def conditions_hold(sample):
        for condition in conditions:
            if sample[condition[0]] != condition[1]:
                return False
        return True

    alpha = len([True for sample in data if conditions_hold(sample) and sample[subject]])
    beta = len([True for sample in data if conditions_hold(sample) and not sample[subject]])
    # pull from the mean distribution
    #print mean(beta_distribution(alpha + 1, beta + 1, size=(10000,)))
    # or, just calculate the distribution mean (alpha/(alpha+beta))
    return alpha / float(alpha + beta)

def bnbayesfit(StructureFileName, DataFileName):
    node_connections = get_csv_rows(StructureFileName, int)
    data = get_csv_rows(DataFileName, int)

    # fittedbn[5]['parents'] = parents to node 5
    # fittedbn[5]['children'] = children to node 5
    # if 5 has one parent (e.g. 4):
    # fittedbn[5]['given'][True] = p(5|4=True)
    # fittedbn[5]['given'][False] = p(5|4=False)
    # if 5 has two parents (e.g. 4, 1):
    # fittedbn[5]['given'][(False, True)] = p(5|1=False, 4=True)
    # if 5 has no parents:
    # fittedbn[5]['given'][tuple()] = p(5)
    # parents always sorted in ascending order
    fittedbn = {
        index : {
            'parents' : [],
            'children' : [],
            'given' : {}
        } for index, node in enumerate(node_connections)
    }

    for node_number, connections in enumerate(node_connections):
        children = [i for i, x in enumerate(connections) if x]
        for child in children:
            fittedbn[node_number]['children'].append(child)
            fittedbn[child]['parents'].append(node_number)

    for node in fittedbn:
        # sort smallest first
        fittedbn[node]['parents'].sort()
        logical_combinations = [combination for combination in product([True, False], repeat=len(fittedbn[node]['parents']))]
        parent_combinations = [zip(fittedbn[node]['parents'], combination) for combination in logical_combinations]
        for parent_combination in parent_combinations:
            if len(parent_combination) == 1:
                fittedbn[node]['given'][parent_combination[0][1]] = get_mean_conditional_probability(node, parent_combination, data)
            else:
                # there are several parents, or none
                # if there are none, key will be empty tuple
                logical_tuple = tuple(condition[1] for condition in parent_combination)
                fittedbn[node]['given'][logical_tuple] = get_mean_conditional_probability(node, parent_combination, data)
    return fittedbn

def sample_probability(mean):
    if random.uniform(0, 1) <= mean:
        return True
    else:
        return False

def bnsample(fittedbn, samples):
    result = []
    for n in xrange(samples):
        # start with nodes with no parents
        edge_nodes = deque(node_number for node_number, element in fittedbn.iteritems() if len(element['parents']) == 0)
        sample = [-1 for node in fittedbn]
        while edge_nodes:
            node = edge_nodes.popleft()
            parents = fittedbn[node]['parents']
            # check all parents have been determined
            determined_parents = [parent for parent in parents if sample[parent] != -1]
            if len(determined_parents) != len(parents):
                # re-add this node, try later
                edge_nodes.append(node)
                continue
            # at this point we know parents have been sampled
            parent_states = -1
            if len(parents) == 1:
                parent_states = sample[parents[0]]
            else:
                # if parents == 0, we will use empty tuple
                parent_states = tuple(sample[parent] for parent in parents)
            sample[node] = int(sample_probability(fittedbn[node]['given'][parent_states]))
            for child in fittedbn[node]['children']:
                edge_nodes.append(child)
        result.append(sample)
    return result


from matplotlib import pyplot as plt
import numpy

fittedbn = bnbayesfit('bnstruct.csv', 'bndata.csv')
data = get_csv_rows('bndata.csv', int)
for i in xrange(100):
    samples = bnsample(fittedbn, 10000)
    real_proportions = numpy.matrix(data).sum(axis=0) / float(len(data))
    sample_proportions = numpy.matrix(samples).sum(axis=0) / float(len(samples))
    error = real_proportions - sample_proportions
    error = error.ravel().tolist()[0]
    plt.plot(list(error), 'bo')
    print 'done', i,'of',100
plt.ylim(-2.0, 2.0)
plt.show()
