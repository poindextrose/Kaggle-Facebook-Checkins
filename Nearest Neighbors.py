import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_validation import train_test_split
from sklearn.metrics import average_precision_score
import numpy as np
import numexpr as ne
import argparse
from pymongo import MongoClient
from sklearn.grid_search import ParameterSampler

day = 1440
hour = 60
week = 10080

limit = 400000

print("reading train.csv")
train = pd.read_csv("train.csv", index_col="row_id")

xy = train.iloc[:,:2].values # units are kilometers
accuracy = train.iloc[:,2].values * 0.001 # assume accuracy is reported in meters so convert to kilometers
time = train.iloc[:,3].values # units are minutes
time_of_day = train.iloc[:,3].values % 1440 # minutes
time_of_week = train.iloc[:,3].values % 10080 # minutes
place_id = train.iloc[:,4].values

neigh = NearestNeighbors(n_jobs=-1, algorithm='kd_tree')

def NN_fit():
    print("NearestNeighbors.fit()")
    neigh.fit(xy, place_id)

permutation = np.random.permutation(len(xy))

validation_indicies = permutation[slice(0,min(limit,len(xy)))]

def time_difference(time1, time2, period=None):
    """Find the different in time even if measure is periodic."""
    if period:
        hp = 0.5 * period
        return ne.evaluate('hp-abs(abs(time1-time2) - hp)')
    else:
        return ne.evaluate('abs(time1-time2)')

def prob_overlap_time(diff, w1, w2, mp):
    """Compute the probability the the time difference is significant."""
    # derive equation of line that connects end of w1 and w2
    # points: (w1, 1), (w2, mp)
    # dy = mp-1, dx = w2-w1, m = (mp-1)/(w2-w1)
    # y = m * x + b
    # substitude in point 1
    # 1 = (mp-1)/(w2-w1) * w1 + b
    # solve for b
    # b = 1 - (mp-1)/(w2-w1) * w1
    # y = (mp-1)/(w2-w1) * x + 1 - (mp-1)/(w2-w1)
    prob = ne.evaluate('(mp-1)/(w2-w1) * diff + 1 - (mp-1)/(w2-w1) * w1')
    prob = np.where(diff < w1, 1, prob)
    return np.where(diff > w2, mp, prob)

def uniqify(seq):
    """Removes duplicates from sequence and maintains order."""
    seen = set()
    seen_add = seen.add
    return np.fromiter((x for x in seq if not (x in seen or seen_add(x))), dtype=np.int64)

def prob_overlap_locations(dist, accuracy1, accuracy2):
    """Compute the probability that location measurements represent the same point."""
    return ne.evaluate('exp(-0.5 * dist * dist / (accuracy1 ** 2 + accuracy2 ** 2)) / (accuracy1 ** 2 + accuracy2 ** 2)')

def sum_by_group(values, groups):
    """Sum a list of values by groups."""
    order = np.argsort(groups)
    groups = groups[order]
    values = values[order]
    values.cumsum(out=values)
    index = np.ones(len(groups), 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    values = values[index]
    groups = groups[index]
    values[1:] = values[1:] - values[:-1]
    return values, groups

def scale_accuracy(accuracy, params):
    scale = params['a_scale']
    bias = params['a_bias']
    a_min = params['a_min']
    return np.maximum(accuracy + bias, a_min) * scale
    
def predict_xy_accuracy_time(test_points, distances, neighbors, parameters, self_validation=False):
    
    neighbor_accuracies = scale_accuracy(accuracy[neighbors], parameters)
    test_accuracy = scale_accuracy(accuracy[test_points, None], parameters)
    colocation_prob = prob_overlap_locations(distances, test_accuracy, neighbor_accuracies)
    
    time_of_day_diff = time_difference(time_of_day[test_points, None], time_of_day[neighbors], day)
    time_of_day_prob = prob_overlap_time(time_of_day_diff, parameters['day_w1'], parameters['day_w2'], parameters['day_mp'])
    
    time_of_week_diff = time_difference(time_of_week[test_points, None], time_of_week[neighbors], week)
    time_of_week_prob = prob_overlap_time(time_of_week_diff, parameters['week_w1'], parameters['week_w2'], parameters['week_mp'])
    
    time_abs_diff = time_difference(time[test_points, None], time[neighbors])
    time_abs_prob = prob_overlap_time(time_abs_diff, parameters['abs_w1'], parameters['abs_w2'], parameters['abs_mp'])
    
    total_prob = ne.evaluate('colocation_prob * time_of_day_prob * time_of_week_prob * time_abs_prob')
    
    s = slice(1,None) if self_validation else slice(0,None) # skip the first neighbor if self validating
    predictions = np.zeros((len(distances),3))
    for i, (prob, places) in enumerate(zip(total_prob[:,s], place_id[neighbors][:,s])):
        # append a few zeros just incase there is only one nearby place
        # we need three for the precision calculation
        prob, places = sum_by_group(np.append(prob, [0,0]), np.append(places, [0,1]))
        prob, places = zip(*sorted(zip(prob, places),reverse=True))
        predictions[i,:] = places[:3]
    return predictions
        
def mean_average_precision3(true, test):
    precision = np.array([1, 1/2, 1/3])
    return ne.evaluate('sum((true == test) * precision)') / len(true)

class StoreInDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        d = getattr(namespace, self.dest)
        for opt in values:
            k,v = opt.split("=", 1)
            k = k.lstrip("-")
            d[k] = v
        setattr(namespace, self.dest, d)

# Prevent argparse from trying to distinguish between positional arguments
# and optional arguments. Yes, it's a hack.
p = argparse.ArgumentParser( prefix_chars=' ' )

# Put all arguments in a single list, and process them with the custom action above,
# which convertes each "--key=value" argument to a "(key,value)" tuple and then
# merges it into the given dictionary.
p.add_argument("options", nargs="*", action=StoreInDict, default=dict())

args = p.parse_args()
# print(args.options)

parameters = { 'kNN': 400,
               'a_scale': 1,
               'a_min': 1,
               'a_bias': 0,
               'day_w1': 0*day,
               'day_w2': 12*hour,
               'day_mp': 0,
               'week_w1': 0*day,
               'week_w2': 3.5*day,
               'week_mp': 0,
               'abs_w1': 20*week,
               'abs_w2': 50*week,
               'abs_mp': 1
             }

for key in args.options:
    parameters[key] = eval(args.options[key])
print(parameters)

client = MongoClient("mongodb://optimizer:bOQ0QxKl1oKX@ds015760.mlab.com:15760/facebook")
scores = client.facebook.scores1

NN_fit()

print("find nearest neighbors")
distances, neighbors = neigh.kneighbors(xy[validation_indicies], n_neighbors=parameters['kNN'])



print("predict")
predictions = predict_xy_accuracy_time(validation_indicies, distances, neighbors, parameters, self_validation=True)

print("evaluate")
mean_average_precision3(place_id[validation_indicies, None], predictions)

# params = {'a': eval('[1,2,3]')}
# list(ParameterSampler(params, n_iter=3))
