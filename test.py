import argparse
import pandas as pd
from pymongo import MongoClient
import facebookcheckins as fc
from sklearn.grid_search import ParameterSampler
from scipy.stats.distributions import uniform, randint
import numpy as np

client = MongoClient("mongodb://optimizer:vzEL0f7e5bS7gvaR5hLu@ds017173.mlab.com:17173/facebook")
db = client.facebook
scores = db.scores

print("read train.df")
# train = train = pd.read_csv("train.csv", index_col="row_id").values
train = pd.read_pickle("train.df").values

limit = 400000

# permutation = np.random.permutation(len(train))
# test_indicies = permutation[slice(0,min(limit,len(permutation)))]
X_test = train[len(train)-limit:,0:4]
y_test = train[len(train)-limit:,4]

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

param_distributions = p.parse_args().options
print("arguments",param_distributions)
for key in param_distributions.keys():
    param_distributions[key] = eval(param_distributions[key])

import time
np.random.seed(int(time.time()))

clf = fc.FacebookCheckins(train[:-limit], e_factor=0, year_hist_bins=0)

score = clf.test(X_test, y_test, X_is_in_train_set=False)
print(score)
