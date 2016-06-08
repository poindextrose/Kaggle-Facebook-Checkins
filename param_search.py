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

import os.path
if not os.path.exists("train.df"):
    print("creating train.df")
    import pandas as pd
    unsorted = pd.read_csv("train.csv", index_col="row_id", dtype={'x': np.float32, 'y': np.float32,
                                                                   'accuracy': np.int32, 'time': np.int32})
    all = unsorted.sort_values('time', axis=0)
    all.to_pickle("train.df")
else:
    print("read train.df")
    all = pd.read_pickle("train.df").values

limit = 400000

train = all[:-limit]
X_test = all[slice(len(train),None),0:4]
y_test = all[slice(len(train),None),4]

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

clf = fc.FacebookCheckins(train)

while True:
    param_list = list(ParameterSampler(param_distributions, n_iter=1))
    for params in param_list:
        clf.set_params(**params)
        output = {}
        output['params'] = clf.get_params()
        output['score'] = clf.test(X_test, y_test, X_is_in_train_set=False)
        output['limit'] = limit
        print(output['score'], output['params'])
        scores.insert(output)
