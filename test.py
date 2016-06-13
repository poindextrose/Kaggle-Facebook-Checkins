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
train = pd.read_pickle("train.df").values

limit = 400000

# permutation = np.random.permutation(len(train))
# test_indicies = permutation[slice(0,min(limit,len(permutation)))]
X_test = train[len(train)-limit:,0:4]
y_test = train[len(train)-limit:,4]

clf = fc.FacebookCheckins(train[:-limit])

score = clf.test(X_test, y_test, X_is_in_train_set=False)
print(score)
