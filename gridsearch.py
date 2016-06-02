from sklearn.base import BaseEstimator, ClassifierMixin


class TestCheckin(BaseEstimator, ClassifierMixin):

    cache = None

    def __init__(self, a, b):
        print("cache in init",self.cache)
        if self.cache is None:
            print("init")
        else:
            print("init - rerun")
        self.a = a
        self.b = b

    def fit(self, X, y):
        print("fit",self.a,self.b,X,y)
        # print("cache",self.cache)
        self.cache = { 'b': 2 }
        return self

    def score(self, X, y, sample_weight=None):
        print("score",X,y)
        return 0

from sklearn import grid_search

clf = TestCheckin(a=1, b=2)
clf.fit([1,2,3],[4,5,6])
clf = TestCheckin(a=1, b=2)
clf.fit([1,2,3],[4,5,6])
# search = grid_search.GridSearchCV(clf, { 'a': [5, 2, 3]}, refit=False)
# search.fit([1, 2, 3],[4,5,6])
