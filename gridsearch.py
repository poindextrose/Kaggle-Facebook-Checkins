from sklearn.base import BaseEstimator, ClassifierMixin


class TestCheckin(BaseEstimator, ClassifierMixin):

    cache = {}

    def __init__(self, a, b):
        if 'a' in self.cache.keys() and self.cache['a'] == a:
            print("init - rerun")
        else:
            print("init")
        self.a = a
        self.b = b
        self.cache['a'] = a

    def fit(self, X, y):
        print("fit",self.a,self.b,X,y)
        return self

    def score(self, X, y, sample_weight=None):
        print("score",X,y)
        return 0

from sklearn import grid_search

clf = TestCheckin(a=1, b=2)
search = grid_search.GridSearchCV(clf, { 'a': [5, 2, 3]}, refit=False)
search.fit_predict([1, 2, 3],[4,5,6])