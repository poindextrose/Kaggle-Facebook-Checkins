from sklearn.grid_search import ParameterSampler
from scipy.stats.distributions import expon, gamma

for params in list(ParameterSampler({ 'a': gamma(a=1, scale=1)}, n_iter=20)):
    print(params)
