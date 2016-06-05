from sklearn.neighbors import NearestNeighbors
import numpy as np
import numexpr as ne
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import poisson

day = 1440 # minutes
hour = 60 # minutes
week = 10080 # minutes
year = 525960 # minutes
kilometers_per_meter = 0.001

class FacebookCheckins(BaseEstimator, ClassifierMixin):
    
    def __init__(self, train, a_scale = 1, a_bias = 0, a_min = 0, kNN=400,
                 day_hist_bins=24, day_hist_min=1, day_hist_min_prob=0,
                 week_hist_bins=7, week_hist_min=1, week_hist_min_prob=0,
                 year_hist_bins=12, year_hist_min=1, year_hist_min_prob=0,
                 e_factor=2/3):
        
        self.x = train[:,0] # kilometers
        self.y = train[:,1] # kilometers
        self.accuracy = train[:,2] * kilometers_per_meter # convert meters to kilometers
        self.time = train[:,3] # units are minutes
        self.time_of_day = train[:,3] % day 
        self.time_of_week = train[:,3] % week
        self.time_of_year = train[:,3] % year
        self.place_id = train[:,4].astype(np.int64)
        self.kNN = kNN
        self.a_scale = a_scale
        self.a_bias = a_bias
        self.a_min = a_min
        self.day_hist_bins = day_hist_bins
        self.day_hist_min = day_hist_min
        self.day_hist_min_prob = day_hist_min_prob
        self.week_hist_bins = week_hist_bins
        self.week_hist_min = week_hist_min
        self.week_hist_min_prob = week_hist_min_prob
        self.year_hist_bins = year_hist_bins
        self.year_hist_min = year_hist_min
        self.year_hist_min_prob = year_hist_min_prob
        self.e_factor = e_factor
        
        # values to cache to prevent duplicating work on successive calls to test()
        self.neighbors = None
        self.X_test = None
        
        self.NN = NearestNeighbors(n_jobs=-1, algorithm='kd_tree').fit(train[:,0:2], self.place_id)
        
        self.end_time = np.amax(self.time)
        
        self.day_hist = {}
        self.week_hist = {}
        self.year_hist = {}
        self.prob_in_business = {}
        
    def set_params(self, **params):
        for key, value in params.items():
            if 'day_hist_bins' == key and self.day_hist_bins != value:
                self.day_hist = {}
            if 'day_hist_min' == key and self.day_hist_min != value:
                self.day_hist = {}
            if 'day_hist_min_prob' == key and self.day_hist_min_prob != value:
                self.day_hist = {}
            if 'week_hist_bins' == key and self.week_hist_bins != value:
                self.week_hist = {}
            if 'week_hist_min' == key and self.week_hist_min != value:
                self.week_hist = {}
            if 'week_hist_min_prob' == key and self.week_hist_min_prob != value:
                self.week_hist = {}
            if 'year_hist_bins' == key and self.year_hist_bins != value:
                self.year_hist = {}
            if 'year_hist_min' == key and self.year_hist_min != value:
                self.year_hist = {}
            if 'year_hist_min_prob' == key and self.year_hist_min_prob != value:
                self.year_hist = {}
            if 'e_factor' == key and self.e_factor != value:
                self.prob_in_business = {}
            if 'kNN' == key and self.kNN != value:
                self.neighbors = None
        return BaseEstimator.set_params(self, **params)
        
    def _generate_time_prob(self):
        do_day = False if len(self.day_hist) else True
        do_week = False if len(self.week_hist) else True
        do_year = False if len(self.year_hist) else True
        do_business = False if len(self.prob_in_business) else True
        
        if not (do_day or do_week or do_business):
            return
        
        if do_day:
            print("compute day probability")
        if do_week:
            print("compute week probability")
        if do_year:
            print("compute year probability")
        if do_business:
            print("compute in business probability")
            first_time = {}
            last_time = {}
            total_checkins = {}
        
        pids = set()
        for pid, tod, tow, time in zip(self.place_id, self.time_of_day, self.time_of_week, self.time):
            if pid not in pids:
                pids.add(pid)
                if do_day:
                    self.day_hist[pid] = np.zeros((self.day_hist_bins))
                if do_week:
                    self.week_hist[pid] = np.zeros((self.week_hist_bins))
                if do_year:
                    self.year_hist[pid] = np.zeros((self.year_hist_bins))
                if do_business:
                    first_time[pid] = time
                    last_time[pid] = time
                    total_checkins[pid] = 0
            if do_day:
                day_index = int(tod / day * self.day_hist_bins)
                self.day_hist[pid][day_index] += 1
            if do_week:
                week_index = int(tow / week * self.week_hist_bins)
                self.week_hist[pid][week_index] += 1
            if do_year:
                year_index = int(tow / year * self.year_hist_bins)
                self.year_hist[pid][year_index] += 1
            if do_business:
                first_time[pid] = min(time, first_time[pid])
                last_time[pid] = max(time, last_time[pid])
                total_checkins[pid] += 1
        for pid in pids:
            if do_day:
                self.day_hist[pid] = np.minimum(1, np.maximum(np.maximum(self.day_hist_min, self.day_hist[pid])                                         / np.amax(self.day_hist[pid]), self.day_hist_min_prob))
            if do_week:
                self.week_hist[pid] = np.minimum(1, np.maximum(np.maximum(self.week_hist_min, self.week_hist[pid])                                         / np.amax(self.week_hist[pid]), self.week_hist_min_prob))
            if do_year:
                self.year_hist[pid] = np.minimum(1, np.maximum(np.maximum(self.year_hist_min, self.year_hist[pid])                                         / np.amax(self.year_hist[pid]), self.year_hist_min_prob))
            if do_business:
                end_interval = self.e_factor * self.end_time - last_time[pid]
                if end_interval > 0:
                    avg_rate_per_end_interval = int((total_checkins[pid] - 1) / end_interval)
                else:
                    avg_rate_per_end_interval = 0
                self.prob_in_business[pid] = poisson.cdf(1, avg_rate_per_end_interval)
        
    def _prob_overlap_locations(self, x1, y1, x2, y2, accuracy1, accuracy2):
        """Compute the probability that location measurements represent the same point."""
        return ne.evaluate('exp(-0.5 * ((x1-x2)**2+(y1-y2)**2) / (accuracy1 ** 2 + accuracy2 ** 2)) /                             (accuracy1 ** 2 + accuracy2 ** 2)') # / (2 * np.pi)

    def _sum_by_group(self, values, groups):
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

    def _prob_time_hist(self, unit, test, neighbors, hist, bins):
        prob = np.zeros_like(neighbors, dtype=np.float)
        for i, (t, n_indicies) in enumerate(zip(test,neighbors)):
            index = int(t / unit * bins) # scalar
            prob[i] = np.fromiter((hist[self.place_id[n]][index] for n in n_indicies), dtype=np.float)
        return prob
    
    def _prob_in_business(self, time, neighbors):
        prob = np.zeros_like(neighbors, dtype=np.float)
        for i, n_indicies in enumerate(neighbors):
            prob[i] = np.fromiter((self.prob_in_business[self.place_id[n]] for n in n_indicies), dtype=np.float)
        return prob

    def _predict(self, X, neighbors, self_validation=False):

        x_test = X[:,0].reshape((-1,1)) # units are kilometers
        y_test = X[:,1].reshape((-1,1)) # units are kilometers
        a_test = X[:,2].reshape((-1,1)) * 0.001
        time_test = X[:,3].reshape((-1,1))
        day_test = X[:,3].reshape((-1,1)) % 1440
        week_test = X[:,3].reshape((-1,1)) % 10080
        year_test = X[:,3].reshape((-1,1)) % 10080

        def scale_accuracy(accuracy):
            scale = self.a_scale
            bias = self.a_bias
            a_min = self.a_min
            return np.maximum(accuracy + bias, a_min) * scale

        neighbor_accuracies = scale_accuracy(self.accuracy[neighbors])
        test_accuracy = scale_accuracy(a_test)
        prob = self._prob_overlap_locations(x_test, y_test, self.x[neighbors], self.y[neighbors],
                                            test_accuracy, neighbor_accuracies)

        prob = prob * self._prob_time_hist(day, day_test, neighbors, self.day_hist, self.day_hist_bins)
        prob = prob * self._prob_time_hist(week, week_test, neighbors, self.week_hist, self.week_hist_bins)
        prob = prob * self._prob_time_hist(year, year_test, neighbors, self.year_hist, self.year_hist_bins)
        if not self_validation:
            prob = prob * self._prob_in_business(time_test, neighbors)

        s = slice(1,None) if self_validation else slice(0,None) # skip the first neighbor if self validating
        predictions = np.zeros((len(X),3),dtype=np.int64)
        for i, (p, places) in enumerate(zip(prob[:,s], self.place_id[neighbors][:,s])):
            # append a few zeros just incase there is only one nearby place
            # we need three for the precision calculation
            p, places = self._sum_by_group(np.append(p, [0,0]), np.append(places, [0,1]))
            p, places = zip(*sorted(zip(p, places),reverse=True))
            predictions[i,:] = places[:3]
        return predictions

    def _mean_average_precision3(self, true, test):
        precision = np.array([1, 1/2, 1/3])
        return ne.evaluate('sum((true == test) * precision)') / len(true)
    
    def test(self, X_test, y_test, X_is_in_train_set=True):
        predictions = self.predict(X_test, X_is_in_train_set)
        return self._mean_average_precision3(y_test.reshape((-1,1)), predictions)
    
    def predict(self, X_test, X_is_in_train_set=False):
        self._generate_time_prob()
        
        if self.X_test is not X_test:
            self.neighbors = None
            self.X_test = X_test
            
        if self.neighbors is None:
            print("  find nearest neighbors to test points")
            self.neighbors = self.NN.kneighbors(X_test[:,0:2], n_neighbors=self.kNN,
                                                   return_distance=False).astype(np.int32)
        return self._predict(X_test, self.neighbors, self_validation=X_is_in_train_set)
