# Introduction

This repo contains scripts for competing in the [Kaggle Facebook Checkins Competition](https://www.kaggle.com/c/facebook-v-predicting-check-ins)

# Starting

Start with [Explore Data.ipynb](https://github.com/poindextrose/Kaggle-Facebook-Checkins/blob/master/Explore%20Data.ipynb). It covers my initial exploration of the data.

# Model

I created a hybrid nearest neighbors model. That used the distance of nearby checkins combined with a probability of a user checking in based on the patterns seen in the time of day.

[facebookcheckins.py](https://github.com/poindextrose/Kaggle-Facebook-Checkins/blob/master/facebookcheckins.py) extends the
`BaseEstimator` and `ClassifierMixin` interface.

[test.py](https://github.com/poindextrose/Kaggle-Facebook-Checkins/blob/master/test.py) is where the model is evaluated.
