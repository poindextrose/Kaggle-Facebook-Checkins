#!/bin/sh

python param_search.py a_scale="gmmma(a=1,scale=1)" a_bias="expon(scale=200)" a_min="expon(0,20)" \
day_hist_bins="randint(1,24)" day_hist_min_prob="uniform(0,0.9)" \
week_hist_bins="randint(1,24)" week_hist_min_prob="uniform(0,0.9)" \
year_hist_bins="randint(1,24)" year_hist_min_prob="uniform(0,0.9)" \
e_factor="uniform(0,0.8)"
