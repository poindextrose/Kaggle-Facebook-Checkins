#!/bin/sh

python param_search.py a_bias="uniform(0,20)" a_min="uniform(0,20)" \
day_hist_bins=randint(1,24)" day_hist_min_prob="uniform(0,0.5)" \
week_hist_bins=randint(1,24)" week_hist_min_prob="uniform(0,0.5)" \
year_hist_bins=randint(1,24)" year_hist_min_prob="uniform(0,0.5)" \
e_factor="uniform(0,0.8)"
