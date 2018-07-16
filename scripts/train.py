#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 18:40:32 2018

@author: abhithegreat
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl


data = pd.read_csv("data/train_data.txt", sep = ";")

x = data.iloc[: , :-1].values
y = data['y'].values

rf_fit = RandomForestClassifier()
rf_fit.fit(x, y)

# save this model. 
pkl.dump(rf_fit, open("models/rf_model.pkl", 'wb'))