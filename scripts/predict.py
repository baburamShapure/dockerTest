#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 18:53:59 2018

@author: abhithegreat
"""

from sklearn.ensemble import RandomForestClassifier
import pickle as pkl 
import sys
import os

def predict_row(x):
    model = pkl.load(open("models/rf_model.pkl", "rb"))
    return(model.predict_proba(x)[:,1])
    


if __name__ == '__main__':
    input_args = sys.argv[1:]   
    pred = predict_row(input_args)
    print(pred[0])