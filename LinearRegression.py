# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 16:46:30 2021

@author: Medyan
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def best_fit(X, Y):
    print(X.shape)
    print(X.shape)
    model = LinearRegression(fit_intercept=False, n_jobs=-1)
    model.fit(X, Y)
    r_sq = model.score(X, Y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
