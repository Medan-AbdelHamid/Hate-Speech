# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:16:24 2017

@author: Medyan
"""

from sklearn.preprocessing import PolynomialFeatures
import numpy as np
X = np.arange(6).reshape(3,2)
poly = PolynomialFeatures()
print(poly.fit_transform(X))
