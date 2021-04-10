# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 22:06:04 2017

@author: Medyan
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd
from sklearn import linear_model

df = pd.read_csv('iris.txt')

print(df.tail())


#print(df)
X = np.array(df.drop(df['iris'])) 
Y = np.array(df['iris'])
Y = Y[3:]
x = [5.6,2.7,4.2,1.3,0]

#print(len(X))
#print(np.ndim(Y))
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

#clf  = linear_model.LassoCV()

clf = svm.SVC()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(clf.predict([x]))

print(accuracy)

