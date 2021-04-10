# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:32:57 2017

@author: Medyan
"""
from sklearn import linear_model

X = [[0,0], [0,2], [1,1], [5,6]]
Y = [0, 0.1, 1, 5]
x = [0, 1]

r = linear_model.RidgeCV(alphas = [0.1, 1.0, 10.0])
#r = linear_model.LassoLars(alpha = 0.1)
#r = linear_model.Lasso(alpha = 0.1)
#r = linear_model.ElasticNet(alpha = 0.1)
#r = linear_model.BayesianRidge()
#r = linear_model.LogisticRegression()

r.fit(X, Y)

print(r.predict([x]))
#print(r.alpha_)
print(r.coef_)

#r = linear_model.Lasso(alpha=0.1)