# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 21:47:08 2017

@author: Medyan
"""

import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]
df['HL_PCT'] = 100.0*(df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']
df['PCT_change'] = 100.0*(df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume', ]]
forecast_col = 'Adj. Close'
forecast_out =math.ceil( 0.05*len(df))

df.fillna(-9999, inplace=True)
df['label'] =df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X = preprocessing.scale(X)
X = X[:-forecast_out]
XL = X[-forecast_out:]
df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
clf = LinearRegression(n_jobs=20)
#clf = LinearRegression()
#clf = svm.SVR(kernel='poly')
#clf = svm.LinearSVR(kernel='poly')
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
forecast_set = clf.predict(XL)
print(forecast_set, accuracy, forecast_out)

df['forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
oneday = 86400
next_unix = last_unix + oneday

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += oneday
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    
df['Adj. Close'].plot()
df['forecast'].plot()
plt.legendnd(loc=4)
plt.xlabel('DAte')
plt.ylabel('Price')
plt.show()
