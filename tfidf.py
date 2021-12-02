# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:56:33 2021

@author: Medyan
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

def get_ifidf_for_words(text):
    tfidf_matrix= tfIdfVectorizer.transform([text]).todense()
    feature_index = tfidf_matrix[0,:].nonzero()[1]
    tfidf_scores = zip([feature_names[i] for i in feature_index], [tfidf_matrix[0, x] for x in feature_index])
    tfidf = np.sort(list(dict(tfidf_scores).values()))[::-1]
    n = len(tfidf)
    return np.pad(tfidf, (0, 30-n), 'constant')


dataset1 = pd.read_csv("thesis-trainE.csv")
dataset = dataset1['FULLTEXT']
tfIdfVectorizer = TfidfVectorizer(min_df=3)
tfIdf = tfIdfVectorizer.fit(dataset)
feature_names = tfIdfVectorizer.get_feature_names()
df = pd.DataFrame(dataset[1], index=feature_names, columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))
tf = get_ifidf_for_words('تضرب بهالشكل متل العقرب')
print(tf)