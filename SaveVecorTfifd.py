# coding: utf-8

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
def load_csv_data(path):
    dataset = pd.read_csv(path)
    X, y = [], []
    text = dataset['FULLTEXT']  
    label = dataset["ANOMALY"]
    i=0
    for line in text:
        X.append(text[i])
        i=i+1
    i=0    
    for line in label:
        y.append(label[i])
        i=i+1
    return X, np.array(y)


# Main Run
        
print('loading  data')    
#test_data, test_labels = load_csv_data("D:\\NLP\MyProject\\CleanL-OSACT2020-sharedTask-train.csv")
#test_data, test_labels = load_csv_data("D:\\NLP\\MyProject\\CleanL-HSAB-AbusHateTrain.csv")
test_data, test_labels = load_csv_data("D:\\NLP\\HateSpeechDetection\\DataSet\\new1.csv")
train_data, train_labels = load_csv_data("D:\\NLP\\HateSpeechDetection\\Updataset\\thesis-trainE.csv")

x_train = np.asarray(train_data)
x_test = np.asarray(test_data)

text_tfidf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ])
#text_tfidf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3),max_features=250,min_df=5, max_df=0.7  )),('tfidf', TfidfTransformer()), ])
text_tfidf.fit(x_train) 
with open("Vector", 'wb') as f:
    pickle.dump(text_tfidf, f)
with open("Vector", 'rb') as f:
    text_clf=pickle.load(f)

vect= text_clf.transform(x_test)
print(vect)
print(x_test)
X = text_clf.transform(x_test).toarray()
print(X.shape)

#vectorizer = TfidfVectorizer(ngram_range=(1,3),max_features=250,min_df=5, max_df=0.7)
vectorizer = TfidfVectorizer()
Xtrain = vectorizer.fit_transform(x_train)
print(vectorizer.get_feature_names())
vect= vectorizer.transform(x_test)
print(vect)
print(x_test)
X = vectorizer.transform(x_test).toarray()
print(X)
print(X.shape)