# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 10:17:32 2020

@author: Medyan
"""

# coding: utf-8

"""
process-data.py is a group of functions which 
        deal with sqlite database and 
        make some manipulation for text
Author: Medyan AbdelHamid
Date: Dec. 2020
"""
import sqlite3
import numpy as np
import string
import re
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
import keras

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)
db_file = 'fdata-2020.db'

def remove_url(text):
    import re
    return re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text) # no emoji

def preprocess_tweet(text):
    from pyarabic.araby import strip_tashkeel, strip_tatweel, tokenize, is_arabicrange
    text = remove_url(text)
    text = strip_all_entities(text)
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    tokens = tokenize(text, conditions=is_arabicrange, morphs=strip_tashkeel)
    tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return ' '.join(tokens)

def reverse_tweet(text):
    # split the string on space
    words = text.split()
    # reverse the words using reversed() function
    words = list(reversed(words))
    # join the words and return
    return ' '.join(words)

def remove_last_word(text):
    return text.rsplit(' ', 1)[0]

def remove_first_word(text):
    return text.split(' ', 1)[1]

def get_test_seq():
    conn = sqlite3.connect(db_file)
    sql = 'SELECT seq FROM SQLITE_SEQUENCE where name=\'TESTS\''
    cur = conn.cursor()
    cur.execute(sql)
    last = cur.fetchone()[0] + 1
    conn.close()
    return last

def add_test_to_database(results, notes):
    conn = sqlite3.connect(db_file)
    sql = "INSERT INTO TESTS (NB, RESULTS, NOTES)  VALUES (?, ?, ?)"
    args = (get_test_seq(), results, notes,)
    conn.execute(sql, args)
    conn.commit()

def normalize_tweets():
    conn = sqlite3.connect(db_file)
    sql = "SELECT * FROM INVERSE "
    cur = conn.cursor()
    cur.execute(sql)
    records = cur.fetchall()
    for rec in records:
        # print(rec[0])
        # print(rec[4])
        # print(preprocess_tweet(rec[4]))
        # exit
        sql = "UPDATE INVERSE SET FULLTEXT = ? WHERE NB = ?"
        args = (preprocess_tweet(rec[1]),rec[2],)
        conn.execute(sql, args)
    conn.commit()

def augment_hate_tweets():
    conn = sqlite3.connect(db_file)
    sql = "SELECT NB, NORM_TEXT FROM TWEETS_M WHERE ANOMALY = 1 "
    cur = conn.cursor()
    cur.execute(sql)
    records = cur.fetchall()
    for rec in records:
        # print(rec[0])
        # print(rec[4])
        # print(preprocess_tweet(rec[4]))
        # exit
        sql = "INSERT INTO TWEETS_M (ANOMALY, FULLTEXT, NORM_TEXT) VALUES (1,?,?) "
        # rev = reverse_tweet(rec[1])
        rev = remove_last_word(rec[1])
        args = (rev, rev)
        conn.execute(sql, args)
    conn.commit()

def get_vec(n_model,dim, token):
    vec = np.zeros(dim)
    if token not in n_model.wv:
        _count = 0
        for w in token.split("_"):
            if w in n_model.wv:
                _count += 1
                vec += n_model.wv[w]
        if _count > 0:
            vec = vec / _count
    else:
        vec = n_model.wv[token]
    return vec

# classifiers to use
theClassifierss = [
    # RandomForestClassifier(n_estimators=300),
    RandomForestClassifier(n_estimators=2000, max_depth=40, min_samples_split=15, min_samples_leaf=4, max_features='sqrt', bootstrap=False ),
    # SGDClassifier(random_state=42),#(loss='log', penalty='l1'),
    # MLPClassifier(hidden_layer_sizes=(100,100,100, 100), activation='relu', solver='adam', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(150,100,100), activation='relu', solver='adam', alpha=0.05, learning_rate = 'invscaling', max_iter=1000),
    # SGDClassifier(alpha=1e-05, max_iter=5000, n_jobs=-1, loss='log', penalty='elasticnet'),
    # LinearSVC(C=1e1),
    # LinearSVC(max_iter=5000, C=1000, loss='hinge', tol=1e0),
    # SVC(kernel='linear', class_weight={1: 1000}),
    # SVC(kernel='poly', probability=True, degree=3, gamma=100, C=0.2, class_weight={0: 1, 1: 500}),
    SVC(kernel='poly', probability=True, class_weight={0: 1, 1: 10}),
    # XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5),
    # XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5, base_score=0.7, booster='gbtree', colsample_bylevel=1, gamma=0, subsample=0.5, tree_method='gpu_hist', predictor='gpu_predictor'),
    # XGBClassifier(learning_rate = 1e-3 , n_estimators=1500, max_depth=5, min_child_weight=1, colsample_bytree=0.85, subsample=0.65, scale_pos_weight = 1, reg_alpha=1.0, predictor='auto', tree_method='gpu_hist', verbosity=0), 
    XGBClassifier(learning_rate = 1e-3 , n_estimators=5000, max_depth=3, min_child_weight=2, colsample_bytree=0.70, subsample=0.85, scale_pos_weight = 1, reg_alpha=1.7, predictor='auto', tree_method='gpu_hist', verbosity=0, use_label_encoder=False),
    # CatBoostClassifier(iterations=500,learning_rate=0.05,depth=6,loss_function='Logloss',border_count=32 ),
    # CatBoostClassifier(iterations=1000,learning_rate=0.01,depth=7,loss_function='Logloss',border_count=32 ),
    CatBoostClassifier(iterations=2000,learning_rate=5e-3,depth=6, logging_level='Silent'),
    # NuSVC(),
    # LogisticRegressionCV(solver='liblinear', n_jobs=-1),
    # LogisticRegression(n_jobs=-1, C=1, solver='saga'),
    # LogisticRegression(),
    # GaussianNB(),
]
# classifiers to use
theClassifiersLevel2 = [
    MLPClassifier(hidden_layer_sizes=(150, 100)),
    LogisticRegressionCV(solver='liblinear', n_jobs=-1),
    LogisticRegression(n_jobs=-1, C=1, solver='saga'),
]
hard_voting_cls = VotingClassifier(
    estimators=[
        ('RandomForestClassifier', theClassifierss[0]),
        # ('SGDClassifier', theClassifierss[1]),
        ('SVC', theClassifierss[2]),
        ('XGBClassifier', theClassifierss[3]),
        ('CatBoostClassifier', theClassifierss[4]),
        # ('LogisticRegressionCV', theClassifierss[5]),
        # ('LogisticRegression', theClassifierss[6]),
        ('MLPClassifier', theClassifierss[1]),
        # ('GaussianNB', theClassifierss[8])
                ],
    voting='hard')

soft_voting_cls = VotingClassifier(
    estimators=[
        ('RandomForestClassifier', theClassifierss[0]),
        # ('SGDClassifier', theClassifierss[1]),
        ('SVC', theClassifierss[2]),
        ('XGBClassifier', theClassifierss[3]),
        ('CatBoostClassifier', theClassifierss[4]),
        # ('LogisticRegressionCV', theClassifierss[5]),
        # ('LogisticRegression', theClassifierss[6]),
        ('MLPClassifier', theClassifierss[1]),
        # ('GaussianNB', theClassifierss[8])
        ],
    voting='soft')

# theClassifiers = theClassifierss
theClassifiers = theClassifierss + [soft_voting_cls, hard_voting_cls]
# theclassifiers_params = ' '.join(cls.__class__.__name__ + ": " + json.dumps(cls.get_params()) for cls in theClassifiers)

mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), activation='relu', solver='adam', max_iter=1000)

# Create layers (Functional API)
knn = keras.Sequential([
                keras.layers.Dense(128,),                          
                keras.layers.Dense(2,)])
knn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifiers = [hard_voting_cls, mlp]
classifiers_params = ['HVotingClassifier', 'mlp']
