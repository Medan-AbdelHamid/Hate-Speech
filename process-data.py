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
db_file = 'C:/Users/ASUS/Documents/Thesis/Code/Python/SoureCode/fdata-2020.db'

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
    sql = "INSERT INTO TESTS (RESULTS, NOTES)  VALUES (?, ?)"
    args = (results, notes,)
    conn.execute(sql, args)
    conn.commit()

def normalize_tweets():
    conn = sqlite3.connect(db_file)
    sql = "SELECT * FROM TRAINSET "
    cur = conn.cursor()
    cur.execute(sql)
    records = cur.fetchall()
    for rec in records:
        # print(rec[0])
        # print(rec[4])
        # print(preprocess_tweet(rec[4]))
        # exit
        sql = "UPDATE TRAINSET SET FULLTEXT = ? WHERE NB = ?"
        args = (preprocess_tweet(rec[4]),rec[0],)
        conn.execute(sql, args)
    conn.commit()

def augment_hate_tweets():
    conn = sqlite3.connect(db_file)
    sql = "SELECT NB, FULLTEXT FROM TRAINSET WHERE ANOMALY = 1 "
    cur = conn.cursor()
    cur.execute(sql)
    records = cur.fetchall()
    for rec in records:
        # print(rec[0])
        # print(rec[4])
        # print(preprocess_tweet(rec[4]))
        # exit
        sql = "INSERT INTO TRAINSET (ANOMALY, FULLTEXT, FULLTEXT) VALUES (1,?,?) "
        rev1 = reverse_tweet(rec[1])
        rev2 = remove_last_word(rec[1])
        rev3 = remove_first_word(rec[1])
        args = (rev1, rev1)
        conn.execute(sql, args)
        args = (rev2, rev2)
        conn.execute(sql, args)
        args = (rev3, rev3)
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

# normalize all tweets
# normalize_tweets()
# augment hate tweets
augment_hate_tweets()