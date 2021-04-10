# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:07:00 2017

@author: Medyan
"""

#!/usr/bin/env python
# encoding: utf-8 
    
import medyan
import sqlite3

from collections import defaultdict
from collections import Counter
from nltk import bigrams 
import math
import operator

conn = sqlite3.connect('test.db')
tweets = medyan.get_data(conn)
#print(medyan.get_data_count(conn))

count_alls = Counter()
count_stop = Counter()
for tweet in tweets:
    terms_alls = [term for term in medyan.preprocess(tweet[4])]
    terms_stop = [term for term in medyan.preprocess(tweet[4]) if term not in medyan.stop and not term.startswith(('#', '@'))] 
    terms_hash = [term for term in medyan.preprocess(tweet[4]) if term.startswith(('#', '@'))] 

    count_alls.update(terms_alls)
    count_stop.update(terms_stop)
    terms_bigram = bigrams(terms_stop)


# n_docs is the total n. of tweets
p_t = {}
p_t_com = defaultdict(lambda : defaultdict(int))
n_docs = 1000
com = medyan.get_co_occurrence_matrix(tweets)
for term, n in count_stop.items():
    p_t[term] = n / n_docs
    for t2 in com[term]:
        p_t_com[term][t2] = com[term][t2] / n_docs
        
        
pmi = defaultdict(lambda : defaultdict(int))
for t1 in p_t:
    for t2 in com[t1]:
        denom = p_t[t1] * p_t[t2]
        pmi[t1][t2] = math.log2(p_t_com[t1][t2] / denom)

semantic_orientation = {}
for term, n in p_t.items():
    positive_assoc = sum(pmi[term][tx] for tx in medyan.get_positive_vocab())
    negative_assoc = sum(pmi[term][tx] for tx in medyan.get_negative_vocab())
    semantic_orientation[term] = positive_assoc - negative_assoc

semantic_sorted = sorted(semantic_orientation.items(), 
                         key=operator.itemgetter(1), 
                         reverse=True)
top_pos = semantic_sorted[:10]
top_neg = semantic_sorted[-10:]

print(top_pos)
print(top_neg)
print("ITA v WAL: %f" % semantic_orientation['#itavwal'])
print("SCO v IRE: %f" % semantic_orientation['#scovire'])
print("ENG v FRA: %f" % semantic_orientation['#engvfra'])
print("#ITA: %f" % semantic_orientation['#ita'])
print("#FRA: %f" % semantic_orientation['#fra'])
print("#SCO: %f" % semantic_orientation['#sco'])
print("#ENG: %f" % semantic_orientation['#eng'])
print("#WAL: %f" % semantic_orientation['#wal'])
print("#IRE: %f" % semantic_orientation['#ire'])

      
      