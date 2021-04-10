# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:07:00 2017

@author: Medyan
"""

#!/usr/bin/env python
# encoding: utf-8 
    
import medyan
import sqlite3
from collections import Counter
from nltk import bigrams 
import vincent
import pandas
 
conn = sqlite3.connect('test.db')
tweets = medyan.get_data(conn)
#print(medyan.get_data_count(conn))

count_alls = Counter()
count_stop = Counter()
for tweet in tweets:
    terms_alls = [term for term in medyan.preprocess(tweet['text'])]
    terms_stop = [term for term in medyan.preprocess(tweet['text']) if term not in medyan.stop and not term.startswith(('#', '@'))] 
    terms_hash = [term for term in medyan.preprocess(tweet['text']) if term.startswith(('#', '@'))] 

    count_alls.update(terms_alls)
    count_stop.update(terms_stop)
    terms_bigram = bigrams(terms_stop)

#print(count_alls.most_common(10))
#print(count_stop.most_common(20))
#print(list(terms_bigram))
#print(list(terms_hash))
#print(medyan.get_most_co_occurrence(medyan.get_co_occurrence_matrix(tweets))[:5])
#print(medyan.get_most_co_occurrence_for_word(tweets, 'الله').most_common(20))
 
#word_freq = medyan.get_most_co_occurrence(medyan.get_co_occurrence_matrix(tweets))[20:30]
word_freq = count_stop.most_common(20)
print(word_freq)
labels, freq = zip(*word_freq)
data = {'data': freq, 'x': labels}
bar = vincent.Bar(data, iter_idx = 'x')
bar.to_json('term_freq.json') 
#bar.to_json('term_freq.json', html_out=True, html_path='chart.html')
 
dates_ITAvWAL = []
for tweet in tweets:
    # let's focus on hashtags only at the moment
    terms_hash = [term for term in medyan.preprocess(tweet['text']) if term.startswith('@')]
    # track when the hashtag is mentioned
    if '@saadhariri' in terms_hash:
        dates_ITAvWAL.append(tweet['created_at'])
 
# a list of "1" to count the hashtags
ones = [1]*len(dates_ITAvWAL)
# the index of the series
idx = pandas.DatetimeIndex(dates_ITAvWAL)
# the actual series (at series of 1s for the moment)
ITAvWAL = pandas.Series(ones, index=idx)
 
# Resampling / bucketing
per_minute = ITAvWAL.resample('1Min').sum().fillna(0)


time_chart = vincent.Line(ITAvWAL)
time_chart.axis_titles(x='Time', y='Freq')
time_chart.to_json('time_chart.json')


# all the data together
match_data = dict(ITAvWAL=per_minute_i, SCOvIRE=per_minute_s, ENGvFRA=per_minute_e)
# we need a DataFrame, to accommodate multiple series
all_matches = pandas.DataFrame(data=match_data,
                               index=per_minute_i.index)
# Resampling as above
all_matches = all_matches.resample('1Min', how='sum').fillna(0)

# and now the plotting
time_chart = vincent.Line(all_matches[['ITAvWAL', 'SCOvIRE', 'ENGvFRA']])
time_chart.axis_titles(x='Time', y='Freq')
time_chart.legend(title='Matches')
time_chart.to_json('time_chart.json')