# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:15:29 2017

@author: Medyan
"""

import re
import json
import string
from collections import Counter


#Twitter API credentials
consumer_key = "uTjRLynXDBv4qt0e5ilbfc4QE"
consumer_secret = "QIpL4HiapJ21C7dMLY9ozMAsEwb2IWIc933PvzW9BBva7GpklU"
access_key = "902812596289626112-GaRIFy1FfFPtDkxESzf7bjzPaH8dnO4"
access_secret = "ISZmScghDtvnvfYKGH3EDkJ6UKQZeFX9DTvt1WwrrW74i"

#global constants
tweetsPerIteration = 100
totalTweets = 25000
maxFollowersCount = 200

#Put your search term
#places = api.geo_search(query="SYR", granularity="country")
#place_id = places[0].id
searchquery = "max_id:1338783072427380736 lang:ar geocode:33.4,38.6,500km "

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
 
punctuation = list(string.punctuation)
#stop_words = [unicode(x.strip(), 'utf-8') for x in open('arabic.txt','r').read().split('\n')]
from nltk.corpus import stopwords
stop_words = stopwords.words('arabic')
#with open('arabic.csv') as fp:
#    stop_words = [line.strip() for line in fp]

#stop = stopwords.words('arabic') + punctuation + ['rt', 'RT', 'via']
stop = stop_words + punctuation + ['rt', '...', 'RT', '،', '.', 'via']

def restart_line(string):
    import sys
    sys.stdout.write('\b'*len(string))
    sys.stdout.flush()

def print_no_newline(string):
    import sys
    sys.stdout.write(string)
    sys.stdout.flush()
    restart_line()
    

def tokenize(s):
#    from nltk.tokenize import TweetTokenizer
#    tknzr = TweetTokenizer()
    #return tknzr.tokenize(s)    
    return tokens_re.findall(s)
 


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def get_data(conn):
    cur = conn.cursor()
    data = cur.execute('SELECT * FROM TWEETS').fetchall()
#        tweets.append(row)
        #tweets.append(json.loads(row[0]))
    return data

def get_data_count(conn):
    conn.ex
    for row in conn.execute('SELECT COUNT(*) FROM TWEETS'):        
        return row[0]
    return 0

def get_only_terms(tweets):
    for tweet in tweets: 
        terms_only = [term for term in preprocess(tweet[4]) 
                      if term not in stop 
                      and not term.startswith(('#', '@'))]
    return terms_only

def get_hash(tweets):
    for tweet in tweets: 
        terms_only = [term for term in preprocess(tweet[4]) 
                      if term not in stop 
                      and term.startswith(('#', '@'))]
    return terms_only


def get_co_occurrence_matrix(tweets):
    from collections import defaultdict 
    com = defaultdict(lambda : defaultdict(int))
 
    for tweet in tweets: 
        terms_only = [term for term in preprocess(tweet[4]) 
                      if term not in stop 
                      and not term.startswith(('#', '@'))]
 
        # Build co-occurrence matrix
        for i in range(len(terms_only)-1):            
            for j in range(i+1, len(terms_only)):
                w1, w2 = sorted([terms_only[i], terms_only[j]])                
                if w1 != w2:
                    com[w1][w2] += 1
    
    return com

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

def get_most_co_occurrence(com):
    import operator
    com_max = []
    # For each term, look for the most common co-occurrent terms
    for t1 in com:
        t1_max_terms = sorted(com[t1].items(), key = operator.itemgetter(1), reverse = True)[:50]
    
        for t2, t2_count in t1_max_terms:
            com_max.append(((t1, t2), t2_count))
    # Get the most frequent co-occurrences
    
    terms_max = sorted(com_max, key = operator.itemgetter(1), reverse = True)
    return terms_max

 
def get_most_co_occurrence_for_word(tweets, search_word):
    count_search = Counter()
    for tweet in tweets:
        terms_only = [term for term in preprocess(tweet[4]) if term not in stop and not term.startswith(('#', '@'))]
        if search_word in terms_only:
            count_search.update(terms_only)
    
    return count_search

def get_positive_vocab():
    positive_vocab = ['good', 'nice', 'great', 'awesome', 'outstanding', 'fantastic', 'terrific', ':)', ':-)', 'like', 'love',]
    return positive_vocab

def get_negative_vocab():
    negative_vocab = ['bad', 'terrible', 'crap', 'useless', 'hate', ':(', ':-(',]
    return negative_vocab

