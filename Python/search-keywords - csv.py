# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:07:00 2017

@author: Medyan
"""

#!/usr/bin/env python
# encoding: utf-8

import tweepy
import time
import base64


with open('keywords.csv') as fp:
    keywords = [line.strip() for line in fp]

#Twitter API credentials
consumer_key = "uTjRLynXDBv4qt0e5ilbfc4QE"
consumer_secret = "QIpL4HiapJ21C7dMLY9ozMAsEwb2IWIc933PvzW9BBva7GpklU"
access_key = "902812596289626112-GaRIFy1FfFPtDkxESzf7bjzPaH8dnO4"
access_secret = "ISZmScghDtvnvfYKGH3EDkJ6UKQZeFX9DTvt1WwrrW74i"

#global constants
filesCount = 0
tweetsInOneFile = 10
maxFollowersCount = 200

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, 
                 wait_on_rate_limit=True, 
                 wait_on_rate_limit_notify=True
                 )
#refer http://docs.tweepy.org/en/v3.2.0/api.html#API
#tells tweepy.API to automatically wait for rate limits to replenish

#Put your search term
#places = api.geo_search(query="SYR", granularity="country")
#place_id = places[0].id
searchquery = "lang:ar geocode:33.4,38.6,500km "

users = tweepy.Cursor(api.search,q=searchquery).items()

fileid = 0

while True:
    count = 0
    errorCount = 0
    tweets=[]
    
    while True:
        try:
            user = next(users)
            count += 1
        except tweepy.TweepError:
            #catches TweepError when rate limiting occurs, sleeps, then restarts.
            #nominally 15 minnutes, make a bit longer to avoid attention.
            print("sleeping....")
            time.sleep(60*16)
            user = next(users)
        except StopIteration:
            print("StopIteration.")
            break
    
        try:
            ss = user._json['user']['followers_count'] 
            geo = user._json['geo']
            text = user._json['text']
            loc = user._json['user']['location']
            okToAdd = True
#            if(loc):
#                okToAdd = okToAdd and (('SYR' in loc) or ('سوري' in loc))
                
#            okToAdd = any(word in text for word in keywords) and ss <= maxFollowersCount
            okToAdd = ss <= maxFollowersCount
            if(okToAdd):
                tweets.append(user._json)
                print("A new tweet is added: %3.0f:%3.0f " % (fileid, len(tweets)))
                #use count-break during dev to avoid twitter restrictions
                if (len(tweets) == tweetsInOneFile): 
                    #we captured enough tweets
                    print(len(tweets))
                    break
                    
        except UnicodeEncodeError:
            errorCount += 1
            print("UnicodeEncodeError,errorCount ="+str(errorCount))
    
    print("completed, errorCount ="+str(errorCount)+" total tweets="+str(count)+" captured tweets="+str(len(tweets)))
    
    # pull out various data from the tweets
    tweet_id = [tweet['id'] for tweet in tweets]
    tweet_text = [tweet['text'] for tweet in tweets]
    tweet_time = [tweet['created_at'] for tweet in tweets]
    tweet_author = [tweet['user']['screen_name'] for tweet in tweets]
    tweet_author_id = [tweet['user']['id_str'] for tweet in tweets]
    tweet_language = [tweet['lang'] for tweet in tweets]
    tweet_geo = [tweet['geo'] for tweet in tweets]
    followers_count = [tweet['user']['followers_count'] for tweet in tweets]
    location = [tweet['user']['location'] for tweet in tweets]
    retweet_count = [tweet['retweet_count'] for tweet in tweets]
    place = [tweet['place'] for tweet in tweets]
    
    
    file = open('searc%03.0f.csv' % fileid, 'w')
    
    rows = zip(tweet_id, tweet_time, tweet_author, tweet_author_id, tweet_language, followers_count, retweet_count, tweet_geo, location, place, tweet_text)
    
    from csv import writer
    csv = writer(file)
    
    for row in rows:
        values = [(base64.encodebytes(value.encode('utf8')) if hasattr(value, 'encode') else value) for value in row]
#        values = [(value.encode('base64') if hasattr(value, 'encode') else value) for value in row]
        csv.writerow(values)
     
    file.close()
    print("Data saved")
    if(fileid == filesCount):
        break
    fileid += 1
    
print("Done")
