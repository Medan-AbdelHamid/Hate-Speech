# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:07:00 2017

@author: Medyan
"""

#!/usr/bin/env python
# encoding: utf-8

import tweepy
import time
import json
import sqlite3

with open('keywords.csv') as fp:
    keywords = [line.strip() for line in fp]
conn = sqlite3.connect('test.db')
#conn.execute("CREATE TABLE TWEETS(TWEET TEXT);")

#Twitter API credentials
consumer_key = "uTjRLynXDBv4qt0e5ilbfc4QE"
consumer_secret = "QIpL4HiapJ21C7dMLY9ozMAsEwb2IWIc933PvzW9BBva7GpklU"
access_key = "902812596289626112-GaRIFy1FfFPtDkxESzf7bjzPaH8dnO4"
access_secret = "ISZmScghDtvnvfYKGH3EDkJ6UKQZeFX9DTvt1WwrrW74i"

#global constants
filesCount = 0
tweetsInOneFile = 10
maxFollowersCount = 200
tweet_separator = ' [...]'

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
                tweets.append(user)
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
        
    for tweet in tweets:
        sql = "INSERT INTO TWEETS (TWEET)  VALUES (?)"
        args = (json.dumps(tweet._json),)
        conn.execute(sql, args)
        conn.commit()
     
    conn.close()
    print("Data saved")
    if(fileid == filesCount):
        break
    fileid += 1
    
print("Done")
