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
import datetime
import medyan

with open('keywords.csv') as fp:
    keywords = [line.strip() for line in fp]
conn = sqlite3.connect('SoureCode/fdata-2020.db')
#conn.execute("CREATE TABLE TWEETS(TWEET TEXT);")

auth = tweepy.OAuthHandler(medyan.consumer_key, medyan.consumer_secret)
auth.set_access_token(medyan.access_key, medyan.access_secret)
api = tweepy.API(auth, 
                 wait_on_rate_limit = True, 
                 wait_on_rate_limit_notify = True
                 )
#refer http://docs.tweepy.org/en/v3.2.0/api.html#API
#tells tweepy.API to automatically wait for rate limits to replenish


users = tweepy.Cursor(api.search, q = medyan.searchquery).items()

errorCount = 0
    
print("Start at: " + str(datetime.datetime.now()))
totals = 0
while totals < medyan.totalTweets:
    tweets = []
    while len(tweets) < medyan.tweetsPerIteration:
        try:
            user = next(users)
        
        except tweepy.TweepError:
            #catches TweepError when rate limiting occurs, sleeps, then restarts.
            #nominally 15 minnutes, make a bit longer to avoid attention.
            print("sleeping...." + str(datetime.datetime.now()))
            time.sleep(60*16)
            user = next(users)
        
        except StopIteration:
            print("StopIteration...." + str(datetime.datetime.now()))
            break
        
        ss = user._json['user']['followers_count'] 
        retweet_count = user._json['retweet_count']
        favorite_count = user._json['favorite_count']
        hashtags = user._json['entities']['hashtags'] 
        mentions = user._json['entities']['user_mentions'] 
    #            if(loc):
    #                okToAdd = okToAdd and (('SYR' in loc) or ('سوري' in loc))
                
    #            okToAdd = any(word in text for word in keywords) and ss <= maxFollowersCount
        okToAdd = ss <= medyan.maxFollowersCount and (len(hashtags)>0 or retweet_count>0 or favorite_count>0 or len(mentions))
        if(okToAdd):
            tweets.append(user)
            print("A new tweet is added: %5.0f " % (len(tweets)))
                       
            
                        
    for tweet in tweets:
        sql = "INSERT INTO TWEETS (TWEET)  VALUES (?)"
        args = (json.dumps(tweet._json),)
        conn.execute(sql, args)
    
    conn.commit()
     
    totals += len(tweets)
    print(totals)
    
conn.close()
print("Data saved")
print("Done")


