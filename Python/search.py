#!/usr/bin/env python
# encoding: utf-8

import tweepy
import json
import time

#Twitter API credentials
consumer_key = "x7XOywj3rgsSKn9MXkBNBpecr"
consumer_secret = "o4lHJPQQyVGQaojTGPTHR9BOXuEzvRzbghBkPMrAyaBoPRQjvd"
access_key = "3994061302-XFnBADeFR4WBxZzmXAy4sXn2sSHu6aYldjZgbVk"
access_secret = "g4FCmePHacsvv7nJOp2sJRaY9nbnH556s97whuOKILFrI"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
#refer http://docs.tweepy.org/en/v3.2.0/api.html#API
#tells tweepy.API to automatically wait for rate limits to replenish

#Put your search term
searchquery = "data protection"

users =tweepy.Cursor(api.search,q=searchquery).items()
count = 0
errorCount=0

file = open('search.json', 'wb') 

while True:
    try:
        user = next(users)
        count += 1
        #use count-break during dev to avoid twitter restrictions
        #if (count>10):
        #    break
    except tweepy.TweepError:
        #catches TweepError when rate limiting occurs, sleeps, then restarts.
        #nominally 15 minnutes, make a bit longer to avoid attention.
        print "sleeping...."
        time.sleep(60*16)
        user = next(users)
    except StopIteration:
        break
    try:
        print "Writing to JSON tweet number:"+str(count)
        json.dump(user._json,file,sort_keys = True,indent = 4)
        
    except UnicodeEncodeError:
        errorCount += 1
        print "UnicodeEncodeError,errorCount ="+str(errorCount)

print "completed, errorCount ="+str(errorCount)+" total tweets="+str(count)
    
    #todo: write users to file, search users for interests, locations etc.

"""
http://docs.tweepy.org/en/v3.5.0/api.html?highlight=tweeperror#TweepError
NB: RateLimitError inherits TweepError.
http://docs.tweepy.org/en/v3.2.0/api.html#API  wait_on_rate_limit & wait_on_rate_limit_notify
NB: possibly makes the sleep redundant but leave until verified.

"""

