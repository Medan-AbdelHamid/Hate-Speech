#!/usr/bin/env python
# encoding: utf-8

import tweepy
import json
import time
import csv

#Twitter API credentials
consumer_key = "uTjRLynXDBv4qt0e5ilbfc4QE"
consumer_secret = "QIpL4HiapJ21C7dMLY9ozMAsEwb2IWIc933PvzW9BBva7GpklU"
access_key = "902812596289626112-GaRIFy1FfFPtDkxESzf7bjzPaH8dnO4"
access_secret = "ISZmScghDtvnvfYKGH3EDkJ6UKQZeFX9DTvt1WwrrW74i"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
#refer http://docs.tweepy.org/en/v3.2.0/api.html#API
#tells tweepy.API to automatically wait for rate limits to replenish

#Put your search term
searchquery = "language=ar"

users =tweepy.Cursor(api.search,q=searchquery).items()
count = 0
errorCount=0

file = open('search.json', 'w') 

while True:
    try:
        user = next(users)
        count += 1
        #use count-break during dev to avoid twitter restrictions
        if (count>1):
            break
    except tweepy.TweepError:
        #catches TweepError when rate limiting occurs, sleeps, then restarts.
        #nominally 15 minnutes, make a bit longer to avoid attention.
        print("sleeping....")
        time.sleep(60*16)
        user = next(users)
    except StopIteration:
        break
    try:
        print("Writing to JSON tweet number:"+str(count))
        with open('file.txt', 'w') as f: 
            f.write('Author,Date,Text')
            writer = csv.writer(f)
            writer.writerow([user.author.screen_name, user.created_at, user.text])
        
    except UnicodeEncodeError:
        errorCount += 1
        print("UnicodeEncodeError,errorCount ="+str(errorCount))

file.close()
print("completed, errorCount ="+str(errorCount)+" total tweets="+str(count))

#with open('search.json') as json_data:
#    data = json.load(json_data)
#    json_data.close()
#
#    print("json value: ", str(data["text"]))    
    #todo: write users to file, search users for interests, locations etc.

"""
http://docs.tweepy.org/en/v3.5.0/api.html?highlight=tweeperror#TweepError
NB: RateLimitError inherits TweepError.
http://docs.tweepy.org/en/v3.2.0/api.html#API  wait_on_rate_limit & wait_on_rate_limit_notify
NB: possibly makes the sleep redundant but leave until verified.

"""

