# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 21:55:57 2017

@author: Medyan
"""

#import tweepy as tw
import tweepy
from tweepy.auth import OAuthHandler

#Twitter API credentials
consumer_key = "uTjRLynXDBv4qt0e5ilbfc4QE"
consumer_secret = "QIpL4HiapJ21C7dMLY9ozMAsEwb2IWIc933PvzW9BBva7GpklU"
access_key = "902812596289626112-GaRIFy1FfFPtDkxESzf7bjzPaH8dnO4"
access_secret = "ISZmScghDtvnvfYKGH3EDkJ6UKQZeFX9DTvt1WwrrW74i"

#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)

    def on_error(self, status_code):
        if status_code == 420:
            return False


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener())


myStream.filter(track=['python'])

