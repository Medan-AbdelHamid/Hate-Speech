# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 00:52:16 2017

@author: Medyan
"""

import csv
import tweepy

#Twitter API credentials
consumer_key = "uTjRLynXDBv4qt0e5ilbfc4QE"
consumer_secret = "QIpL4HiapJ21C7dMLY9ozMAsEwb2IWIc933PvzW9BBva7GpklU"
access_key = "902812596289626112-GaRIFy1FfFPtDkxESzf7bjzPaH8dnO4"
access_secret = "ISZmScghDtvnvfYKGH3EDkJ6UKQZeFX9DTvt1WwrrW74i"


#use variables to access twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

#create an object called 'customStreamListener'

class CustomStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        with open('file.txt', 'w') as f: 
            f.write('Author,Date,Text')
            writer = csv.writer(f)
            writer.writerow([status.author.screen_name, status.created_at])

    def on_error(self, status_code):
        print('Encountered error with status code:', status_code)
        return True # Don't kill the stream

    def on_timeout(self):
        print('Timeout...')
        return True # Don't kill the stream


streamingAPI = tweepy.streaming.Stream(auth, CustomStreamListener())
streamingAPI.filter(track=["love"])
