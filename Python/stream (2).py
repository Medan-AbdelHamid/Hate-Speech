#!/usr/bin/env python
# encoding: utf-8

import tweepy
import json

#Twitter API credentials
consumer_key = "uTjRLynXDBv4qt0e5ilbfc4QE"
consumer_secret = "QIpL4HiapJ21C7dMLY9ozMAsEwb2IWIc933PvzW9BBva7GpklU"
access_key = "902812596289626112-GaRIFy1FfFPtDkxESzf7bjzPaH8dnO4"
access_secret = "ISZmScghDtvnvfYKGH3EDkJ6UKQZeFX9DTvt1WwrrW74i"

# This is the listener, resposible for receiving data
class StdOutListener(tweepy.StreamListener):
    def on_data(self, data):
        # Parsing 
        decoded = json.loads(data)
        #open a file to store the status objects
        file = open('stream.json', 'w')  
        #write json to file
        json.dump(decoded,file,sort_keys = True,indent = 4)
        #show progress
        print("Writing tweets to file,CTRL+C to terminate the program")        
        return True

    def on_timeout(self):
        print('Timeout...')
        return True # Don't kill the stream

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    l = StdOutListener()
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)

    # There are different kinds of streams: public stream, user stream, multi-user streams
    # For more details refer to https://dev.twitter.com/docs/streaming-apis
    stream = tweepy.Stream(auth, l)
    #Hashtag to stream
    stream.filter(track=["#love"])