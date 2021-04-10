#!/usr/bin/env python
# encoding: utf-8

import tweepy
import json

# Authentication details. To  obtain these visit dev.twitter.com
access_token = "your access token goes here"
access_token_secret = "your access token secret goes here"
consumer_key = "your consumer key goes here"
consumer_secret = "your consumer key  secret goes here"

# This is the listener, resposible for receiving data
class StdOutListener(tweepy.StreamListener):
    def on_data(self, data):
        # Parsing 
        decoded = json.loads(data)
        #open a file to store the status objects
        file = open('stream.json', 'wb')  
        #write json to file
        json.dump(decoded,file,sort_keys = True,indent = 4)
        #show progress
        print "Writing tweets to file,CTRL+C to terminate the program"

        
        return True

    def on_error(self, status):
        print status

if __name__ == '__main__':
    l = StdOutListener()
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # There are different kinds of streams: public stream, user stream, multi-user streams
    # For more details refer to https://dev.twitter.com/docs/streaming-apis
    stream = tweepy.Stream(auth, l)
    #Hashtag to stream
    stream.filter(track=["#love"])