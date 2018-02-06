"""
collect.py
"""

# Imports you'll need.
from collections import Counter,defaultdict
import matplotlib.pyplot as plt  
#%matplotlib inline
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import configparser
import datetime
import pickle

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

consumer_key = 'bZgQOA5wkMtnkKopqeWtOvOPG'
consumer_secret = 'HRysAhdQu4vMgBHBIgaWd4KMB4vLNNK1CZWwn9zIQnHTY2QGlv'
access_token = '900790963622293505-jGn5klK2exJqAMYpjEMnMIw4s7Ccyss'
access_token_secret = '3A0AWSwfbes78pja61Agz5Kczq5lD05H8QqASbaRWdZH4'


def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def writeToFile(filename,tweets):
    output = open(filename, 'wb')
    pickle.dump(tweets, output)
    output.close()

def getTweets(twitter,search):
    tweets=[]
    total=6 #collecting data on 8th,9th,10th Nov demotization effected date
    #https://docs.python.org/3/library/datetime.html
    #https://docs.python.org/3.2/library/time.html
    for i in range (3,total):
        parameter=datetime.datetime.now()-datetime.timedelta(days=i)
        parameter=parameter.strftime('%Y-%m-%d')
        request = robust_request(twitter,'search/tweets', {'q':search,'count':100,'until':parameter,'lang':'en'})
        for r in request:
            tweets.append(r)
    writeToFile('tweets.pkl',tweets)
    return tweets

def get_friends(twitter, screen_name):
    flag=[]
    request=robust_request(twitter,'friends/ids',{'screen_name':screen_name,'count':5000,'cursor':-1}) #get friend list max 5000 at times
    for r in request:
        flag.append(r)
    return sorted(flag)

def getUserFriends(twitter,tweets):
    friends=[]
    for req in tweets:
                req['user']['friends']=get_friends(twitter,req['user']['screen_name'])
                friends.append(req)
                writeToFile('data.pkl',friends)

def getAffinData():
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinnFile = zipfile.open('AFINN/AFINN-111.txt')
    afinnData = dict()

    for line in afinnFile:
        data = line.strip().split()
        if len(data) == 2:
            afinnData[data[0].decode("utf-8")] = int(data[1])
    return afinnData

neg_words = []
pos_words = []
def posneg(afinnData):
    
    pos_words=set([k for k, v in afinnData.items() if v>=0])
    output = open('pos.txt', 'wb')
    
    pickle.dump(pos_words, output)#putting positive words in pos.txt file
    output.close()
    
    neg_words=set([key for key, value in afinnData.items() if value<0])
    output = open('neg.txt', 'wb')
    
    pickle.dump(neg_words, output)#putting negative words in neg.txt file
    output.close()

def main():
    
    afinnData=getAffinData()
    posneg(afinnData)
    twitter = get_twitter()
    tweets=getTweets(twitter,'demonetization -filter:retweets')#filetering the retweets removing redundancy
    getUserFriends(twitter,tweets)

if __name__ == '__main__':
    main()