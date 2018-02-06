"""
sumarize.py
"""


import datetime
import pickle
import sys
import time
from TwitterAPI import TwitterAPI
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from collections import Counter, defaultdict


    
def main():
    fp = open('summary.txt','w')
    filename='tweets.pkl'
    fp1 = open(filename, 'rb')
    tweets = pickle.load(fp1)
    u = set([t['user']['screen_name'] for t in tweets])

    fp.write("Number of users collected:%d\n"%len(u))
    fp.write("Number of messages collected:%d\n"%len(tweets))
    
    filename1='communities.pkl'
    fp1 = open(filename1, 'rb')
    c = pickle.load(fp1)
    fp.write("Number of communities discovered:%d\n"%len(c))
    
    numberOfCommunities=len(c)
    communitiesCount = dict(Counter([len(c1) for c1 in c]))
    
    fp.write('Average number of users per community:'+ str((sum((comm[0]*comm[1]) for comm in communitiesCount.items())/numberOfCommunities)))  
    
    filename2='classification.pkl'
    fp2 = open(filename2, 'rb')
    classification = pickle.load(fp2)
    
    fp.write("\nNumber of instances per class found:\n")
    fp.write("\nNumber of positive instances per class: %d"%classification['posCount'])
    fp.write("\nNumber of negative instances per class: %d"%classification['negCount'])
    if(classification['negCount']>classification['posCount']):
    	fp.write("\nAccording to the analysis done on tweets collected more number of people didn't like DEMONITIZATION\n")
    else:
    	fp.write("\nAccording to the analysis done on tweets collected more number of people did like DEMONITIZATION\n")
    
    fp.write("\nOne example from each class:\n")
    fp.write("\nPositive Class Example:\n%s"%classification['pos'][1].encode('utf8').decode('unicode_escape').encode('ascii','ignore').decode("utf-8"))
    fp.write("\nNegative Class Example:\n%s"%classification['neg'][1].encode('utf8').decode('unicode_escape').encode('ascii','ignore').decode("utf-8"))
    
    
    fp.close()
    fp1.close()
    fp2.close()
    
if __name__ == '__main__':
    main()
