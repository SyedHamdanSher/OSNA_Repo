# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/
# Note that I have not provided many doctests for this one. I strongly
# recommend that you write your own for each function to ensure your
# implementation is correct.

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    flag=[]
    for index,row in movies.iterrows():
        flag.append(tokenize_string(row.genres))
    #print(flag)
    movies['tokens']=flag
    #print(movies)
    return movies


def featurize(movies):
    vocabList=set()
    vocab=defaultdict(lambda: 0)
    
    tfidf=defaultdict()
    df1=defaultdict(lambda: 0)
    tokenSet=set()
    for tokens in movies['tokens']:
        for token in set(tokens):
            tokenSet.add(token)
            df1[token]+=1
            
    sortedTokens=sorted(list(tokenSet))
    vocab=defaultdict(lambda: 0)
    counter=0
    
    #print(sortedTokens)
    for k in sortedTokens:
        vocab[k]=counter
        counter+=1
    #print(tokenDict)
    #print("==================================================")
    #print(movies)
    
    tfdic=defaultdict()
    for index,row in movies.iterrows():
         tfdic[row.movieId]=dict(Counter(row.tokens))
    #print(tfdic)
    
    max_k=defaultdict()
    for k,v in tfdic.items():
        max_k[k]= max(v.values())
    #print(max_k)
    
    N=len(movies)
    csrmatrix1=[]
    #print(df1)
    for index,row in movies.iterrows():
        value={}
        for token in row.tokens:
            dic=tfdic[row.movieId]
            val=dic[token]/max_k[row.movieId] * math.log10(N/df1[token])
            value[token]=val
        tfidf[row.movieId]=value
        dic1=tfidf[row.movieId]
        column=[]
        data=[]
        rows=[]
        for token1 in row.tokens:
            column.append(vocab[token1])
            rows.append(0)
            data.append(dic1[token1])
        csrmatrix1.append(csr_matrix((np.array(data), (np.array(rows),np.array(column))), shape=(1, len(vocab))))
    #print(csrmatrix1)
    movies['features']=csrmatrix1
    
    return movies,vocab


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    return sum([i*j for (i, j) in zip(a.toarray()[0], b.toarray()[0])])/(math.sqrt(sum([i*i for i in  a.toarray()[0]]))*math.sqrt(sum([i*i for i in  b.toarray()[0]])))


def make_predictions(movies, ratings_train, ratings_test):
    res=[]
    #print(ratings_train,ratings_test)
    for index,row in ratings_test.iterrows():
        features=movies[movies.movieId==row.movieId].iloc[0]['features']
        #print(features)
        flag=False
        
        div=0
        wsum=0.0
        userRatings=ratings_train[ratings_train.userId==row.userId]
        
        for index1,row1 in userRatings.iterrows():
            features1=movies[movies.movieId==row1.movieId].iloc[0]['features']
            val=cosine_sim(features1,features)
            #print("sss")
            
            if(val>0):
                div=div+val
                wsum=wsum+(val*row1.rating)
                flag=True
        
        if(flag):
            res.append(wsum/div)
        if(not flag):
            res.append(np.mean(userRatings.rating))       
    #print (res)
    return np.array(res)


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
