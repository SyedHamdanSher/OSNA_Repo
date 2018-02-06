"""
classify.py
"""

from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request
import pickle


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    #https://docs.python.org/2/library/re.html
    #W == [^a-zA-Z0-9_]
    if(keep_internal_punct):
        return np.array([re.sub('\W+$', '',re.sub('^\W+', '', x))for x in doc.lower().split()])
    else:
        return np.array(re.sub('\W+', ' ',doc.lower()).split())


def token_features(tokens, feats):
    for token in tokens:
        token='token='+token
        feats[token]+=1


def token_pair_features(tokens, feats, k=3):
	
	def token_pair_features1(window,feats):
		for i in range(0,len(window)):
			for j in range(i+1,len(window)):
				token="token_pair="+window[i]+"__"+window[j]
				feats[token]+=1

	window = []
	for i in range(0,len(tokens)):
		j=i
		c=0
		if(i+k-1<len(tokens)):
			while(c<k):
				window.append(tokens[j])
				c+=1
				j+=1
				#print(window)
		else:
			break
		#print(window)
		token_pair_features1(window,feats)
		window.clear()

def lexicon_features(tokens, feats):
    fp = open('pos.txt', 'rb')
    pos_words = pickle.load(fp)
    fp.close()
    
    fp = open('neg.txt', 'rb')
    neg_words = pickle.load(fp)
    fp.close()

    feats['pos_words']=0
    feats['neg_words']=0
    for token in tokens:
        if token.lower() in pos_words:
            feats['pos_words']+=1
        elif token.lower() in neg_words:
            feats['neg_words']+=1


def featurize(tokens, feature_fns):
    feats=defaultdict(lambda: 0)
    for func in feature_fns:
        func(tokens,feats)
    
    feats1=sorted(feats.items(),key=lambda x:(x[0]))
    return feats1

def vectorize(tokens_list, feature_fns, min_freq=2, vocab=None):
    features=[]
    
    feats=defaultdict(lambda: 0)   
    for token in tokens_list:
        feats=featurize(token,feature_fns) 
        #print(dict(feats))
        features.append(dict(feats)) 
    #print(features)
    
    if(vocab==None):
        
        freq=defaultdict(lambda: 0)
        ll=[]
        DD=defaultdict(lambda: 0)
        for dic in features:
            #print(dic)
            for key,value in dic.items():
                if dic[key]>0:
                    freq[key]=freq[key]+1
                if (key not in DD) and (freq[key]>=min_freq):
                    ll.append(key)
                    DD[key]=0
                    
        ll=sorted(ll)
        c=0
        vocab=defaultdict(lambda: 0)
        for key in ll:
                vocab[key]=c
                c+=1
     
    rows=[]
    column=[]
    data=[]
    i=0
    for dic in features:
        for key,value in dic.items():
            if key in vocab:
                rows.append(i)
                column.append(vocab[key])
                data.append(value)
        i+=1
    
    X=csr_matrix((np.array(data,dtype='int64'), (np.array(rows,dtype='int64'),np.array(column,dtype='int64'))), shape=(i, len(vocab)))
    return X,vocab

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)

def cross_validation_accuracy(clf, X, labels, k):
    #http://lijiancheng0614.github.io/scikit-learn/modules/generated/sklearn.cross_validation.KFold.html
    accuracies=[]
    cv=KFold(len(labels),k)
    
    for train_index,test_index in cv:
        clf.fit(X[train_index],labels[train_index])
        predict = clf.predict(X[test_index])
        accuracy = accuracy_score(labels[test_index], predict)
        accuracies.append(accuracy)
    
    return np.mean(accuracies)


def eval_all_combinations(docs, labels, punct_vals,feature_fns, min_freqs):
	
    ll = []
    for punc in punct_vals:
        tokens=[]
        for doc in docs:
            tokens.append(tokenize(doc,punc))
        for freq in min_freqs:
            for x in range(1, len(feature_fns)+1):
                #https://docs.python.org/3/library/itertools.html#itertools.combinations
                for feature in combinations(feature_fns,x):
                    #print(feature)
                    dic={}
                    features=list(feature)
                    X,vocab = vectorize(tokens,features,freq)
                    
                    accuracy = cross_validation_accuracy(LogisticRegression(), X, labels, 5)
                    #print(accuracy)
                    dic['features']=feature
                    dic['punct']=punc
                    dic['accuracy']=accuracy
                    dic['min_freq']=freq
                    
                    ll.append(dic)
                    
    return sorted(ll,key=lambda x: (-x['accuracy'],-x['min_freq']))


def fit_best_classifier(docs, labels, feature_fns):
    tokensList = [tokenize(doc) for doc in docs]
    X,vocab=vectorize(tokensList,feature_fns)
    
    clf = LogisticRegression()
    clf.fit(X,labels)
    return clf,vocab
  

def parse_test_data(feature_fns, vocab,tweets):
    tokensList = [ tokenize(d) for d in tweets ]
    X_test,vocb=vectorize(tokensList,feature_fns,2,vocab)
    return X_test

def writeToFile(filename,classification):
        fp = open(filename, 'wb')
        pickle.dump(classification, fp)
        fp.close()

def print_top_prediction(X_test, clf, tweets):
    
    prediction=clf.predict(X_test)
    prediction1=prediction
    otweets=tweets[:10]
    #printing top 10 predicted tweets
    for t in zip(prediction1,otweets):
        if(t[0] == 0):
            print("Negative Tweet: "+t[1].encode('utf8').decode('unicode_escape').encode('ascii','ignore').decode("utf-8"))
        
        elif(t[0]==1):
             print("Positive Tweet: "+t[1].encode('utf8').decode('unicode_escape').encode('ascii','ignore').decode("utf-8"))
        print("\n===========================================================================\n")
    
    pos=[]
    neg=[]
    posCount=0
    negCount=0
    
    for t in zip(prediction,tweets):
        if t[0]==0:
            neg.append(t[1])
            negCount+=1
        
        elif t[0]==1:
            pos.append(t[1])
            posCount+=1
    
    classification={}        
    classification['posCount']=posCount
    classification['negCount']=negCount 
    classification['pos']=pos
    classification['neg']=neg

    writeToFile('classification.pkl',classification)
    
    


def readFromFile(filename):
    fp = open(filename, 'rb')
    tweets = pickle.load(fp)
    return tweets

def main():
    feature_fns = [token_pair_features, lexicon_features]
    docs, labels = read_data(os.path.join('data', 'train'))
    fname='tweets.pkl'
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    clf, vocab = fit_best_classifier(docs, labels,feature_fns)
    tweets=readFromFile(fname)
    uniquetweets = set()
    for t in tweets:
        uniquetweets.add(t['text'])
    uniquetweets=list(uniquetweets)
    X_test = parse_test_data(feature_fns, vocab,uniquetweets)
    print('\nPrinting Classified top 10 Tweets based on negative and positive sentiments:\n')
    print_top_prediction(X_test, clf,list(uniquetweets))


if __name__ == '__main__':
    main()
