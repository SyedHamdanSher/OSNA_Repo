# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
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


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


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


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
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


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
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


def plot_sorted_accuracies(results):
    accuracies=[]
    for dic in results:
        accuracies.append(dic['accuracy'])
        
    accuracies=sorted(accuracies)
    plt.plot(range(42), accuracies,'b-')
    plt.xlabel('Setting')
    plt.ylabel('Accuracy')
    plt.savefig('accuracies.png')


def mean_accuracy_per_setting(results):
    trueAccuracies=[]  
    falseAccuracies=[] 
    meanAccuracies=[]
    
    for dic in results:
        if(dic['punct']):
            trueAccuracies.append(dic['accuracy'])
        if(not dic['punct']):
            falseAccuracies.append(dic['accuracy'])
    
    tup=tuple((np.mean(trueAccuracies),'punct=True'))
    meanAccuracies.append(tup)
    tup=tuple((np.mean(falseAccuracies),'punct=False'))
    meanAccuracies.append(tup)
    
    accuracies2=[]  
    accuracies5=[]
    accuracies10=[]  
    
    for dic in results:
        if(dic['min_freq']==2):
             accuracies2.append(dic['accuracy'])
        elif(dic['min_freq']==5):
            accuracies5.append(dic['accuracy'])
        elif(dic['min_freq']==10):
            accuracies10.append(dic['accuracy'])
    
    tup=tuple((np.mean(accuracies2),'min_freq=2'))
    meanAccuracies.append(tup)
    
    tup=tuple((np.mean(accuracies5),'min_freq=5'))
    meanAccuracies.append(tup)
    
    tup=tuple((np.mean(accuracies10),'min_freq=10'))
    meanAccuracies.append(tup)
    
    for features in set([doc['features'] for doc in results]):
        mean=[]
        for x in results:
            if x['features']==features:
                mean.append(x['accuracy'])
        feature='features='+' '.join([f.__name__ for f in list(features)])
        meanAccuracies.append((np.mean(mean),feature))
    
    return sorted(meanAccuracies, key=lambda x: (-x[0]))


def fit_best_classifier(docs, labels, best_result):
    tokens = [tokenize(doc,best_result['punct']) for doc in docs]
    X,vocab=vectorize(tokens,best_result['features'],best_result['min_freq'])
    
    clf = LogisticRegression()
    clf.fit(X,labels)
    return clf,vocab


def top_coefs(clf, label, n, vocab):
    #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    coef = clf.coef_[0]
    #print(coef)
    if(label==0):
        highIndex = np.argsort(coef)[:n]
    if(label==1):
        highIndex = np.argsort(coef)[::-1][:n]
    
    highTerms = np.array([k for k,v in sorted(vocab.items(), key=lambda x: x[1])])[highIndex]
    bestCoef = coef[highIndex]
    
    if(label==0):
        neg=[]
        for f in zip(highTerms, bestCoef*-1):
            neg.append(f)
        return neg
    if(label==1):
        ll=[]
        #https://www.saltycrane.com/blog/2008/04/how-to-use-pythons-enumerate-and-zip-to/
        for f_c in zip(highTerms, bestCoef):
            #print(f_c)
            ll.append(f_c) 
        return ll


def parse_test_data(best_result, vocab):
    test_docs, test_labels = read_data(os.path.join('data', 'test'))
    tokens = [tokenize(x,best_result['punct']) for x in test_docs]
    X_test,vocab=vectorize(tokens,best_result['features'],best_result['min_freq'],vocab)
    
    return test_docs,test_labels,X_test


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba
    ll=[]
    predict = clf.predict(X_test)
    predictProba = clf.predict_proba(X_test)
    
    for x in range(len(predict)):
        dic = {}
        if predict[x] != test_labels[x]:
            if predict[x] == 0:
                dic['truth'] = test_labels[x]
                dic['predicted']=predict[x]
                dic['proba'] = predictProba[x][0]
                dic['test'] =test_docs[x] 
            else:
                dic['truth'] = test_labels[x]
                dic['predicted']=predict[x]
                dic['proba'] = predictProba[x][1]
                dic['test'] =test_docs[x] 
            ll.append(dic)
    ll=sorted(ll, key=lambda x: (-x['proba']))[:n]
    for l in ll:
        print('truth=%d predicted=%d proba=%.6f'%(l['truth'],l['predicted'],l['proba']))
        print(l['test']+"\n")


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
