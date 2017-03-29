# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:21:36 2017

@author: xu.liu
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:21:09 2017
@author: xuliu
"""

import os
import pandas as pd
import re
import nltk
from nltk import pos_tag
import numpy as np
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
import xgboost as xgb
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#os.chdir("/Users/xuliu/Documents/kaggle/quora/")
os.chdir("D:/LX/Kaggle/Quora/")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_id = test.test_id
### remove missing values
### make 0 for missing values
train = train.ix[~train.question2.isnull(),:]
test = test.ix[~test.question2.isnull(),:]
test = test.ix[~test.question1.isnull(),:]
data = train.copy()

tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
#cvect = CountVectorizer(stop_words='english', ngram_range=(1, 1))

#tfidf_txt = pd.Series(train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()).astype(str)
tfidf_txt = pd.Series(train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()).astype(str)
tfidf.fit_transform(tfidf_txt)

def split_words(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)

def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()

def word_match(word1, word2):
    q1words = {}
    q2words = {}
    for word in word1:
        word = word.lower()
        if word not in stops:
            q1words[word] = 1
    for word in word2:
        word = word.lower()
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/float(len(q1words) + len(q2words))
    return R
    
    
    
def process_feature(data):
    word_lists1 = map(split_words, data.question1)
    word_lists2 = map(split_words, data.question2)
    data['sent_len_diff1'] = [float(i - j)/(j+0.005) for i,j in zip(map(len, word_lists1), map(len, word_lists2))]
    data['sent_len_diff2'] = [float(j - i)/(i+0.005) for i,j in zip(map(len, word_lists1), map(len, word_lists2))]
    data['sent1'] = map(len, word_lists1)
    data['sent2'] = map(len, word_lists2)
    data['diff_ratio'] = [diff_ratios(i, j) for i,j in zip(data.question1, data.question2)]
    data['word_match_ratio'] = [word_match(i, j) for i,j in zip(word_lists1, word_lists2)]
    tfidf1 = data.question1.map(lambda x: tfidf.transform([str(x)]).data)
    tfidf2 = data.question2.map(lambda x: tfidf.transform([str(x)]).data)
    
    data['tfidf_sum1'] = [np.sum(i) for i in tfidf1]
    data['tfidf_sum2'] = [np.sum(i) for i in tfidf2]
    data['tfidf_mean1'] = [np.mean(i) for i in tfidf1]
    data['tfidf_mean2'] = [np.mean(i) for i in tfidf2]
    data['tfidf_sum_diff'] = data['tfidf_sum1'] - data['tfidf_sum2']
    data['tfidf_mean_diff'] = data['tfidf_mean1'] - data['tfidf_mean2']
    return data
    
train = process_feature(train)

#params = {}
#params["objective"] = "binary:logistic"
#params['eval_metric'] = 'logloss'
#params["eta"] = 0.05
#params["subsample"] = 0.5
#params["min_child_weight"] = 3
#params["colsample_bytree"] = 0.5
#params["max_depth"] = 4
#params["seed"] = 1632
#y_train = data.is_duplicate
#d_train = data[col]
#d_train = xgb.DMatrix(d_train, label = y_train)
#history = xgb.cv(params,dtrain = d_train , num_boost_round = 10000, verbose_eval = 2, nfold = 5 ,early_stopping_rounds=30)
### 2000
test = process_feature(test)
#test.to_csv("test_featured.csv", index = False)
def get_noun(word_list_tag):
    if len(word_list_tag) ==0:
        return ""
    else:
        return sorted([w.lower() for w,t in word_list_tag if t[:1] in ['N']])

def get_noun_match_ratio(i, j):
    if len(i) and len(j):
        return sum([x in j for x in i])*2.0/(len(i) + len(j))
    else:
        return 0
        
def process_feature2(train):
    word_lists1 = map(split_words, train.question1)
    word_lists2 = map(split_words, train.question2)
    print "first word"
    train['first_word_match'] = [1 if len(i) and len(j) and i[0] == j[0] else 0 for i,j in zip(word_lists1, word_lists2)]
    print "math"
    question1_math = train.question1.apply(lambda x: '[math]' in x)
    question2_math = train.question2.apply(lambda x: '[math]' in x)
    train['math_match'] = [0 if i==j else 1 for i,j in zip(question1_math, question2_math)]
    print "number"
    number1 = train.question1.apply(lambda x: sorted([y for y in x if y.isdigit()]))
    number2 = train.question2.apply(lambda x: sorted([y for y in x if y.isdigit()]))
    train['number_match'] = [1 if i == j and len(i) and len(j) else 0 for i,j in zip(number1, number2)]
    #question1_noun = train.question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if len(w) and t[:1] in ['N']])
    print "nuon"
    word_list1_tag = [pos_tag(i) if len(i) else [] for i in word_lists1]
    word_list2_tag = [pos_tag(i) if len(i) else [] for i in word_lists2]
    question1_noun = map(get_noun, word_list1_tag)
    question2_noun = map(get_noun, word_list2_tag)                 
    train['noum_match_ratio'] = [get_noun_match_ratio(i, j) for i,j in zip(question1_noun, question2_noun)]
    return train                     
                      
col = ['sent_len_diff1', 'sent_len_diff2', 'sent1', 'sent2', 'diff_ratio', 'word_match_ratio', 'tfidf_sum1', 'tfidf_sum2', 'tfidf_mean1', 'tfidf_mean2', 'tfidf_sum_diff', 'tfidf_mean_diff']

train.to_csv("train_updated_feature.csv", index = False)

pos_train = train[train.is_duplicate == 1]
neg_train = train[train.is_duplicate == 0]
# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((len(pos_train)/ (len(pos_train) + len(neg_train) +0.0)) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train) + 0.0))

x_train = pd.concat([pos_train, neg_train])
#y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

x_train.to_csv("train_feature_update_with_resample.csv", index = False)

test_save = test.copy()

x_test = process_feature2(test)
x_test.to_csv("test_feature_update_with_resample.csv", index = False)




