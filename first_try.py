#!/usr/bin/env python3

from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from dask import dataframe as dd 
from collections import Counter
from sklearn.metrics import f1_score
import dask.multiprocessing
import xgboost as xgb
import pandas as pd
import numpy as np
import io
import json
import time


porter=PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def load_vectors(fname, uniq_words):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    how_many = 0
    dd = time.time()
    ll = len(uniq_words)
    cc = 0
    for line in fin:
        how_many += 1
        if how_many % 10000 == 0:
            print(how_many, time.time() - dd)
            dd = time.time()
        tokens = line.rstrip().split(' ')
        word = stemSentence(tokens[0])[:-1]
        if word in uniq_words:
            cc += 1
            data[word] = list(map(float, tokens[1:]))
            if cc == ll:
                break            
    return data

def stem2words(train):
    texts = train["question_text"].values.tolist()
    return [word for text in texts for word in text.split(" ")]
    

def clean(row):
    row = row.replace(":", " ").replace("/", " ") \
                .replace("-", " ").replace("=", " ") \
                .replace("\\", " ").replace("'", "") \
                .replace("\"", "").replace(".", " ") \
                .replace(",", " ")
    return stemSentence(row)

#test = pd.read_csv("./data/test.csv")

def get_uniq_dict():
    train = pd.read_csv("./data/train_restem2.csv")
    all_words = stem2words(train)

    count = [(k, v) for k, v in Counter(all_words).items()]
    count = list(filter(lambda item: item[1] >= 10, count))
    uniq_words = [k for k, v in count]
    #uniq_words = list(set(all_words))
    print(len(all_words), len(uniq_words))
    word2vec = load_vectors("wiki-news-300d-1M-subword.vec", uniq_words)
    json.dump(word2vec, open("word2vec.json", "w"))
#print(cc)

#train = dd.from_pandas(train, npartitions=8)
def restem():
    train = dd.read_csv("./data/train.csv")
    train["question_text"] = train["question_text"].apply(clean, meta=('str'))
    train = train.compute(scheduler='threads')
    train.to_csv("train_restem.csv", index=False)



def vectorize():
    def do_vectorize(row):
        try:
            split_row = filter(lambda item: item!='', row.split(" "))
        except:
            split_row = [row]
        new_row = np.asarray([0.0]*300)
        for word in split_row:
            if word in words2vec:
                new_row += words2vec[word]
        return new_row


    words2vec = json.load(open("word2vec.json", "r"))
    train = dd.read_csv("./data/train_restem3.csv")
    train["question_text"] = train["question_text"].apply(do_vectorize, meta=pd.Series())
    train = train.compute(scheduler='threads')
    train.to_csv("train_vectors.csv", index=False)




def learning():

    print("load data...")
    
    Y = np.load("./data/Y_train.npy")
    X = np.load("./data/all_desc32.npy")

    print("format train/test dataset...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("learning...")
    xg_reg = xgb.XGBClassifier(objective ='binary:hinge', colsample_bytree = 0.3, learning_rate = 0.05,
                    max_depth = 5, alpha = 10, n_estimators = 42)

    xg_reg.fit(X_train, Y_train)


    print("predict...")
    Y_preds = xg_reg.predict(X_test)

    print("f1 score:", f1_score(Y_test, Y_preds))


def diry():
    vectors = pd.read_csv("./data/train_vectors2.csv")["question_text"]
    desc = []
    cc = 0
    cut = 1300000
    for v in vectors:
        cc += 1
        
        if cc > cut:
            v = v.replace("[", "").replace("]", "")
            v = filter(lambda item: item!='', v.split(" "))
            true_vec = [float(val) for val in v]
            desc.append(true_vec)
            if cc % 1400000 == 0:
                desc = np.array(desc)
                np.save("desc" + str(cc) + ".npy", desc)
                break

    desc = np.array(desc)
    np.save("desc" + str(cc) + ".npy", desc)

    desc = np.load("desc100000.npy")
    for i in range(2, 14):
        di = np.load("desc" + str(i*100000) + ".npy")
        desc = np.vstack((desc, di))

    dlast = np.load("desc1306122.npy")
    desc = np.vstack((desc, dlast))
    print(desc.shape)

    np.save("./data/all_desc.npy", desc)

    vectors = pd.read_csv("./data/train_vectors2.csv")
    Y = vectors["target"].values.reshape(-1, 1)
    np.save("./data/Y_train", Y)

    X = np.load("./data/all_desc.npy").astype(np.float32)
    np.save("./data/all_desc32.npy", X)


def bayesLearning():
    from sklearn.naive_bayes import GaussianNB

    print("load data...")
    
    Y = np.load("./data/Y_train.npy").ravel()
    X = np.load("./data/all_desc32.npy")

    print("format train/test dataset...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("fit...")
    clf = GaussianNB()
    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)
    print("f1 score:", f1_score(Y_test, y_pred))


def bayes2Learning():
    from sklearn.svm import NuSVC

    print("load data...")
    
    Y = np.load("./data/Y_train.npy").ravel()
    X = np.load("./data/all_desc32.npy")

    print("format train/test dataset...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("fit...")
    clf = NuSVC(nu=0.1)
    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)
    print("f1 score:", f1_score(Y_test, y_pred))

def randomForest2learning():
    from sklearn.ensemble import RandomForestClassifier

    print("load data...")
    
    Y = np.load("./data/Y_train.npy").ravel()
    X = np.load("./data/all_desc32.npy")

    print("format train/test dataset...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("fit...")
    clf = RandomForestClassifier(n_estimators=2, n_jobs=-1, random_state=42, verbose=1, min_samples_split=10)
    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)
    print("f1 score:", f1_score(Y_test, y_pred))

def isolationForest():
    from sklearn.ensemble import IsolationForest

    print("load data...")
    
    Y = np.load("./data/Y_train.npy").ravel()
    X = np.load("./data/all_desc32.npy")

    print("format train/test dataset...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("fit...")
    clf = IsolationForest(n_estimators=100, n_jobs=-1, random_state=42, verbose=1)
    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)
    print("f1 score:", f1_score(Y_test, y_pred))

isolationForest()

#vectorize()
#restem()
#stem2words(train)
#texts = train["question_text"].values.tolist()

#texts = [stemSentence(text) for text in texts]
#all_words = [word for text in texts for word in text.split(" ")]
#print(len(all_words), len(list(set(all_words))))


#train["question_text"] = train["question_text"].apply(reformat)
#print(train)

