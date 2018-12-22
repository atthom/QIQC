#!/usr/bin/env python3

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import io
import pandas as pd
from dask import dataframe as dd 
import dask.multiprocessing
from collections import Counter
import json
import time
import numpy as np

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
        row = filter(lambda item: item!='', row.split(" "))
        new_row = np.asarray([words2vec[word] for word in row])
        print(new_row.shape)
        new_row = np.sum(new_row, axis=1)
        print(new_row.shape)
        return new_row


    words2vec = json.load(open("word2vec.json", "r"))
    train = dd.read_csv("./data/train_restem3.csv")
    train["question_text"] = train["question_text"].apply(do_vectorize, meta=('list'))
    train = train.compute(scheduler='threads')
    train.to_csv("train_vectors.csv", index=False)

vectorize()
#restem()
#stem2words(train)
#texts = train["question_text"].values.tolist()

#texts = [stemSentence(text) for text in texts]
#all_words = [word for text in texts for word in text.split(" ")]
#print(len(all_words), len(list(set(all_words))))


#train["question_text"] = train["question_text"].apply(reformat)
#print(train)

