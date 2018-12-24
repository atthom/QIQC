#!/usr/bin/env python3

from gensim.utils import lemmatize
from multiprocessing import Pool
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import lemmatize
import json
import pandas as pd
import numpy as np
from sklearn.externals import joblib

from gensim.test.utils import common_texts


def getXYtrain():
    from sklearn.model_selection import train_test_split
    print("load data...")
    
    Y = np.load("./data/Y_train.npy").ravel()
    X = np.load("doc2vec.model.docvecs.vectors_docs.npy")

    print("format train/test dataset...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test


def predict(model, X_test, Y_test):
    from sklearn.metrics import classification_report, f1_score
    print("predict...")
    Y_preds = model.predict(X_test)
    print(classification_report(Y_test, Y_preds))
    print(f1_score(Y_test, Y_preds))
    print(f1_score(Y_test + 1, Y_preds + 1))



def create_submit_vector():
    data = np.load("./data/phrases_test.npy")
    model = Doc2Vec.load("doc2vec.model")
    vectors = [model.infer_vector(phrase) for phrase in data]
    np.save("./data/vectors_test.npy", vectors)


def do_submit(model, kargs):
    ids = pd.read_csv("./data/test.csv")["qid"].values.ravel()
    X = np.load("./data/vectors_test.npy")

    model = joblib.load("first.model")

    y_test = model.predict(X)
    test = list(zip(ids, y_test))
    test = np.array(test)
    test = pd.DataFrame(test, columns=["qid","prediction"])
    test.to_csv("submission.csv", index=False)


from sklearn.naive_bayes import GaussianNB

do_submit(GaussianNB, {})