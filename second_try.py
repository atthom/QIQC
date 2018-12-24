#!/usr/bin/env python3

from gensim.utils import lemmatize
from multiprocessing import Pool
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import lemmatize
import json
import pandas as pd
import numpy as np

from gensim.test.utils import common_texts


def create_model():
    data = pd.read_csv("./data/train.csv")
    if False:
        sentences = data["question_text"].values
        print(sentences[0])
        p = Pool(10)
        phrases = p.map(lemmatize, sentences)
        phrases = np.array(phrases)
        print(phrases[0])
        np.save("./data/phrases_lemmatize.npy", phrases)
    if True:
        phrases = np.load("./data/phrases.npy")
    else:
        pp = np.load("./data/phrases_lemmatize.npy")
        print(pp[0])
        phrases = []
        for p in pp:
            phrases.append([m.decode("utf-8").split("/")[0] for m in p])
        print(phrases[0])
        np.save("./data/phrases.npy", phrases)
    tag = data["qid"].values

    print("construct document...")
    documents = [TaggedDocument(doc, [qid]) for doc, qid in zip(phrases, tag)]
    print("create model")
    model = Doc2Vec(documents, vector_size=100, window=5, min_count=10, workers=6, epochs=80, verbose=1)

    #fname = get_tmpfile("doc2vec.model")
    model.save("doc2vec.model")
    model.save("doc2vec_model")


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


def do_learning(model, kargs):
    X_train, X_test, Y_train, Y_test = getXYtrain()

    clf = model(**kargs)
    print("model", str(model), str(kargs), "...")
    clf.fit(X_train, Y_train)
    predict(clf, X_test, Y_test)


import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

do_learning(GaussianNB, {})
do_learning(xgb.XGBClassifier, {})
do_learning(xgb.XGBClassifier, {"objective":'binary:logistic', "colsample_bytree": 0.3, \
                                "learning_rate": 0.05, "max_depth": 5, "alpha": 10, "n_estimators": 42})
do_learning(xgb.XGBClassifier, {"objective":'binary:logistic', "learning_rate": 0.05, \
                                 "alpha": 10, "n_estimators": 100})
do_learning(xgb.XGBClassifier, {"objective":'binary:logistic', "learning_rate": 0.05, \
                                "colsample_bytree": 0.3, "alpha": 10, "n_estimators": 100})

do_learning(RandomForestClassifier, {"n_estimators":20, "n_jobs":-1, "random_state":42})
do_learning(RandomForestClassifier, {"n_estimators":20, "n_jobs":-1, "random_state":42, "min_samples_split":10})
do_learning(RandomForestClassifier, {"n_estimators":100, "n_jobs":-1, "random_state":42, \
                                      "min_samples_split":10})





