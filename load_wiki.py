#!/usr/bin/env python3

import io
import pandas as pd

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        break
    return data

def clean(row):
    row = row.replace("?", " ").replace("-", "") \
                .replace(")", "").replace("(", "") \
                .replace("'", "").replace("\\", "") \
                .replace("/", "").replace(".", " ") \
                .replace(",", " ").replace("%", "") \
                .replace("\"", "").replace(":", " ") \
                .replace("]", "").replace("[", "") \
                .replace("}", "").replace("{", "")  \
                .replace("+", " ").replace("“", "")
    return row.split(" ")
#vectors = load_vectors("wiki-news-300d-1M-subword.vec")
#print(list(vectors[","]))
train = pd.read_csv("./data/train.csv")

train["question_text"] = train["question_text"].apply(clean)
questions = train["question_text"].values.ravel().tolist()
uniques_words = list(set([word for q in questions for word in q]))
#print(uniques_words)
print(len(uniques_words))