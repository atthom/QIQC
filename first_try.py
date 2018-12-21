from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import io
import pandas as pd
from dask import dataframe as dd 

porter=PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def stem2words(train):
    texts = train["question_text"].values.tolist()
    all_words = [word for text in texts for word in text.split(" ")]
    print(len(all_words), len(list(set(all_words))))

def clean(row):
    return stemSentence(row)

#test = pd.read_csv("./data/test.csv")

train = pd.read_csv("./data/train.csv")

stem2words(train)

train = dd.from_pandas(train, npartitions=8)
train["question_text"] = train["question_text"].apply(clean)
train = train.compute()

stem2words(train)
#texts = train["question_text"].values.tolist()


#texts = [stemSentence(text) for text in texts]
#all_words = [word for text in texts for word in text.split(" ")]
#print(len(all_words), len(list(set(all_words))))


#train["question_text"] = train["question_text"].apply(reformat)
#print(train)

