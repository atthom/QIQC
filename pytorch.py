#!/usr/bin/env python3
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def getXYtrain():
    from sklearn.model_selection import train_test_split
    print("load data...")
    
    Y = np.load("./data/Y_train.npy").ravel()
    X = np.load("doc2vec.model.docvecs.vectors_docs.npy")

    print("format train/test dataset...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

def predict(model, X_test, Y_test):
    from sklearn.metrics import classification_report
    print("predict...")
    Y_preds = model.predict(X_test)
    print(classification_report(Y_test, Y_preds))


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2) 


def create_model():
    n_in, n_h, n_out, batch_size = 100, 25, 2, 64

    model = nn.Sequential(nn.Linear(n_in, n_h),
                     nn.ReLU(),
                     nn.Linear(n_h, n_out),
                     nn.Sigmoid())
    
    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
