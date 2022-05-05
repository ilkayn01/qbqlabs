import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split                                        #split = 0.35
from sklearn.metrics import accuracy_score
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn

Y, X = data.iloc[:, -1:], data.iloc[:,:-1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = Y, random_state=42)

def arr_tensor(x):
  x = torch.tensor(np.array(x), dtype=torch.float)                                #converting train sets to numpy arrays to tensor
  return x

def training(X_train, X_test, Y_train, Y_test, epochs, alpha):
  X_train = arr_tensor(X_train)
  Y_train = arr_tensor(Y_train)
  
  X_test = arr_tensor(X_test)
  Y_test = arr_tensor(Y_test)

  td = torch.utils.data.TensorDataset(X_train, Y_train)

  classifier = nn.Sequential(nn.Linear(in_features=30, out_features=15), nn.ReLU(), nn.Linear(15, 15), nn.ReLU(), nn.Linear(15, 1), nn.Sigmoid())
  criterion = nn.BCELoss()
  optimizer = torch.optim.Adam(classifier.parameters(), lr=alpha, weight_decay=0.0006)
  return
