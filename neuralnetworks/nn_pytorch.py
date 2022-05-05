import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split                                        #split = 0.35
from sklearn.metrics import accuracy_score
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn

data = pd.read_csv('cancer.csv')
del data['Unnamed: 0']
Y = data.iloc[:, -1:]
X = data.iloc[:,:-1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = Y, random_state=42)


X_train = torch.tensor(np.array(X_train), dtype=torch.float)                                #converting train sets to numpy arrays to tensor
Y_train = torch.tensor(np.array(Y_train), dtype = torch.float)
td = torch.utils.data.TensorDataset(X_train, Y_train)

classifier = nn.Sequential(nn.Linear(in_features=30, out_features=15), nn.ReLU(), nn.Linear(15, 15), nn.ReLU(), nn.Linear(15, 1), nn.Sigmoid())
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.006, weight_decay=0.0006)
