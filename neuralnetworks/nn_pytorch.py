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
data_tf = torch.utils.data.TensorDataset(X_train, Y_train)
