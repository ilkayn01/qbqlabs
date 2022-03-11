import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv('cancer.csv')
del data['Unnamed: 0']

X = data.drop("class", axis = 1)
Y = data['class']
data.head()

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, stratify = Y, random_state=42)

from sklearn.metrics import accuracy_score
X_train = X_train.values
X_test = X_test.values

class Perceptron:
    def __init__ (self):
        self.w = None                                             #weights initialized to none
        self.theta = None                                         #threshold initialized to none
        
    def output_function_Y(self, x):                               #function to find Y using Yin and theta
      Yin = np.dot(self.w, x)                                     #sum of weighted input Yin = w1*x1 + w2*x2 + ..... + wn*xn
      if (Yin > self.theta):                                      #did not consider -1 since target variale class in breast cancer dataset contains only 1 and 0
        return 1
      else:
        return 0

    def predict_output_Y(self, X):
        Y = []
        for x in X:
            Y.append(self.output_function_Y(x))
        return np.array(Y)
    
    def accuracy_predict(self, X, Y, epoch, alpha = 1):           #initialized alpha to 1
        self.w, acc, accuracy = np.ones(X.shape[1]), {}, 0
        self.theta = 0                                            #threshold = 0

        for i in range(epoch):
            for x, y in zip(X, Y):
                y_pred = self.output_function_Y(x)                #predicts values for given features                
                                                                  #If Y != t, change weights
                if (y == 1 and y_pred == 0):                      #if outputs do not match
                    
                    #self.w = #write your code here                #wi (new) = wi (old) + (alpha*t*xi)
                    
                    self.w = self.w + (alpha*1*x)
                    
                    #self.theta = #write your code here            #w0 (new) = w0 (old) + (alpha*t)
                    
                    self.theta = self.theta - (alpha*1)
                
                elif (y == 0 and y_pred == 1):
                    self.w = self.w - (alpha*1*x)                 #wi (new) = wi (old) + (alpha*t*xi)
                    self.theta = self.theta + (alpha*1)           #w0 (new) = w0 (old) + (alpha*t)

            acc[i] = accuracy_score(self.predict_output_Y(X), Y)
            if (acc[i] > accuracy):
                accuracy = acc[i]
                weights, threshold = self.w, self.theta
        
        self.w, self.theta = weights, threshold
        print(f'Accuracy of perceptron model = {accuracy*100}%')

perceptron = Perceptron()
perceptron.accuracy_predict(X_train, Y_train, 15000, 0.6)         #play with epochs and learning rate

Y_pred_test = perceptron.predict_output_Y(X_test)
print(f'Accuracy of perceptron model (test set) = {accuracy_score(Y_pred_test, Y_test)*100}%')
