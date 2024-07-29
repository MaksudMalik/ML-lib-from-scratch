import numpy as np
import pandas as pd

class LinearRegression:
    
    def __init__(self):
        self.coefficient=None
        self.bias=None

    def train(self, X, Y):
        X=X.values
        Y=Y.values
        ones=np.ones(X.shape[0])
        X=np.column_stack((ones,X))
        c=np.dot(X.T,X)
        z=np.dot(X.T,Y)
        w=np.linalg.inv(c)@z
        self.coefficient=w[1:]
        self.bias=w[0]
        return
    
    def predict(self,X):
        try:
            X=X.values
        except:
            None
        pred=(self.coefficient@X.T)+self.bias
        return pred