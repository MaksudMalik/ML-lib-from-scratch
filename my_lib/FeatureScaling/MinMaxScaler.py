import pandas as pd
import numpy as np
class MinMaxScaler:

    def __init__(self):
        self.params={}
        return

    def normalize(self,x,xmax,xmin):
        x_=(x-xmin)/(xmax-xmin)
        return x_
    
    def fit(self,df):
        for col in df.columns:
            xmax=np.max(df[col])
            xmin=np.min(df[col])
            self.params[col]=[xmax,xmin]
        return

    def transform(self,df):
        normval=[]
        for col in df.columns:
            xmax=self.params[col][0]
            xmin=self.params[col][1]
            normval.append(df[col].apply(lambda x:self.normalize(x,xmax,xmin)))
        normval=np.column_stack(normval)
        return normval

    def fit_transform(self,df):
        self.fit(df)
        normval=self.transform(df)
        return normval