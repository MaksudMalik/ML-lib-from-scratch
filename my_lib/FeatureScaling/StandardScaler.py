import pandas as pd
import numpy as np
class StandardScaler:

    def __init__(self):
        self.params={}
        return

    def standardize(self,x,miu,sigma):
        z=(x-miu)/sigma
        return z

    def fit(self,df):
        for col in df.columns:
            mean=np.mean(df[col])
            sd=np.std(df[col])
            self.params[col]=[mean,sd]
        return

    def transform(self,df):
        stdval=[]
        for col in df.columns:
            mean=self.params[col][0]
            sd=self.params[col][1]
            stdval.append(df[col].apply(lambda x:self.standardize(x,mean,sd)))
        stdval=np.column_stack(stdval)
        return stdval
    
    def fit_transform(self,df):
        self.fit(df)
        stdval=self.transform(df)
        return stdval