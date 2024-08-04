import pandas as pd
import numpy as np
class RobustScaler:

    def __init__(self):
        self.params={}
        return
    
    def fit(self,df):
        for col in df.columns:
            median=df[col].median()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            self.params[col]=median,iqr
        return
    
    def scale(self,x,median,iqr):
        xnew=(x-median)/iqr
        return xnew

    def transform(self,df):
        scaledval=[]
        for col in df.columns:
            median=self.params[col][0]
            iqr=self.params[col][1]
            scaledval.append(df[col].apply(lambda x:self.scale(x,median,iqr)).values)
        scaledval=np.column_stack(scaledval)
        return scaledval

    def fit_transform(self,df):
        self.fit(df)
        scaledval=self.transform(df)
        return scaledval