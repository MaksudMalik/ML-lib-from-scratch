import pandas as pd
import numpy as np
class MaxAbsScaler:

    def __init__(self):
        self.params={}
        return
    
    def fit(self,df):
        for col in df.columns:
            xmax=np.max(df[col])
            self.params[col]=xmax
        return

    def transform(self,df):
        scaledval=[]
        for col in df.columns:
            scaledval.append(df[col]/self.params[col])
        scaledval=np.column_stack(scaledval)
        return scaledval

    def fit_transform(self,df):
        self.fit(df)
        scaledval=self.transform(df)
        return scaledval