import pandas as pd
import numpy as np
class Normalizer:

    def __init__(self):
        return

    def normalize(self,vals):
        res=np.sqrt(vals@vals)
        norm=[i/res for i in vals] 
        return norm
    
    def transform(self,df):
        norm_rows=[]
        for i in range(len(df)):
            if isinstance(df, pd.DataFrame):
                df=df.reset_index(drop=True)
                norm_row=self.normalize(df.loc[i].values)
            else:
                norm_row=self.normalize(df[i])
            norm_rows.append(norm_row)
        norm_rows=np.vstack(norm_rows)
        return norm_rows