import pandas as pd
import numpy as np
class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.

    Input the pandas dataframe and features of that dataframe that are to be encoded.
    feature_names attribute contains dictionary of all the encoded features and the values they used to hold.

    Examples can be found in playground.ipynb

    """
    def __init__(self):
        self.feature_names={}
        return

    def transform(self,df,features,drop_first=False,dtype=np.int8):
        if type(drop_first)!=bool:
            raise TypeError(f"drop_first parameter must be of type bool, but got {type(drop_first)}")
        df_encoded=df.copy()
        for feature in features:
            values=df_encoded[feature].unique()
            self.feature_names[feature]=values
            for var in values[int(drop_first):]:
                feature_name=f'{feature}_{var}'
                df_encoded[feature_name]=(df_encoded[feature]==var).astype(dtype)
            df_encoded.drop(feature, axis=1, inplace=True)
        return df_encoded

    
    
    