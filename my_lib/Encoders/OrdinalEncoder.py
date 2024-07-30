import pandas as pd
import numpy as np
class OrdinalEncoder:
    """
    Encode categorical features as an integer array

    Input the pandas dataframe and features (list of n items) of that dataframe that are to be encoded.
    The variables will be encoded arbitrarily with integers. if order in the feature matters, additional
    parameter named category can be given defining the order in a matrix of (n,), note that the shape of
    category must be (n,), so those features don't have orders in variables should be set to None.

    feature_names attribute contains the list of all the variable mappings.

    Example can be found in the playground.ipynb
    
    """
    def __init__(self):
        self.feature_maps=[]
        return

    def transform(self,df,features,category=None):
        df_encoded=df.copy()
        if category==None:
            for feature in features:
                values=sorted(df_encoded[feature].unique())
                feature_map={values[i]:i for i in range(len(values))}
                self.feature_maps.append(feature_map)
                df_encoded[feature]=df_encoded[feature].map(feature_map)
        else:
            for (i,feature) in enumerate(features):
                if category[i]==None:
                    values=sorted(df_encoded[feature].unique())
                    feature_map={values[i]:i for i in range(len(values))}
                    self.feature_maps.append(feature_map)
                    df_encoded[feature]=df_encoded[feature].map(feature_map)
                else:
                    values=category[i]
                    feature_map={values[i]:i for i in range(len(values))}
                    self.feature_maps.append(feature_map)
                    df_encoded[feature]=df_encoded[feature].map(feature_map)
        return df_encoded