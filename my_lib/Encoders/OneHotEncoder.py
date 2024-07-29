import pandas as pd

class OneHotEncoder:
    def __init__(self):
        return
    def encode(df,features):
        df_encoded=df.copy()
        for feature in features:
            for var in df_encoded[feature].unique():
                feature_name=f'{feature}_{var}'
                df_encoded[feature_name]=(df_encoded[feature]==var).astype(int)
        return df_encoded