import pandas as pd
import numpy as np

class DataCleaner:
    def clean_data(self, raw_data):

        #Convert to pandas DataFrame

        df = pd.read_excel(raw_data)

        #Remove duplicates
        df.drop_duplicates(inplace=True)

        #Handle missing values
        df.fillna(df.mean(), inplace=True)

        #Normalize numerical collumns

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean())/ df[numerical_cols].std()

        return df