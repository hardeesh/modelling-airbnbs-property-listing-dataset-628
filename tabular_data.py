import pandas as pd
import numpy as np
from sklearn import datasets



def clean_tabular_data(data):
    def remove_rows_with_missing_ratings(df):
        cleaner_df = df.dropna(subset=['Cleanliness_rating'])
        return cleaner_df
    
    def combine_description_strings (df):
        df.dropna(subset=['Description'], inplace=True)
        cleaner_df = df[~df['Description'].str.contains('sleeps 6 with pool')]
        cleaner_df = cleaner_df.drop(['Unnamed: 19'], axis=1)
        cleaner_df['Description'] = cleaner_df['Description'].apply(eval)
        cleaner_df['Description'] = cleaner_df['Description'].apply(lambda x: [item for item in x if item])
        cleaner_df['Description'] = cleaner_df['Description'].apply(lambda x: " ".join(x))
        cleaner_df['Description'] = cleaner_df['Description'].str.replace('About this space', '')
        return cleaner_df
    
    def set_default_feature_values(df):
        df['beds'] = df['beds'].fillna(1)
        df['bathrooms'] = df['bathrooms'].fillna(1)
        df['bedrooms'] = df['bedrooms'].fillna(1)
        df['guests'] = df['guests'].fillna(1)
        return df
    
    data = remove_rows_with_missing_ratings(data)
    data = combine_description_strings(data)
    data = set_default_feature_values(data)

    return data
    
if __name__ == "__main__":
    data = pd.read_csv('tabular_data/listing.csv')
    data = pd.DataFrame(data)
    
    data = clean_tabular_data(data)
    
    data.to_csv(r"C:\Users\harde\Documents\AiCore\Airbnb/tabular_data/clean_tabular_data.csv", index=False)


def load_airbnb(label):
    features = data.drop(columns=[label])
    labels = data[label]
    
    return features, labels
