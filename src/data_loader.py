import pandas as pd
from src.config import FEATURE_PREFIX

def load_data():
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")
    return train, test

def get_features(df):
    return [col for col in df.columns if col.startswith(FEATURE_PREFIX)]