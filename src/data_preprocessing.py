import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    def clean_text(self, text):
        if pd.isnull(text):
            return ""
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def load_and_process(self):
        df = pd.read_csv(self.file_path)
        df.drop_duplicates(inplace=True)
        df.dropna(subset=['side_effects', 'drug_name'], inplace=True)

        df['clean_side_effects'] = df['side_effects'].apply(self.clean_text)

        X_text = df['clean_side_effects']

        X_vectorized = self.vectorizer.fit_transform(X_text)

        drug_names = df['drug_name'].values

        return X_vectorized, drug_names, self.vectorizer
