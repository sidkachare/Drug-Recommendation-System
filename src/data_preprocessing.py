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
        df.dropna(subset=['medical_condition', 'side_effects', 'rating'], inplace=True)

        df['clean_side_effects'] = df['side_effects'].apply(self.clean_text)

        X_text = df['clean_side_effects']
        y = df['rating']

        X_features = self.vectorizer.fit_transform(X_text)

        X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, self.vectorizer
