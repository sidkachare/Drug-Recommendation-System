import pandas as pd
import re
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPreprocessor:
    def __init__(self, raw_file_path, cleaned_dir="data/processed", cleaned_filename="cleaned_drugs_com_reviews.csv"):
        self.raw_file_path = raw_file_path
        self.cleaned_dir = cleaned_dir
        self.cleaned_file_path = os.path.join(cleaned_dir, cleaned_filename)
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

        os.makedirs(self.cleaned_dir, exist_ok=True)

    def clean_text(self, text):
        if pd.isnull(text):
            return ""
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def load_and_process(self):
        df = pd.read_csv(self.raw_file_path)
        df.dropna(subset=['side_effects', 'drug_name'], inplace=True)
        df.drop_duplicates(inplace=True)

        df['clean_side_effects'] = df['side_effects'].apply(self.clean_text)

        df.to_csv(self.cleaned_file_path, index=False)
        X_text = df['clean_side_effects']
        X_vectorized = self.vectorizer.fit_transform(X_text)
        drug_names = df['drug_name'].values

        return X_vectorized, drug_names, self.vectorizer
