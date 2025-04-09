import joblib
import re
import string

class InferenceEngine:
    def __init__(self, model_path, vectorizer, drug_names):
        self.model = joblib.load(model_path)
        self.vectorizer = vectorizer
        self.drug_names = drug_names

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def recommend(self, side_effect_text, top_n=5):
        cleaned = self.clean_text(side_effect_text)
        query_vector = self.vectorizer.transform([cleaned])
        distances, indices = self.model.kneighbors(query_vector, n_neighbors=top_n)
        recommendations = [self.drug_names[idx] for idx in indices[0]]
        return recommendations

