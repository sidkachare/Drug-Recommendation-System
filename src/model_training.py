import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
features = df['side_effects'] + " " + df['drug_classes']
tfidf_matrix = tfidf.fit_transform(features)

# Compute similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Save model
with open("models/drug_recommender.pkl", "wb") as f:
    pickle.dump((tfidf, similarity_matrix, df), f)

print("✅ Model Training Completed & Saved")
