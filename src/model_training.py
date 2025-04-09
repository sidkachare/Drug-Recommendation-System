# model_trainer.py

from sklearn.neighbors import NearestNeighbors
import joblib

class ModelTrainer:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto')

    def train(self, X):
        self.model.fit(X)
        return self.model

    def save_model(self, model_path="drug_recommender_model.pkl"):
        joblib.dump(self.model, model_path)

