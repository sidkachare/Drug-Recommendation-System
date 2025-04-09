from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ModelEvaluator:
    def evaluate(self, X_features):
        similarity_matrix = cosine_similarity(X_features)
        avg_similarity = np.mean(similarity_matrix)
        print(f"Average pairwise cosine similarity: {avg_similarity:.4f}")
        return avg_similarity
