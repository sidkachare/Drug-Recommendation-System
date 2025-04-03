import pickle
import pandas as pd

def recommend_drug(drug_name, model_path="models/drug_recommender.pkl"):
    with open(model_path, "rb") as f:
        tfidf, similarity_matrix, df = pickle.load(f)

    if drug_name not in df['drug_name'].values:
        return "Drug not found!"

    idx = df[df['drug_name'] == drug_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_drugs = [df.iloc[i[0]]['drug_name'] for i in scores[1:6]]

    return top_drugs

# Test function
if __name__ == "__main__":
    print(recommend_drug("doxycycline"))
