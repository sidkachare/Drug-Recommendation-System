from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load model
with open("models/drug_recommender.pkl", "rb") as f:
    tfidf, similarity_matrix, df = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Drug Recommendation API is running!"}

@app.get("/recommend")
def recommend(drug: str):
    if drug not in df['drug_name'].values:
        return {"error": "Drug not found!"}

    idx = df[df['drug_name'] == drug].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_drugs = [df.iloc[i[0]]['drug_name'] for i in scores[1:6]]

    return {"recommended_drugs": top_drugs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
