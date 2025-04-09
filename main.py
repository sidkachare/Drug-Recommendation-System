from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.inference import InferenceEngine

def main():
    preprocessor = DataPreprocessor(raw_file_path="data/raw/drugs_side_effects_drugs_com.csv")
    X, drug_names, vectorizer = preprocessor.load_and_process()

    trainer = ModelTrainer(n_neighbors=5)
    model = trainer.train(X)
    trainer.save_model()

    evaluator = ModelEvaluator()
    evaluator.evaluate(X)

    engine = InferenceEngine("drug_recommender_model.pkl", vectorizer, drug_names)
    test_input = "nausea, dizziness, and headache"
    recommendations = engine.recommend(test_input)

    print("Top recommended drugs:")
    for i, drug in enumerate(recommendations, 1):
        print(f"{i}. {drug}")

if __name__ == "__main__":
    main()
