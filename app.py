from pathlib import Path
import pickle
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

app = FastAPI(
    title="Real-Time ML Inference API",
    description="REST API for positive/negative sentiment classification.",
    version="1.0.0",
)

class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100, description="Texts to classify")

class Prediction(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict[str, float]

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model_version: str = "1.0.0"

def train_demo_model():
    positive = [
        "I loved this movie", "This film was wonderful", "Amazing story and great acting",
        "The product works perfectly", "Excellent quality and fast delivery",
        "I am very happy with this purchase", "The service was friendly and helpful",
        "A fantastic experience", "Really enjoyable and entertaining", "Highly recommend this",
        "The food tasted delicious", "Beautiful design and easy to use",
        "The app is useful and reliable", "Brilliant performance by the actors",
        "Everything was great", "A very satisfying experience", "I would buy this again",
        "The book was inspiring", "Superb quality for the price", "The support team solved my issue",
        "It exceeded my expectations", "Very comfortable and durable", "The ending was uplifting",
        "A delightful surprise", "I had a great time", "Wonderful customer service",
        "The interface is smooth and intuitive", "Fresh ingredients and tasty food",
        "This is one of my favorites", "A smart and worthwhile choice",
        "Loved every minute of it", "The music was beautiful", "It arrived in perfect condition",
        "Very impressive results", "The lesson was clear and helpful", "I feel pleased with it",
        "A great value", "The experience was seamless", "Strongly recommend this product",
        "It made my day"
    ]
    negative = [
        "I hated this movie", "This film was terrible", "Awful story and poor acting",
        "The product stopped working", "Very poor quality and slow delivery",
        "I am unhappy with this purchase", "The service was rude and unhelpful",
        "A disappointing experience", "Boring and not entertaining", "I do not recommend this",
        "The food tasted awful", "Bad design and difficult to use",
        "The app is buggy and unreliable", "Terrible performance by the actors",
        "Everything went wrong", "A very frustrating experience", "I would not buy this again",
        "The book was boring", "Poor quality for the price", "The support team ignored my issue",
        "It failed my expectations", "Very uncomfortable and fragile", "The ending was depressing",
        "A horrible surprise", "I had a bad time", "Terrible customer service",
        "The interface is confusing and slow", "Stale ingredients and tasteless food",
        "This is one of my least favorites", "A waste of money",
        "Regretted every minute of it", "The music was irritating", "It arrived damaged",
        "Very disappointing results", "The lesson was confusing and unhelpful", "I feel frustrated with it",
        "Bad value", "The experience was stressful", "Do not recommend this product",
        "It ruined my day"
    ]
    texts = positive + negative
    labels = ["Positive"] * len(positive) + ["Negative"] * len(negative)
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                          alpha=1e-4, max_iter=500, random_state=42, early_stopping=True)
    model.fit(X, labels)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)
    with VECTORIZER_PATH.open("wb") as f:
        pickle.dump(vectorizer, f)
    return model, vectorizer

def load_artifacts():
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        with MODEL_PATH.open("rb") as f:
            model = pickle.load(f)
        with VECTORIZER_PATH.open("rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    return train_demo_model()

model, vectorizer = load_artifacts()

@app.get("/")
def root():
    return {
        "message": "Real-Time ML Inference API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_version": "1.0.0"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    cleaned = [text.strip() for text in request.texts]
    if any(not text for text in cleaned):
        raise HTTPException(status_code=422, detail="Each text must contain at least one non-whitespace character.")
    X = vectorizer.transform(cleaned)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    results = []
    for text, probs in zip(cleaned, probabilities):
        index = int(np.argmax(probs))
        label = str(classes[index])
        results.append(Prediction(
            text=text,
            label=label,
            confidence=round(float(probs[index]), 6),
            probabilities={str(cls): round(float(prob), 6) for cls, prob in zip(classes, probs)}
        ))
    return PredictResponse(predictions=results)
