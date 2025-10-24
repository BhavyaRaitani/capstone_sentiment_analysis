# demo.py
"""
Live demo for Amazon Review Sentiment Analysis using FastText.
Predicts sentiment of sample reviews.
"""

import fasttext
import os

def predict_reviews(model_path, samples):
    """
    Load FastText model and predict sentiment for a list of reviews.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    model = fasttext.load_model(model_path)
    print("Model loaded successfully!\n")

    labels, probs = model.predict(samples)
    
    for i, review in enumerate(samples):
        sentiment = "Positive" if labels[i][0] == "__label__2" else "Negative"
        print(f"Review {i+1}: {review}")
        print(f"Predicted Label: {labels[i][0]} ({sentiment}) | Probability: {probs[i][0]:.4f}")
        print("-" * 60)

def main():
    # Path to your trained model
    MODEL_PATH = os.path.join("models", "sentiment_analysis_model.bin")
    
    # Sample reviews for demo
    samples = [
        "This product is amazing! Works perfectly.",
        "Terrible experience, complete waste of money.",
        "Not bad but could be better.",
        "Exceeded my expectations, highly recommend!",
        "Arrived late and packaging was damaged."
    ]

    predict_reviews(MODEL_PATH, samples)

if __name__ == "__main__":
    main()
