"""
Web interface for Amazon Review Sentiment Analysis using FastText.
"""

import fasttext
import gradio as gr
import os

# --- Load FastText model once ---
MODEL_PATH = os.path.join("models", "sentiment_analysis_model.bin")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")

model = fasttext.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# --- Define prediction function ---
def predict_sentiment(review):
    """
    Predict sentiment for a single review using FastText.
    """
    if not review.strip():
        return "No input provided.", ""
    
    labels, probs = model.predict([review])
    label = labels[0][0]
    prob = probs[0][0]
    sentiment = "😊 Positive" if label == "__label__2" else "😠 Negative"

    # Return both pieces of info separately
    return sentiment, f"{prob:.2f}"

# --- Gradio interface ---
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Enter a review text", placeholder="Type your review here..."),
    outputs=[
        gr.Textbox(label="Predicted Sentiment", interactive=False),
        gr.Textbox(label="Confidence", interactive=False)
    ],
    title="Amazon Review Sentiment Analysis",
    description="Type a product review below to see if it's positive or negative.",
    examples=[
        ["This product is amazing! Works perfectly."],
        ["Terrible experience, complete waste of money."],
        ["Not bad but could be better."],
    ]
)

# --- Launch the app ---
if __name__ == "__main__":
    iface.launch()
