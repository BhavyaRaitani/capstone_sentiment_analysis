import fasttext
import os

def train_model(train_path, model_output_path):
    print(f"Training model using data from: {train_path}")
    
    model = fasttext.train_supervised(
        input=train_path,
        lr=0.1,          # learning rate
        epoch=10,        # number of epochs
        wordNgrams=2,    # use bigrams
        dim=100,         # embedding dimensions
        loss='softmax',  # loss function
        thread=8         # number of threads
    )

    model.save_model(model_output_path)
    print(f"✅ Model saved at {model_output_path}")
    return model


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    train_file = os.path.join(base_dir, "Dataset", "train.ft.txt")
    model_file = os.path.join(base_dir, "models", "sentiment_analysis_model.bin")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    train_model(train_file, model_file)