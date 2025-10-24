import fasttext
import os

def print_results(N, p, r):
    print(f"Total Samples (N): {N}")
    print(f"Precision@1: {p:.3f}")
    print(f"Recall@1: {r:.3f}")
    print(f"Accuracy: {(p * 100):.2f}%")

def test_model(model_path, test_file):
    model = fasttext.load_model(model_path)
    results = model.test(test_file)
    print_results(*results)

    label_metrics = model.test_label(test_file)
    print("\nLabel-wise metrics:")
    for label, metrics in label_metrics.items():
        print(f"{label}: {metrics}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    test_file = os.path.join(base_dir, "Dataset", "test.ft.txt")
    model_file = os.path.join(base_dir, "models", "sentiment_analysis_model.bin")
    
    test_model(model_file, test_file)
