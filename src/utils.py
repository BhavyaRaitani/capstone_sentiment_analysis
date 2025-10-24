import os
def preview_file(file_path, n=5):
    """Preview the first n lines of a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for i in range(n):
            print(f.readline().strip())

if __name__ == "__main__":
    train_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset", "train.ft.txt"))
    preview_file(train_file)
    print()
    test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset", "test.ft.txt"))
    preview_file(test_file)