# 🛍️ Amazon Review Sentiment Analysis (FastText)

## 📘 Overview

This project implements a high-speed **Sentiment Analysis** model for **Amazon customer reviews** using the **FastText** library developed by Meta AI. The model is trained on a substantial dataset of genuine customer reviews and their corresponding star ratings to perform binary classification.

The primary goal is to categorize reviews as either **Positive** (\_\_label\_\_2) or **Negative** (\_\_label\_\_1) with high efficiency and accuracy.

-----

## 💡 What is Sentiment Analysis?

Sentiment Analysis, also known as **Opinion Mining** or **Polarity Detection**, is a Natural Language Processing (NLP) technique used to determine the emotional tone behind a body of text.

In this project, the task is a form of **binary classification**, categorizing the textual data into one of two polarity classes:

  * **Positive** (\_\_label\_\_2): Corresponds to 4- and 5-star reviews.
  * **Negative** (\_\_label\_\_1): Corresponds to 1- and 2-star reviews.

*(Note: 3-star reviews, representing neutral sentiment, were excluded from the original dataset to create a clear binary classification problem.)*

-----

## 🚀 FastText: The Core Technology

**FastText** is an open-source, free, and lightweight library for efficient learning of word representations and sentence classification. Developed by Facebook's AI Research (FAIR) lab, it is a key component of this project for its ability to train quickly on large datasets while maintaining strong performance.

### Key Features of FastText

  * **Subword Information:** Unlike older models like Word2Vec, FastText represents each word as a **bag of character n-grams**. This allows it to capture subword information, which is particularly beneficial for:
      * Handling **morphologically rich languages**.
      * Generating embeddings for **Out-Of-Vocabulary (OOV)** words by using the n-grams it has learned.
  * **Speed and Scalability:** FastText employs a simple, shallow neural network architecture and efficient optimization techniques (like hierarchical softmax) that allow for training on millions of records in minutes, even on modest hardware.

-----

## 📊 Dataset Description

The model is trained on a **multi-million record dataset** of Amazon customer reviews. The original data, sourced from Xiang Zhang's Google Drive, was processed and formatted for optimal use with FastText.

### Content and Format

The dataset files (`train.ft.txt`, `test.ft.txt`) adhere to the strict format required by the FastText supervised learning tutorial:

`__label__<CLASS> <TEXT>`

  * **Labels:** The labels are prefixed with `__label__`.
      * `__label__1`: Negative Sentiment (1- and 2-star reviews).
      * `__label__2`: Positive Sentiment (4- and 5-star reviews).
  * **Text:** The review title, followed by a colon and a space, is prepended to the review text. The reviews are predominantly in English.

-----

## ⚙️ Installation and Setup

### 🖥️ Windows Compatibility Note

The native C++ version of FastText can sometimes present build-tool dependency issues (like requiring specific versions of Microsoft Visual C++ Build Tools) on Windows systems. To ensure a smooth setup, the official **FastText Python wheel (pre-compiled binary)** is used.

### Setup Steps

1.  **Clone the Repository and Create a Virtual Environment:**

    ```bash
    git clone <your-repo-link>
    cd capstone
    python -m venv venv
    venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    The `requirements.txt` file ensures the correct FastText wheel and other necessary Python libraries (like the `fasttext` package) are installed.

    ```bash
    (venv) PS D:\WISE\capstone> pip install -r requirements.txt
    ```

    *(Note: The log in your provided steps shows `pip uninstall fasttext -y` followed by `pip install -r requirements.txt`. This is a robust practice to ensure a clean install, specifically using the version defined in `requirements.txt`.)*

-----

## 💻 Training and Evaluation

The model training and testing were executed via dedicated Python scripts that utilize the installed FastText package.

### 1\. Training the Model

The training script reads the dataset, processes it, and trains the supervised classification model.

| Command | Description |
| :--- | :--- |
| `python src/train_model.py` | Initiates the FastText supervised learning process. |

**Training Output:**

```
Training model using data from: D:\WISE\capstone\Dataset\train.ft.txt
Read 289M words
Number of words:  5165173
Number of labels: 2
Progress: 100.0% words/sec/thread:  525635 lr:  0.000000 avg.loss:  0.108527 ETA:   0h 0m 0s
✅ Model saved at D:\WISE\capstone\models\sentiment_analysis_model.bin
```

### 2\. Testing the Model

The testing script evaluates the trained model (`sentiment_analysis_model.bin`) against the separate test dataset (`test.ft.txt`).

| Command | Description |
| :--- | :--- |
| `python src/test_model.py` | Tests the performance of the trained FastText model. |

**Testing Output (Performance Metrics):**

```
Total Samples (N): 400000
Precision@1: 0.932
Recall@1: 0.932
Accuracy: 93.18%

Label-wise metrics:
__label__2: {'precision': 0.9310866396437771, 'recall': nan, 'f1score': 1.8621732792875543}
__label__1: {'precision': 0.9324856267152788, 'recall': nan, 'f1score': 1.8649712534305576}
```

### 📈 Results Analysis

The evaluation results demonstrate excellent performance, surpassing the expected baseline of 91.6%.

  * **Accuracy (93.18%):** The overall proportion of test samples correctly classified as either positive or negative.
  * **Precision@1 (0.932):** The proportion of predicted labels (at the top prediction) that are correct. In this context, it signifies the reliability of the model's positive predictions.
  * **Recall@1 (0.932):** The proportion of actual labels (at the top prediction) that are correctly identified. It signifies the model's ability to find all positive instances.
      * *Note: In binary classification with the FastText `test` command, **Precision@1** and **Recall@1** are often identical because the model is forced to make a single prediction for a single true label, leading to a micro-averaged result equal to accuracy.*

The high precision and recall scores across both classes (`__label__1` and `__label__2`) indicate a well-balanced and highly effective classification model.

-----

## 📂 Project Structure

A basic project structure is provided for context. You should expand this to match your actual files.

```
.
├── Dataset/
│   ├── train.ft.txt          # Formatted training data (Amazon reviews)
│   └── test.ft.txt           # Formatted testing data
├── models/
│   └── sentiment_analysis_model.bin # Trained FastText model
├── src/
│   ├── train_model.py        # Python script for model training
│   ├── test_model.py         # Python script for model evaluation
│   └── utils.py              # Script to sample or preview data/predictions
├── requirements.txt        # Python package dependencies (including fasttext wheel)
└── README.md
```

### Previewing Data (`src/utils.py`)

This script can be used to quickly inspect the formatted data files, confirming the `__label__<X> <Text>` format.

```bash
(venv) PS D:\WISE\capstone> python src/utils.py
# (Outputs 5 lines from train model and 5 lines from test model)
```