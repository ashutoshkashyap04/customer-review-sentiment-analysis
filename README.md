# Customer Review Sentiment Analysis using RNN, LSTM & GRU

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)

## 📌 Project Overview

This project focuses on **multi-class sentiment classification** of customer reviews using Deep Learning. The primary objective is to compare the performance of three recurrent neural network architectures:

- Simple RNN
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

Rather than building a single model, this project demonstrates how different recurrent architectures learn textual information and how their ability to capture long-term dependencies affects sentiment classification performance.

The project follows a complete NLP workflow including:

- Data preprocessing
- Text tokenization
- Sequence padding
- Word Embedding
- Model training
- Performance evaluation
- Architecture comparison

---

## 🎯 Problem Statement

Given a customer review, predict its sentiment rating among **five different classes**.

| Rating | Sentiment |
|---------|-----------|
| 1 ⭐ | Very Negative |
| 2 ⭐⭐ | Negative |
| 3 ⭐⭐⭐ | Neutral |
| 4 ⭐⭐⭐⭐ | Positive |
| 5 ⭐⭐⭐⭐⭐ | Very Positive |

---

## 📂 Dataset

**Dataset:** Yelp Review Full Dataset

The dataset contains customer reviews with ratings from **1 to 5 stars**, making it a balanced multi-class sentiment classification problem.

For this project:

- Training Samples: **100,000**
- Test Samples: **20,000**
- Number of Classes: **5**

Each class contains an equal number of reviews.

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## 📁 Project Structure

```
customer-review-sentiment-analysis/
│
├── data/
│   |__ raw/
│   |__ processed/
│   |__ tokenized/
|       
├── models/
│   ├── rnn_model_v1.keras
│   ├── lstm_model_v1.keras
│   ├── gru_model_v1.keras
|   |__ lstm_history.pkl
|   |__ gru_history.pkl
│   └── tokenizer.pkl
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_tokenization.ipynb
│   ├── 05_rnn.ipynb
│   ├── 06_lstm.ipynb
│   └── 07_gru.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🔄 NLP Pipeline

```
Customer Reviews
        │
        ▼
Text Cleaning
        │
        ▼
Tokenization
        │
        ▼
Padding Sequences
        │
        ▼
Embedding Layer
        │
        ▼
RNN / LSTM / GRU
        │
        ▼
Softmax Output (5 Classes)
```

---

## 🧠 Models Implemented

### 1. Simple RNN

The baseline model for sequence classification.

Characteristics:

- Embedding layer
- Simple recurrent layer
- Dense output layer
- Softmax activation

---

### 2. LSTM

The LSTM architecture introduces memory cells and gating mechanisms, allowing the model to retain important contextual information over long sequences.

Advantages:

- Handles long-term dependencies
- Reduces vanishing gradient issues
- Better contextual understanding

---

### 3. GRU

GRU is a computationally efficient gated recurrent architecture.

Advantages:

- Fewer parameters than LSTM
- Faster training
- Strong sequence modeling capability
- Comparable performance to LSTM

---

# 📊 Model Performance

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 Score |
|--------|---------:|----------------:|-------------:|---------------:|
| Simple RNN | **26%** | 0.31 | 0.26 | 0.21 |
| LSTM | **57%** | 0.57 | 0.57 | 0.57 |
| GRU | **57%** | **0.58** | 0.57 | 0.57 |

---

# 📈 Class-wise Performance

## Simple RNN

| Class | Precision | Recall | F1 |
|------|----------:|-------:|---:|
| 0 | 0.48 | 0.21 | 0.30 |
| 1 | 0.32 | 0.04 | 0.07 |
| 2 | 0.24 | 0.17 | 0.20 |
| 3 | 0.27 | 0.10 | 0.14 |
| 4 | 0.23 | 0.78 | 0.36 |

---

## LSTM

| Class | Precision | Recall | F1 |
|------|----------:|-------:|---:|
| 0 | 0.68 | 0.68 | 0.68 |
| 1 | 0.49 | 0.55 | 0.52 |
| 2 | 0.50 | 0.49 | 0.49 |
| 3 | 0.51 | 0.50 | 0.50 |
| 4 | 0.71 | 0.62 | 0.66 |

---

## GRU

| Class | Precision | Recall | F1 |
|------|----------:|-------:|---:|
| 0 | 0.72 | 0.65 | 0.68 |
| 1 | 0.48 | 0.59 | 0.53 |
| 2 | 0.50 | 0.53 | 0.51 |
| 3 | 0.53 | 0.42 | 0.47 |
| 4 | 0.67 | 0.68 | 0.68 |


---

# 🔍 Key Observations

## 1. Simple RNN Performance

The Simple RNN struggled to learn meaningful sequential patterns from customer reviews.

### Key Findings

- Achieved only **26% test accuracy** on a 5-class classification problem.
- Macro F1 Score of **0.21** indicates poor performance across classes.
- The model became heavily biased toward predicting **Class 4**, achieving a recall of **0.78**, while failing to correctly identify several other classes.
- Class 1 had a recall of only **0.04**, meaning most negative reviews were misclassified.

### Interpretation

These results highlight one of the major limitations of vanilla recurrent neural networks. As review sequences become longer, Simple RNNs struggle to retain important contextual information due to the **vanishing gradient problem**, resulting in poor sequence understanding and highly imbalanced predictions.

---

## 2. LSTM Improvements over Simple RNN

Replacing the Simple RNN with an LSTM produced a significant improvement in overall performance.

### Performance Improvement

| Metric | Simple RNN | LSTM |
|--------|-----------:|------:|
| Accuracy | 26% | **57%** |
| Macro F1 Score | 0.21 | **0.57** |

The model achieved an improvement of **31 percentage points** in overall accuracy.

### Class-wise Improvements

Compared to the Simple RNN:

- Class 0 Recall improved from **0.21 → 0.68**
- Class 1 Recall improved from **0.04 → 0.55**
- Class 3 Recall improved from **0.10 → 0.50**

Unlike the Simple RNN, the LSTM produced much more balanced predictions across all sentiment classes.

### Interpretation

The gating mechanisms of LSTM enabled the model to preserve important contextual information throughout long review sequences, leading to substantially better feature learning and more reliable sentiment classification.

---

## 3. GRU vs LSTM

Although both models achieved the same overall accuracy (**57%**), the class-wise results reveal meaningful differences.

### Areas where GRU Improved

Compared with the LSTM:

- Class 1 Recall increased from **0.55 → 0.59**
- Class 2 Recall increased from **0.49 → 0.53**
- Class 4 Recall increased from **0.62 → 0.68**

Additionally, GRU achieved a slightly higher macro precision (**0.58**) compared to LSTM (**0.57**).

### Trade-offs

While GRU improved performance on several classes, it showed a slight reduction in recall for:

- Class 0 (**0.68 → 0.65**)
- Class 3 (**0.50 → 0.42**)

Overall, both architectures delivered nearly identical performance, with GRU demonstrating slightly better class-wise balance in some sentiment categories.

### Interpretation

These results suggest that a computationally simpler gated architecture like GRU can achieve performance comparable to LSTM while maintaining competitive generalization across multiple sentiment classes.

---

# 📊 Performance Summary

| Model | Accuracy | Key Insight |
|--------|---------:|-------------|
| Simple RNN | **26%** | Unable to capture long-term dependencies effectively. |
| LSTM | **57%** | Significant improvement through memory cells and gating mechanisms. |
| GRU | **57%** | Comparable performance to LSTM with slightly stronger class-wise balance for some sentiment categories. |

---

# 💡 Key Learnings

Throughout this project, I gained practical experience with:

- End-to-end NLP workflow
- Text preprocessing
- Tokenization and sequence padding
- Word embeddings
- Building recurrent neural networks using TensorFlow/Keras
- Training and evaluating deep learning models
- Comparing different recurrent architectures
- Understanding the impact of long-term dependency learning
- Evaluating models using Precision, Recall, F1 Score, and Confusion Matrix

---

# 🚀 Future Improvements

Potential enhancements for this project include:

- Bidirectional LSTM
- Bidirectional GRU
- Attention Mechanism
- Transformer-based architectures (BERT, RoBERTa)
- Hyperparameter tuning
- Pre-trained word embeddings (GloVe / FastText)
- Model deployment using FastAPI
- Docker containerization
- CI/CD pipeline
- MLOps integration

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ashutoshkashyap04/customer-review-sentiment-analysis.git
```

Navigate to the project directory:

```bash
cd customer-review-sentiment-analysis
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Run the notebooks in the following order:

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Tokenization & Padding
5. Simple RNN Model
6. LSTM Model
7. GRU Model

Each notebook is self-contained and documents the complete workflow for that stage of the project.

---

# 📌 Results

- Successfully built and compared three recurrent neural network architectures for multi-class sentiment classification.
- Improved test accuracy from **26%** using a Simple RNN to **57%** using gated recurrent architectures.
- Demonstrated the importance of LSTM and GRU in learning long-term dependencies within textual data.
- Performed detailed class-wise evaluation using Precision, Recall, and F1 Score to compare model behavior beyond overall accuracy.

---

# 🤝 Contributing

Contributions, suggestions, and improvements are always welcome.

If you find this project useful, consider giving it a ⭐ on GitHub.

---

# 👨‍💻 Author

**Ashutosh Kashyap**

---

