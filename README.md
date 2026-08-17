# Customer Review Sentiment Analysis using RNN, LSTM & GRU

A complete deep learning and NLP project for **5-class customer review sentiment classification**, comparing Simple RNN, LSTM, and GRU architectures and integrating the selected model into a web application using **FastAPI** and **Streamlit**.

---

## 📌 Project Overview

This project focuses on **multi-class sentiment classification of customer reviews using Deep Learning**.

The primary objective was not only to build a sentiment classifier, but also to understand how different recurrent neural network architectures handle sequential textual information.

Three architectures were implemented and compared:

* **Simple RNN**
* **LSTM (Long Short-Term Memory)**
* **GRU (Gated Recurrent Unit)**

After evaluating their performance, the **GRU model was selected as the final model** based on its competitive performance and class-wise behavior.

The project was then extended beyond model development by building:

* A **FastAPI backend** for model inference
* A **Streamlit frontend** for user interaction
* A complete inference pipeline using the trained tokenizer and GRU model

This makes the project a complete **end-to-end NLP deep learning application**, from data preprocessing and model development to API-based inference and user-facing deployment.

---

## 🎯 Problem Statement

Given a customer review, predict its sentiment rating among **five different classes**.

| Rating  | Sentiment     |
| ------- | ------------- |
| 1 ⭐     | Very Negative |
| 2 ⭐⭐    | Negative      |
| 3 ⭐⭐⭐   | Neutral       |
| 4 ⭐⭐⭐⭐  | Positive      |
| 5 ⭐⭐⭐⭐⭐ | Very Positive |

The model performs **5-class classification** rather than simply predicting positive or negative sentiment.

---

## 📂 Dataset

### Yelp Review Full Dataset

The project uses the **Yelp Review Full Dataset**, which contains customer reviews associated with ratings from **1 to 5 stars**.

For this project:

* **Training Samples:** 100,000
* **Test Samples:** 20,000
* **Number of Classes:** 5
* **Classes:** 1-star to 5-star reviews
* **Class Distribution:** Balanced

The original rating labels were mapped into five numerical classes:

```text
0 → 1 Star → Very Negative
1 → 2 Stars → Negative
2 → 3 Stars → Neutral
3 → 4 Stars → Positive
4 → 5 Stars → Very Positive
```

---

# 🔄 End-to-End Architecture

```text
                    CUSTOMER REVIEW
                           │
                           ▼
                  Text Preprocessing
                           │
                           ▼
                      Tokenization
                           │
                           ▼
                    Sequence Padding
                           │
                           ▼
                     Word Embedding
                           │
                           ▼
                 ┌─────────────────────┐
                 │   GRU Deep Learning │
                 │       Model         │
                 └─────────────────────┘
                           │
                           ▼
                  Softmax (5 Classes)
                           │
                           ▼
                  Sentiment Prediction
                           │
                           ▼
                    FastAPI Backend
                           │
                           ▼
                   Streamlit Frontend
                           │
                           ▼
                     User Interface
```

---

# 🧠 NLP Pipeline

The project follows a complete NLP preprocessing and modeling pipeline.

### 1. Text Preprocessing

Customer reviews are cleaned and transformed into a suitable format for the neural networks.

The preprocessing pipeline includes operations such as:

* Converting text to lowercase
* Removing unnecessary textual noise
* Preparing text for tokenization

### 2. Tokenization

The cleaned text is converted into numerical sequences using a tokenizer.

Each word is mapped to an integer index.

### 3. Sequence Padding

Since reviews have different lengths, the sequences are padded to a fixed maximum length.

```text
Review → Token Sequence → Padded Sequence
```

The project uses a maximum sequence length of **150 tokens**.

### 4. Word Embedding

The numerical sequences are passed through an embedding layer that learns dense vector representations of words.

### 5. Recurrent Neural Network

The embedded sequences are processed using:

* Simple RNN
* LSTM
* GRU

### 6. Classification

The final dense layer uses **Softmax activation** to produce probabilities across the five sentiment classes.

---

# 🧠 Models Implemented

## 1. Simple RNN

The Simple RNN was implemented as the baseline architecture.

### Characteristics

* Embedding layer
* Simple recurrent layer
* Dense output layer
* Softmax activation

The model struggled to maintain meaningful contextual information across longer sequences.

---

## 2. LSTM

LSTM introduces memory cells and gating mechanisms to better preserve important information over longer sequences.

### Advantages

* Handles long-term dependencies
* Reduces the impact of vanishing gradients
* Better contextual information retention
* More balanced classification compared with Simple RNN

---

## 3. GRU

GRU is a gated recurrent architecture designed to provide strong sequence modeling capabilities with a simpler structure than LSTM.

### Advantages

* Fewer parameters than LSTM
* Computationally efficient
* Strong sequence modeling capability
* Competitive performance with LSTM

The GRU model was selected as the **final model for inference and application integration**.

---

# 📊 Model Performance

| Model      | Accuracy | Macro Precision | Macro Recall | Macro F1 Score |
| ---------- | -------: | --------------: | -----------: | -------------: |
| Simple RNN |  **26%** |            0.31 |         0.26 |           0.21 |
| LSTM       |  **57%** |            0.57 |         0.57 |           0.57 |
| **GRU**    |  **57%** |        **0.58** |         0.57 |           0.57 |

The Simple RNN performed poorly, while both gated architectures showed a substantial improvement.

---

# 📈 Class-wise Performance

## Simple RNN

| Class | Precision | Recall |   F1 |
| ----- | --------: | -----: | ---: |
| 0     |      0.48 |   0.21 | 0.30 |
| 1     |      0.32 |   0.04 | 0.07 |
| 2     |      0.24 |   0.17 | 0.20 |
| 3     |      0.27 |   0.10 | 0.14 |
| 4     |      0.23 |   0.78 | 0.36 |

The model became heavily biased toward Class 4 and struggled to distinguish several of the remaining sentiment categories.

---

## LSTM

| Class | Precision | Recall |   F1 |
| ----- | --------: | -----: | ---: |
| 0     |      0.68 |   0.68 | 0.68 |
| 1     |      0.49 |   0.55 | 0.52 |
| 2     |      0.50 |   0.49 | 0.49 |
| 3     |      0.51 |   0.50 | 0.50 |
| 4     |      0.71 |   0.62 | 0.66 |

LSTM produced significantly more balanced predictions across the five sentiment classes.

---

## GRU

| Class | Precision |   Recall |   F1 |
| ----- | --------: | -------: | ---: |
| 0     |  **0.72** |     0.65 | 0.68 |
| 1     |      0.48 | **0.59** | 0.53 |
| 2     |      0.50 | **0.53** | 0.51 |
| 3     |      0.53 |     0.42 | 0.47 |
| 4     |      0.67 | **0.68** | 0.68 |

The GRU achieved slightly higher macro precision than LSTM and improved recall for several classes.

---

# 🔍 Key Observations

## Simple RNN

The Simple RNN achieved only **26% accuracy** and a **0.21 macro F1 score**.

It showed strong prediction bias toward Class 4 while performing poorly on several other classes.

For example:

* Class 1 recall: **0.04**
* Class 3 recall: **0.10**
* Class 4 recall: **0.78**

This demonstrates the difficulty of a vanilla RNN in preserving useful contextual information across longer sequences.

---

## LSTM Improvement

Replacing the Simple RNN with LSTM resulted in a major improvement.

| Metric   | Simple RNN |     LSTM |
| -------- | ---------: | -------: |
| Accuracy |        26% |  **57%** |
| Macro F1 |       0.21 | **0.57** |

Accuracy improved by **31 percentage points**.

Class-wise recall also improved substantially:

* Class 0: **0.21 → 0.68**
* Class 1: **0.04 → 0.55**
* Class 3: **0.10 → 0.50**

This demonstrates the advantage of gated recurrent architectures for sequence modeling.

---

## GRU vs LSTM

Both LSTM and GRU achieved **57% accuracy**, but their class-wise behavior differed.

GRU improved recall for:

* Class 1: **0.55 → 0.59**
* Class 2: **0.49 → 0.53**
* Class 4: **0.62 → 0.68**

GRU also achieved:

**Macro Precision: 0.58**

compared with:

**LSTM: 0.57**

However, GRU showed lower recall for:

* Class 0: **0.68 → 0.65**
* Class 3: **0.50 → 0.42**

Overall, the two gated architectures performed similarly, but GRU was selected as the final model because of its competitive performance and computationally simpler architecture.

---

# 🚀 From Model to Application

After completing the model comparison, the project was extended into a complete application.

The final system consists of three major components:

```text
                 ┌──────────────────┐
                 │ Streamlit        │
                 │ Frontend         │
                 └────────┬─────────┘
                          │
                          │ HTTP Request
                          ▼
                 ┌──────────────────┐
                 │ FastAPI          │
                 │ Backend          │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ GRU Model        │
                 │ + Tokenizer      │
                 └────────┬─────────┘
                          │
                          ▼
                 Sentiment Prediction
```

---

# ⚡ FastAPI Backend

The backend was developed using **FastAPI**.

Its responsibility is to:

1. Receive a customer review from the frontend.
2. Validate the incoming request.
3. Preprocess the review.
4. Tokenize the text.
5. Apply sequence padding.
6. Pass the sequence to the trained GRU model.
7. Generate the predicted sentiment.
8. Return the prediction to the frontend.

### Backend Structure

```text
backend/
├── app.py
└── schemas.py
```

### `app.py`

Contains the FastAPI application, model loading, preprocessing logic, and prediction endpoint.

### `schemas.py`

Contains the Pydantic request schema used to validate incoming review data.

---

# 🖥️ Streamlit Frontend

The user interface was built using **Streamlit**.

The frontend provides a simple interface where users can enter a customer review and receive the predicted sentiment.

The frontend communicates with the FastAPI backend through an HTTP request.

### Frontend Structure

```text
frontend/
└── app.py
```

The separation between frontend and backend makes the application architecture more modular and easier to maintain.

---

# 🔗 Inference Workflow

When a user submits a review:

```text
User enters review
        │
        ▼
Streamlit Frontend
        │
        │ POST Request
        ▼
FastAPI `/predict`
        │
        ▼
Text Preprocessing
        │
        ▼
Tokenizer
        │
        ▼
Sequence Padding
        │
        ▼
GRU Model
        │
        ▼
Predicted Class
        │
        ▼
Sentiment Mapping
        │
        ▼
Streamlit displays result
```

---

# 📁 Project Structure

The current repository structure is:

```text
customer-review-sentiment-analysis/
│
├── backend/
│   ├── app.py
│   └── schemas.py
│
├── frontend/
│   └── app.py
│
├── models/
│   ├── gru_history.pkl
│   ├── lstm_history.pkl
│   └── tokenizer.pkl
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_text_preprocessing.ipynb
│   ├── 04_tokenization.ipynb
│   ├── 05_rnn.ipynb
│   ├── 06_lstm.ipynb
│   └── 07_gru.ipynb
│
├── .gitignore
├── .python-version
├── requirements.txt
└── README.md
```

The repository currently contains dedicated backend and frontend directories, seven notebooks covering the modeling workflow, model-support artifacts, and the Python version configuration.

> **Note:** The trained `.keras` model files are managed separately from the lightweight source-code structure because of their large file sizes.

---

# 🗂️ Notebook Workflow

The notebooks are organized sequentially to document the development process.

| Notebook                      | Purpose                           |
| ----------------------------- | --------------------------------- |
| `01_data_loading.ipynb`       | Load and inspect the dataset      |
| `02_eda.ipynb`                | Exploratory data analysis         |
| `03_text_preprocessing.ipynb` | Clean and preprocess review text  |
| `04_tokenization.ipynb`       | Tokenization and sequence padding |
| `05_rnn.ipynb`                | Build and evaluate Simple RNN     |
| `06_lstm.ipynb`               | Build and evaluate LSTM           |
| `07_gru.ipynb`                | Build and evaluate GRU            |

This structure keeps the experimentation and model development process separate from the application code.

---

# 🛠️ Tech Stack

### Programming & Development

* Python 3.13
* Jupyter Notebook

### Data & NLP

* NumPy
* Pandas
* NLTK
* Scikit-learn

### Deep Learning

* TensorFlow
* Keras

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Frontend

* Streamlit

### Visualization

* Matplotlib
* Seaborn

The repository currently pins its Python ecosystem dependencies, including TensorFlow, Keras, FastAPI, Streamlit, Uvicorn, NLTK, NumPy, Pandas, and Scikit-learn.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/ashutoshkashyap04/customer-review-sentiment-analysis.git
```

Navigate into the project:

```bash
cd customer-review-sentiment-analysis
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The repository specifies Python **3.13** through the `.python-version` file.

---

# ▶️ Running the Application

The application consists of two components:

1. FastAPI backend
2. Streamlit frontend

Both need to be running for the complete application workflow.

---

## 1. Start the FastAPI Backend

From the project root:

```bash
uvicorn backend.app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

---

## 2. Start the Streamlit Frontend

Open another terminal and activate the same virtual environment.

Then run:

```bash
streamlit run frontend/app.py
```

Streamlit will provide a local URL where the application can be accessed through a web browser.

---

# 🔌 API Endpoint

The primary prediction endpoint is:

```text
POST /predict
```

The endpoint accepts a customer review and returns the predicted sentiment.

### Example Request

```json
{
    "review": "The product was excellent and I really enjoyed the experience."
}
```

### Example Response

```json
{
        'review' : 'The product was excellent and I really enjoyed the experience.',
        'predicted_class' : 3,
        'rating' : 4,
        'confidence'  : 0.85
}
```


---

# 📊 Sentiment Mapping

The model predicts one of five numerical classes.

| Model Class |  Rating | Sentiment     |
| ----------: | ------: | ------------- |
|           0 |     1 ⭐ | Very Negative |
|           1 |    2 ⭐⭐ | Negative      |
|           2 |   3 ⭐⭐⭐ | Neutral       |
|           3 |  4 ⭐⭐⭐⭐ | Positive      |
|           4 | 5 ⭐⭐⭐⭐⭐ | Very Positive |

This mapping converts the model's numerical output into an interpretable customer-facing result.

---

# 💡 Key Learnings

This project provided practical experience across both **Deep Learning and ML deployment**.

### NLP & Deep Learning

* End-to-end NLP workflow
* Text preprocessing
* Tokenization
* Sequence padding
* Word embeddings
* Recurrent neural networks
* LSTM architecture
* GRU architecture
* Long-term dependency learning
* Multi-class classification

### Model Evaluation

* Accuracy
* Precision
* Recall
* Macro F1 Score
* Class-wise performance analysis
* Confusion matrix analysis
* Architecture comparison

### Backend Development

* FastAPI application development
* REST API design
* Pydantic request validation
* Model inference through API endpoints
* Uvicorn server

### Frontend Development

* Streamlit application development
* User input handling
* API communication
* Displaying model predictions

### Deployment-Oriented Engineering

* Separating frontend and backend
* Managing deep learning model artifacts
* Dependency management
* Python version compatibility
* Preparing a deep learning model for application inference

---

# 📌 Results

The project successfully demonstrates the progression from a basic recurrent architecture to a deployable deep learning application.

### Model Development

* Simple RNN achieved **26% accuracy**
* LSTM achieved **57% accuracy**
* GRU achieved **57% accuracy**
* GRU achieved the highest macro precision at **0.58**

### Application Development

The final project includes:

* A trained sentiment classification pipeline
* GRU-based inference
* FastAPI backend
* Streamlit frontend
* Tokenizer artifact
* Request validation
* Five-class sentiment prediction

This transforms the project from a notebook-based experiment into a complete **NLP deep learning application**.

---

# 🚀 Future Improvements

The current version represents **Version 1** of the project.

Possible future improvements include:

### Model Architecture

* Bidirectional LSTM
* Bidirectional GRU
* Stacked LSTM
* Stacked GRU
* Attention mechanisms

### Advanced NLP

* Pre-trained word embeddings
* GloVe
* FastText
* Transformer-based architectures
* BERT
* RoBERTa

### Model Optimization

* Hyperparameter tuning
* Improved regularization
* Learning-rate optimization
* Better sequence-length analysis

### Application & MLOps

* Docker containerization
* CI/CD pipeline
* Automated model testing
* Model versioning
* Monitoring
* Logging
* MLOps integration

---

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

If you find this project useful, consider giving it a ⭐ on GitHub.

---

# 👨‍💻 Author

**Ashutosh Kashyap**

GitHub:
https://github.com/ashutoshkashyap04

---

## 🔗 Repository

Complete source code, notebooks, model artifacts, backend, and frontend:

https://github.com/ashutoshkashyap04/customer-review-sentiment-analysis


