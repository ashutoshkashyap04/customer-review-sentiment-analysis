from fastapi import FastAPI
import pickle
import numpy as np
from pathlib import Path
from backend.schemas import ReviewSentiment

# for loading .keras model
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(
    title= 'Customer Review Sentiment Analysis API',
    description= 'API for predicting customer review ratings using a GRU model',
    version= '1.0.0'
)

#load saved objects
BASE_DIR = Path(__file__).resolve().parent.parent

model_path = hf_hub_download(
    repo_id = "ashutoshkashyap04/cutomer-review-sentiment-analysis",
    filename= "gru_model_v1.keras"
)
model = load_model(model_path)

with open(BASE_DIR / 'models' / 'tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    
# text preprocessing

MAX_LEN = 150

def preprocess_text(text : str) -> str:
    
        # convert to lowercase
    text = text.lower()
    
    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # remove html tags
    text = re.sub(r"<.*?>", "", text)
    
    # remove extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text
    

# root endpoint
@app.get('/')
def home():
    return {
        'Message' : 'Review Sentiment Analysis'
    }
    
    

# prediction endpoint

@app.post('/predict')
def predict(request: ReviewSentiment):
    
    # get review text
    review = request.review
    
    # preprocess review
    clean_review = preprocess_text(review)
    
    # tokenization
    sequence = tokenizer.texts_to_sequences([clean_review])
    
    # pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen = MAX_LEN,
        padding = 'post',
        truncating = 'post'
    )
    
    # make prediction
    prediction = model.predict(
        padded_sequence,
        verbose = 0
    )
    
    # get predicted class
    predicted_class = int(np.argmax(prediction[0]))   # returns the index of the largest value(probability)
    
    # get confidence
    confidence = float(prediction[0][predicted_class])
    
    # get the rating
    rating = predicted_class + 1
    
    return {
        'review' : review,
        'predicted_class' : predicted_class,
        'rating' : rating,
        'confidence'  : round(confidence, 4)
    }
