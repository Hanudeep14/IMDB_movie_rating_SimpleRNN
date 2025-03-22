import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()
revrese_word_index = {value : key for key, value in word_index.items()}

model = load_model('simple_rnn__imdb.h5')

def decoded_review(review):
    return ' '.join([revrese_word_index.get(i-3, '?') for i in review])

def preprocess_review(review):
    review = review.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in review]
    return sequence.pad_sequences([encoded_review], maxlen = 500)

def predict_sentiment(review):
    preprocess_input = preprocess_review(review)
    prediction = model.predict(preprocess_input)
    return 'Positive'if prediction[0][0] > 0.5 else "Negative", prediction[0][0]

st.title('IMDB movie review using SimpleRNN')
st.write('Write a movie review')

user_input = st.text_area('Movie reviw')

if st.button('Classify'):
    processed_input = preprocess_review(user_input)
    prediction = model.predict(processed_input)
    st.write(f'Sentiment: {(prediction[0][0])} {"Positive" if prediction[0][0] > 0.5 else "Negative"}')
else:
    st.write('Please enter a review')