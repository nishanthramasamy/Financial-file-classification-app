#import necessary libraries
import os
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import re
from html import unescape
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import pickle
import streamlit as st


#NLP text pre-proceesing requirements:
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

#Function to clean and lemmatize the texts
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [lemmatizer.lemmatize(token) for token in text.split() if token not in stop_words]
    return ' '.join(tokens)

#Creating the streamlit homepage
st.title("File Classification App")
st.write("Upload a file and see its details!")

uploaded_file = st.file_uploader("Choose a file", type=['html'])

#Extracting texts from html file
if uploaded_file is not None:
    soup = BeautifulSoup(uploaded_file, 'html.parser')
    text = soup.get_text()
    match = re.search(r'<document_content>(.*?)</document_content>', text, re.DOTALL) # Check for html tags
    if match:
        text_content = match.group(1)
        cleaned_text = re.sub(r'<[^>]+>', '', text_content)
        cleaned_text = unescape(cleaned_text)
    else:
        cleaned_text = text
  


    
    processed_text = preprocess_text(cleaned_text)
    #loading the trained model: Also use Random forest model in the repo
    with open('logistic_regression.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
        
    
    output_class = int(loaded_model.predict([processed_text]))

    file_type = {
        0 : 'Balance Sheets',
        1 : 'Cash Flow',
        2 : 'Income Statement',
        3 : 'Notes',
        4 : 'Others'
    }

    outputs = file_type[output_class]
    st.write(f"The file belongs to the category: ",outputs)



        
