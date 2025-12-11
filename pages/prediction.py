
import streamlit as st
import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.embeddings import EmbeddingManager
from functions.entrainement import NewsClassifier
from functions.pretraitement_text import clean_text, tokenize_text, remove_stopwords

# ----------------------------------------------------
# Load modele
@st.cache_resource
def load_resources():
    embedder = EmbeddingManager()
    classifier = NewsClassifier()

    if os.path.exists('models/news_classifier.pkl'):
        classifier.load_model('models/news_classifier.pkl')
    else:
        st.warning("Model not found. Please train the model first.")
    return embedder, classifier

# ----------------------------------------------------
st.title("News Article Classifier")
st.write("Classify news articles into: World, Sports, Business, Sci/Tech")

# ----------------------------------------------------
embedder, classifier = load_resources()

text_input = st.text_area("Enter news article text:")

# ----------------------------------------------------
# prediction
if st.button("Classify"):
    if text_input:
        # Preprocess text
        cleaned_text = clean_text(text_input)
        tokened = tokenize_text(cleaned_text)
        no_stop = remove_stopwords(tokened)

        # embedding
        embedding = embedder.generate_embeddings([cleaned_text])
        
        # Predict
        if hasattr(classifier.model, 'predict'):
            prediction = classifier.predict(embedding)[0]
            
            labels = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
            result = labels.get(prediction, "Unknown")
            
            st.success(f"Category: {result}")
        else:
            st.error("Model is not trained/loaded.")
    else:
        st.warning("Please enter some text.")
