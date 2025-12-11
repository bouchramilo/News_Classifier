
import streamlit as st
import sys
import os
import joblib

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.embeddings import EmbeddingManager
from functions.entrainement import NewsClassifier
from functions.pretraitement_text import clean_text

# Load resources
@st.cache_resource
def load_resources():
    embedder = EmbeddingManager()
    classifier = NewsClassifier()
    # Check if model exists
    if os.path.exists('model.joblib'):
        classifier.load_model('model.joblib')
    else:
        st.warning("Model not found. Please train the model first.")
    return embedder, classifier

st.title("News Article Classifier")
st.write("Classify news articles into: World, Sports, Business, Sci/Tech")

embedder, classifier = load_resources()

text_input = st.text_area("Enter news article text:")

if st.button("Classify"):
    if text_input:
        # Preprocess
        cleaned_text = clean_text(text_input)
        
        # Embed
        embedding = embedder.generate_embeddings([cleaned_text])
        
        # Predict
        if hasattr(classifier.model, 'predict'):
            prediction = classifier.predict(embedding)[0]
            
            # Map label to name (ag_news labels: 0-World, 1-Sports, 2-Business, 3-Sci/Tech)
            labels = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
            result = labels.get(prediction, "Unknown")
            
            st.success(f"Category: {result}")
        else:
            st.error("Model is not trained/loaded.")
    else:
        st.warning("Please enter some text.")
