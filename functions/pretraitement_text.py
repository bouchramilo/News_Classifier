import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Downloads moved inside functions/Dockerfile to avoid DAG parsing overhead

# ! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower() # miniscule
    text = re.sub(r'http\S+|www\S+', '', text)  # URLs
    text = re.sub(r'\S+@\S+', '', text)  # emails
    text = re.sub(r'\d+', '', text)  # chiffres
    text = text.translate(str.maketrans('', '', string.punctuation))  # ponctuation
    text = re.sub(r'\s+', ' ', text).strip() # espaces
    
    return text
    
    
# ! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def tokenize_text(text):
    # Ensure necessary data is present (handled by Dockerfile mostly, but safe fallback)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
         nltk.download('punkt', quiet=True)
         nltk.download('punkt_tab', quiet=True)
         
    return word_tokenize(text)

# ! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def remove_stopwords(tokens):
    # Lazy load stopwords to check efficiently
    if not hasattr(remove_stopwords, 'stop_set'):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        remove_stopwords.stop_set = set(stopwords.words('english'))
        
    return [word for word in tokens if word not in remove_stopwords.stop_set]


# ! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def preprocess_dataframe(df, text_column='text'):
    
    # nettoyer le text
    df[f'clean_{text_column}'] = df[text_column].apply(clean_text)
    
    # supprimer les doublons 
    df = df.dropna(subset=[f'clean_{text_column}'])
    df[f'clean_{text_column}'] = df[f'clean_{text_column}'].str.strip()
    df = df[df[f'clean_{text_column}'] != ""]
    
    # Tokeniser le texte
    df['text_tokenized'] = df[f'clean_{text_column}'].apply(tokenize_text)
    
    # supprimer les stop words
    df['text_no_stopwords'] = df['text_tokenized'].apply(remove_stopwords)

    return df
