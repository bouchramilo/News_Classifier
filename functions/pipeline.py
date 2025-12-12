from functions.data_loader import load_data
from functions.pretraitement_text import preprocess_dataframe
from functions.embeddings import EmbeddingManager
from functions.entrainement import NewsClassifier
import pandas as pd

def run_full_pipeline():
    # 1. Load Data
    print("Step 1: Loading Data")
    train_df, test_df = load_data()

    # 2. Preprocess
    print("Step 2: Preprocessing")
    train_df = preprocess_dataframe(train_df, text_column='text')
    test_df = preprocess_dataframe(test_df, text_column='text')
    
    # 3. Embeddings & Storage
    print("Step 3: Generating and Storing Embeddings")
    embedder = EmbeddingManager()
    
    train_texts = train_df['clean_text'].tolist()
    train_metadatas = [{'label': int(l)} for l in train_df['label'].tolist()]
    train_ids = [f"train_{i}" for i in range(len(train_df))]
    
    X_train = embedder.generate_embeddings(train_texts)
    
    embedder.store_embeddings("train_collection", train_texts, train_metadatas, train_ids)
    
    # 4. Train
    print("Step 4: Training Model")
    classifier = NewsClassifier()
    classifier.training(X_train, train_df['label'])
    
    # 5. Evaluate
    print("Step 5: Evaluation")
    test_texts = test_df['clean_text'].tolist()
    X_test = embedder.generate_embeddings(test_texts)
    classifier.evaluate(X_test, test_df['label'])
    
    # 6. Save
    classifier.save_model()
    print("Pipeline Complete.")

if __name__ == "__main__":
    run_full_pipeline()
