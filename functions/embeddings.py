


from sentence_transformers import SentenceTransformer
# Monkeypatch sqlite3 for ChromaDB compatibility
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

import pandas as pd

import os

class EmbeddingManager:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", db_path=None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, '..', 'data', 'chroma_db')
            
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        
    def generate_embeddings(self, texts):
        return self.model.encode(texts).tolist()
    
    def store_embeddings(self, collection_name, texts, metadatas, ids, batch_size=1000):

        collection = self.client.get_or_create_collection(name=collection_name)
        
        # Generation l'embeddings
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.generate_embeddings(texts)
        
        # Add to collection avec batch        
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            print(f"Processing batch {i//batch_size + 1} / {len(texts)//batch_size + 1}")

            embeddings = self.generate_embeddings(batch_texts)

            collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        print(f"Successfully stored {len(texts)} documents.")
    
    def get_collection(self, collection_name):
        return self.client.get_collection(name=collection_name)