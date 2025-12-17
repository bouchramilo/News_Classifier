from sentence_transformers import SentenceTransformer
import logging
import os
import numpy as np

# Monkeypatch sqlite3 for ChromaDB compatibility
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb


class EmbeddingManager:
    def __init__(
        self,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        db_path=None,
        batch_size=32
    ):
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, '..', 'data', 'chroma_db')

        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        self.batch_size = batch_size

        logging.info(f"EmbeddingManager initialized (batch_size={batch_size})")

    def generate_embeddings(self, texts):
        """Generate embeddings safely by batch"""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            logging.info(
                f"Embedding batch {i // self.batch_size + 1} / "
                f"{(len(texts) - 1) // self.batch_size + 1}"
            )

            emb = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.append(emb)

        return np.vstack(all_embeddings)

    def store_embeddings(
        self,
        collection_name,
        texts,
        metadatas,
        ids
    ):
        collection = self.client.get_or_create_collection(name=collection_name)

        logging.info(f"Storing {len(texts)} documents into ChromaDB")

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_metadatas = metadatas[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size]

            embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            collection.add(
                documents=batch_texts,
                embeddings=embeddings.tolist(),
                metadatas=batch_metadatas,
                ids=batch_ids
            )

            logging.info(
                f"Stored batch {i // self.batch_size + 1}"
            )

        logging.info("All embeddings successfully stored.")

    def get_collection(self, collection_name):
        return self.client.get_collection(name=collection_name)
