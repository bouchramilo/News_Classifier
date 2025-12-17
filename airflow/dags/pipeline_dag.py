"""
dags/news_pipeline.py

Airflow DAG to orchestrate the News classification pipeline:
1. load_data
2. preprocess
3. generate embeddings
4. train model
5. evaluate model
6. save & optionally upload model
"""

from datetime import datetime, timedelta
import os
import sys
import json
import logging
from pathlib import Path
import shutil
import numpy as np
import json

from airflow import DAG
from airflow.operators.python import PythonOperator

from airflow.models import Variable

# ---- Assure-toi que Airflow peut importer ton package functions ----
# Ajuste le chemin vers la racine de ton projet (où se trouve "functions")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Imports du projet (doivent exister)
from functions.data_loader import load_data
from functions.pretraitement_text import preprocess_dataframe
from functions.embeddings import EmbeddingManager
from functions.entrainement import NewsClassifier
import pandas as pd

# Config DAG
DAG_ID = "news_classification_pipeline"
DEFAULT_ARGS = {
    "owner": "you",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- Task functions ----------
def task_load_data(**context):
    """Charge les datasets et push via XCom les dataframes en pickle sur disk."""
    train_df, test_df = load_data()
    tmp_dir = os.path.join(PROJECT_ROOT, "airflow_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    train_path = os.path.join(tmp_dir, "train.pkl")
    test_path = os.path.join(tmp_dir, "test.pkl")
    train_df.to_pickle(train_path)
    test_df.to_pickle(test_path)
    context['ti'].xcom_push(key="train_pickle", value=train_path)
    context['ti'].xcom_push(key="test_pickle", value=test_path)
    logging.info(f"Saved train/test to {tmp_dir}")
    return {"train_path": train_path, "test_path": test_path}

# ----------
def task_preprocess(**context):
    train_path = context['ti'].xcom_pull(key="train_pickle")
    test_path  = context['ti'].xcom_pull(key="test_pickle")
    train_df = pd.read_pickle(train_path)
    test_df  = pd.read_pickle(test_path)

    train_df = preprocess_dataframe(train_df, text_column='text')
    test_df  = preprocess_dataframe(test_df, text_column='text')

    tmp_dir = os.path.join(PROJECT_ROOT, "airflow_tmp")
    pre_train = os.path.join(tmp_dir, "train_preprocessed.pkl")
    pre_test  = os.path.join(tmp_dir, "test_preprocessed.pkl")
    train_df.to_pickle(pre_train)
    test_df.to_pickle(pre_test)

    context['ti'].xcom_push(key="train_preprocessed", value=pre_train)
    context['ti'].xcom_push(key="test_preprocessed", value=pre_test)

    logging.info("Preprocessing done.")
    return True

# ----------
def task_embeddings(**context):
    import logging
    import os

    # 1. Récupération des datasets prétraités
    pre_train = context['ti'].xcom_pull(key="train_preprocessed")
    pre_test  = context['ti'].xcom_pull(key="test_preprocessed")

    train_df = pd.read_pickle(pre_train)
    test_df  = pd.read_pickle(pre_test)

    # 2. Initialisation de l'EmbeddingManager (ChromaDB + batch)
    embedder = EmbeddingManager(
        db_path=os.path.join(PROJECT_ROOT, "data", "chroma_db"),
        batch_size=32
    )

    # 3. Préparation des données
    train_texts = train_df["clean_text"].astype(str).tolist()
    test_texts  = test_df["clean_text"].astype(str).tolist()

    train_metadatas = [{"label": int(l), "split": "train"} for l in train_df["label"]]
    test_metadatas  = [{"label": int(l), "split": "test"} for l in test_df["label"]]

    train_ids = [f"train_{i}" for i in range(len(train_texts))]
    test_ids  = [f"test_{i}" for i in range(len(test_texts))]

    logging.info(f"Storing {len(train_texts)} train embeddings")
    logging.info(f"Storing {len(test_texts)} test embeddings")

    # 4. Stockage des embeddings dans ChromaDB
    embedder.store_embeddings(
        collection_name="train_collection",
        texts=train_texts,
        metadatas=train_metadatas,
        ids=train_ids
    )

    embedder.store_embeddings(
        collection_name="test_collection",
        texts=test_texts,
        metadatas=test_metadatas,
        ids=test_ids
    )

    # 5. Push XCom (signal de succès uniquement)
    context['ti'].xcom_push(key="embeddings_stored", value=True)

    logging.info("Embeddings successfully stored in ChromaDB.")
    return True

# ----------
def task_train(**context):
    embedder = EmbeddingManager(
        db_path=os.path.join(PROJECT_ROOT, "data", "chroma_db")
    )

    collection = embedder.get_collection("train_collection")
    data = collection.get(include=["embeddings", "metadatas"])

    X_train = np.array(data["embeddings"])
    y_train = np.array([m["label"] for m in data["metadatas"]], dtype=int)

    classifier = NewsClassifier()
    classifier.training(X_train, y_train)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_path = os.path.join(MODELS_DIR, f"news_classifier_{ts}.pkl")
    classifier.save_model(model_path)

    context['ti'].xcom_push(key="model_path", value=model_path)



# ----------
def task_cleanup(**context):
    tmp_dir = os.path.join(PROJECT_ROOT, "airflow_tmp", DAG_ID)

    keep_tmp = Variable.get(
        "KEEP_TMP_FILES",
        default_var="false"
    ).lower() == "true"

    if not keep_tmp and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logging.info(f"Temporary DAG files removed: {tmp_dir}")
    else:
        logging.info("Temporary files kept.")



# ---------- DAG definition ----------
with DAG(
    dag_id=DAG_ID,
    default_args=DEFAULT_ARGS,
    description="Pipeline to train/evaluate news classifier",
    schedule_interval="0 0 * * 1",  # minute heure jour_du_mois mois jour_de_la_semaine
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["news", "nlp", "training"]
) as dag:

    load_task = PythonOperator(
        task_id="load_data",
        python_callable=task_load_data,
        provide_context=True
    )

    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=task_preprocess,
        provide_context=True
    )

    embeddings_task = PythonOperator(
        task_id="generate_embeddings",
        python_callable=task_embeddings,
        provide_context=True
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=task_train,
        provide_context=True
    )

    cleanup_task = PythonOperator(
        task_id="cleanup",
        python_callable=task_cleanup,
        provide_context=True,
        trigger_rule="all_done"
    )

    # ----------
    load_task >> preprocess_task >> embeddings_task >> train_task >> cleanup_task
