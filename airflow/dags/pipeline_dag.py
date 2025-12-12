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

# Where to save models locally (relative to project root)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def _log_and_return(context, msg):
    logging.info(msg)
    return msg

# ---------- Task functions ----------
def task_load_data(**context):
    """Charge les datasets et push via XCom les dataframes en pickle sur disk."""
    train_df, test_df = load_data()  # renvoie pandas DataFrame
    # sauvegarde temporaire (pickle) afin d'éviter XCom lourds
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


def task_embeddings(**context):
    pre_train = context['ti'].xcom_pull(key="train_preprocessed")
    pre_test  = context['ti'].xcom_pull(key="test_preprocessed")
    train_df = pd.read_pickle(pre_train)
    test_df  = pd.read_pickle(pre_test)

    embedder = EmbeddingManager()

    train_texts = train_df["clean_text"].astype(str).tolist()
    test_texts  = test_df["clean_text"].astype(str).tolist()

    X_train = embedder.generate_embeddings(train_texts)
    X_test  = embedder.generate_embeddings(test_texts)

    # Sauvegarde des embeddings (npy)
    tmp_dir = os.path.join(PROJECT_ROOT, "airflow_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    train_emb_path = os.path.join(tmp_dir, "X_train.npy")
    test_emb_path  = os.path.join(tmp_dir, "X_test.npy")
    np.save(train_emb_path, X_train)
    np.save(test_emb_path, X_test)

    context['ti'].xcom_push(key="X_train_path", value=train_emb_path)
    context['ti'].xcom_push(key="X_test_path", value=test_emb_path)

    # push also labels locations
    y_train = train_df['label'].astype(int).tolist()
    y_test  = test_df['label'].astype(int).tolist()
    y_train_path = os.path.join(tmp_dir, "y_train.json")
    y_test_path  = os.path.join(tmp_dir, "y_test.json")
    with open(y_train_path, "w") as f:
        json.dump(y_train, f)
    with open(y_test_path, "w") as f:
        json.dump(y_test, f)
    context['ti'].xcom_push(key="y_train_path", value=y_train_path)
    context['ti'].xcom_push(key="y_test_path", value=y_test_path)

    logging.info("Embeddings generated and saved.")


def task_train(**context):
    X_train_path = context['ti'].xcom_pull(key="X_train_path")
    y_train_path = context['ti'].xcom_pull(key="y_train_path")

    X_train = np.load(X_train_path)
    with open(y_train_path, "r") as f:
        y_train = json.load(f)

    # instantiate classifier (tu peux remplacer par param_grid si besoin)
    classifier = NewsClassifier()

    # train
    classifier.training(X_train, np.array(y_train, dtype=int))

    # save trained model with timestamped filename
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_filename = f"news_classifier_{ts}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    classifier.save_model(model_path)

    # push path to XCom for next tasks
    context['ti'].xcom_push(key="model_path", value=model_path)
    logging.info(f"Model trained and saved to {model_path}")


def task_evaluate(**context):
    X_test_path = context['ti'].xcom_pull(key="X_test_path")
    y_test_path = context['ti'].xcom_pull(key="y_test_path")
    model_path   = context['ti'].xcom_pull(key="model_path")

    X_test = np.load(X_test_path)
    with open(y_test_path, "r") as f:
        y_test = json.load(f)

    # load model and evaluate
    classifier = NewsClassifier()
    loaded_ok = classifier.load_model(model_path)
    if not loaded_ok:
        raise RuntimeError(f"Model not found at {model_path}")

    classifier.evaluate(X_test, np.array(y_test, dtype=int))
    logging.info("Evaluation completed.")


def task_cleanup(**context):
    # Optionnel: supprime fichiers temporaires ou conserve selon Variable
    tmp_dir = os.path.join(PROJECT_ROOT, "airflow_tmp")
    keep_tmp = Variable.get("KEEP_TMP_FILES", default_var="false").lower() == "true"
    if not keep_tmp and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        logging.info("Temporary files removed.")
    else:
        logging.info("Temporary files kept.")


# ---------- DAG definition ----------
with DAG(
    dag_id=DAG_ID,
    default_args=DEFAULT_ARGS,
    description="Pipeline to train/evaluate news classifier",
    schedule_interval=None,      # manual trigger; change to cron if souhaité
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

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate,
        provide_context=True
    )

    cleanup_task = PythonOperator(
        task_id="cleanup",
        python_callable=task_cleanup,
        provide_context=True
    )

    # dependencies
    load_task >> preprocess_task >> embeddings_task >> train_task >> evaluate_task >> cleanup_task
