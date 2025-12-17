FROM apache/airflow:2.9.1-python3.10

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    sqlite3 \
    libsqlite3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

ENV HF_HOME=/opt/airflow/airflow_tmp/hf_cache
ENV NLTK_DATA=/home/airflow/nltk_data

COPY requirements.txt /requirements.txt

# Torch CPU
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --default-timeout=1000 --no-cache-dir -r /requirements.txt
RUN pip install --no-cache-dir regex

# Download NLTK resources
RUN mkdir -p /home/airflow/nltk_data \
 && python -m nltk.downloader -d /home/airflow/nltk_data punkt punkt_tab stopwords
