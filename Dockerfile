FROM apache/airflow:2.7.1

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  vim \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
RUN rm -rf /home/airflow/.cache/pip
COPY requirements.txt /requirements.txt
ENV HF_HOME /opt/airflow/airflow_tmp/hf_cache

# Install CPU-only torch first to avoid downloading full CUDA version and hash errors
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install requirements with increased timeout to avoid ReadTimeoutError
RUN pip install --default-timeout=1000 --no-cache-dir -r /requirements.txt

# Ajout de l'installation du module 'regex' manquant (si non inclus dans requirements)
RUN pip install --default-timeout=1000 --no-cache-dir regex

# Download NLTK data
RUN python -m nltk.downloader punkt punkt_tab stopwords