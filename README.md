# 📰 News Article Classification Pipeline

## 📌 Description du projet

Ce projet consiste à concevoir et implémenter un **système intelligent de classification automatique d’articles d’actualité** en utilisant les techniques de **Natural Language Processing (NLP)** et de **Machine Learning**.  
L’objectif est de classer les articles en **quatre catégories stratégiques** : **World**, **Sports**, **Business** et **Sci/Tech**.

Le projet met en place une **pipeline NLP complète et automatisée**, depuis le chargement des données jusqu’au déploiement du modèle final dans une application **Streamlit**, avec une orchestration globale assurée par **Apache Airflow** et un stockage vectoriel via **ChromaDB**.

---

## 🚀 Fonctionnalités principales

- Chargement automatique du dataset **SetFit/ag_news** depuis Hugging Face  
- Analyse exploratoire des données (EDA)  
- Prétraitement avancé des textes (normalisation, nettoyage, suppression des stopwords, regex)  
- Génération d’embeddings avec **Sentence Transformers**  
- Stockage vectoriel des embeddings dans **ChromaDB** (train / test)  
- Entraînement et évaluation de modèles de Machine Learning  
- Vérification de l’overfitting avec plusieurs métriques  
- Orchestration complète du pipeline via **Airflow DAG**  
- Déploiement du modèle dans une interface interactive **Streamlit**  

---

## 🗂️ Structure du projet
```bach
News_Classifier/
├── Dockerfile # Image Docker pour containeriser le projet
├── docker-compose.yaml # Orchestration des services (Airflow, app, etc.)
├── README.md # Documentation du projet
├── requirements.txt # Dépendances Python
│
├── accueil.py # Page d’accueil Streamlit
├── pages/
│ └── prediction.py # Interface Streamlit pour la prédiction des articles
│
├── airflow/
│ └── dags/
│ └── pipeline_dag.py # DAG Airflow orchestrant toute la pipeline NLP
│
├── functions/
│ ├── data_loader.py # Chargement des données depuis Hugging Face
│ ├── analyse_exploratoire.py # Analyse exploratoire des données (EDA)
│ ├── pretraitement_text.py # Nettoyage et normalisation des textes
│ ├── embeddings.py # Génération des embeddings NLP
│ ├── entrainement.py # Entraînement et évaluation des modèles ML
│ └── pipeline.py # Pipeline globale (ETL + ML)
│
├── data/
│ ├── train/ # Base vectorielle ChromaDB (train)
│ ├── test/ # Base vectorielle ChromaDB (test)
│ └── chroma_db/ # Données persistées ChromaDB
│
├── models/
│ └── news_classifier.pkl # Modèle ML entraîné et sauvegardé
│
├── notebooks/
│ └── partie_1.ipynb # Exploration et tests initiaux
│
├── articles_test.ipynb # Tests de prédiction sur des articles
└── taches.ipynb # Suivi et organisation des tâches
```

---

## 🛠️ Technologies utilisées

- **Python**
- **Hugging Face Datasets**
- **Pandas / NumPy**
- **NLTK / Regex**
- **Sentence Transformers**
- **paraphrase-multilingual-MiniLM-L12-v2**
- **Scikit-learn**
- **ChromaDB (Vector Database)**
- **Apache Airflow**
- **Streamlit**
- **Docker & Docker Compose**
- **Jupyter Notebook**

---

## 🐳 Exécution du projet avec Docker

### ✅ Prérequis

- Docker  
- Docker Compose  

Vérifier l’installation :
```bash
docker --version
docker-compose --version
```

---

## ⚙️ Installation et exécution du projet

### 1️⃣ Cloner le dépôt

```bash
git clone https://github.com/bouchramilo/News_Classifier.git
cd News_Classifier
```

### 2️⃣ Construire et lancer les conteneurs

```bash
docker-compose up --build
```

Cette commande :

- construit l’image Docker,
- démarre Airflow (scheduler + webserver),
- initialise la pipeline NLP,
- rend l’application Streamlit accessible.

3️⃣ Accéder aux services
🔹 Apache Airflow

- URL : `http://localhost:8080`
- Activer le DAG : pipeline_dag

🔹 Application Streamlit

- URL : `http://localhost:8501`
- Permet de tester la classification d’articles en temps réel

4️⃣ Arrêter les conteneurs

```bash
docker-compose down
```

---
Fin 😊
