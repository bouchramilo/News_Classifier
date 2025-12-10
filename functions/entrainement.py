
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os

class NewsClassifier:
    def __init__(self, modele=LogisticRegression(max_iter=1000)):
        self.model = modele
        
    def training(self, X_train, y_train):
        print(f"Training model {self.model}...")
        self.model.fit(X_train, y_train)
        print("Training complete.")
        
    def evaluate(self, X_test, y_test):
        print("\n", "+++"*50)
        print(f"Evaluating model {self.model}...")
        y_pred = self.model.predict(X_test)        
        
        # Accuracy
        acc = accuracy_score(y_test, y_pred)

        # Precision, Recall, F1
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        # affichage des métriques
        print("Accuracy:", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1_Score:", f1)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        return acc, precision, recall, f1
    
    def save_model(self, path='../models/news_classifier.pkl'):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
        
        
    