from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


class NewsClassifier:
    def __init__(self, modele=LogisticRegression(max_iter=1000)):
        self.model = modele
        
    def training(self, X_train, y_train):
        print(f"Training model {self.model}...")
        self.model.fit(X_train, y_train)
        print("Training complete.")
        
        
    # -----------------------------
    # Matrice de Confusion
    # -----------------------------
    def matrice_de_confusion(self, y_test, preds):
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matrice de Confusion - {self.model.__class__.__name__}")
        plt.xlabel("Prédictions")
        plt.ylabel("Réel")
        plt.show()



    def plot_roc_auc(self, X_test, y_test):

        # Vérifier predict_proba ou decision_function
        if hasattr(self.model, "predict_proba"):
            y_score = self.model.predict_proba(X_test)

        elif hasattr(self.model, "decision_function"):
            y_score = self.model.decision_function(X_test)

        else:
            print(f"⚠️ ROC-AUC non supporté pour le modèle {self.model.__class__.__name__}")
            return

        # Classes uniques
        classes = np.unique(y_test)

        # Cas binaire simple → ROC normal
        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
            auc_value = auc(fpr, tpr)

            plt.figure(figsize=(6,5))
            plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {self.model.__class__.__name__}")
            plt.legend()
            plt.grid(True)
            plt.show()
            return

        # -----------------------------------
        # MULTICLASS ROC (One-vs-Rest)
        # -----------------------------------

        # Binarisation des labels
        y_test_bin = label_binarize(y_test, classes=classes)

        plt.figure(figsize=(7,6))

        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            auc_value = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Classe {class_label} (AUC={auc_value:.3f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Multiclass - {self.model.__class__.__name__}")
        plt.legend()
        plt.grid(True)
        plt.show()


    # -----------------------------
    # Evaluation globale
    # -----------------------------
    def evaluate(self, X_test, y_test):
        print("\n", "+++"*50)
        print(f"Evaluating model {self.model}...")

        y_pred = self.model.predict(X_test)
        
        # Accuracy, Precision, Recall, F1
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Matrice de confusion
        self.matrice_de_confusion(y_test, y_pred)

        # ROC-AUC (si possible)
        print("\n📈 Vérification ROC-AUC...")
        self.plot_roc_auc(X_test, y_test)

        # Affichage des métriques
        print("Accuracy:", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1_Score:", f1)
        print("Classification Report:\n", classification_report(y_test, y_pred))

        return acc, precision, recall, f1
    

    # -----------------------------
    # Save Model
    # -----------------------------
    def save_model(self, path='../models/news_classifier.pkl'):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
