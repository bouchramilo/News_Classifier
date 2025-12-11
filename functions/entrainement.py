from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report, auc
)
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np




class NewsClassifier:
    def __init__(self,model_name="LogisticRegression", modele=LogisticRegression(max_iter=1000), param_grid=None):
        self.model = Pipeline([
            ('clf', modele)
        ])
        self.model_name = model_name
        self.param_grid = param_grid
        self.best_model = None
        self.optimized = False
        
    
    # -------------------------------------------------------
    # Fonction d'optimisation : GRID SEARCH
    # -------------------------------------------------------
    def optimize(self, X_train, y_train, X_test, y_test, scoring="accuracy", cv=3):
        
        print(f"\n🔧 Optimisation des hyperparamètres : {self.model_name}")
        print("📌 Paramètres testés :", self.param_grid)

        grid = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        # meilleure pipeline
        self.model = grid.best_estimator_
        self.best_model = grid.best_estimator_
        self.optimized = True

        # évaluation du meilleur modèle
        acc, precision, recall, f1 = self.evaluate(X_test, y_test)

        print(f"Meilleurs paramètres : {grid.best_params_}")
        print(f"Score CV ({scoring}) : {grid.best_score_:.4f}")

        return self.best_model, acc, precision, recall, f1


    # -------------------------------------------------------
    # Fonction d'entraînement
    # -------------------------------------------------------
    def training(self, X_train, y_train):
        """
        Entraîne soit :
        - le modèle simple
        - le modèle optimisé si optimize() a été appelé
        """
        print(f"\n🚀 Entraînement du modèle : {self.model_name}")

        if self.optimized and self.best_model:
            print("👉 Utilisation du modèle optimisé.")
            self.best_model.fit(X_train, y_train)
        else:
            print("👉 Entraînement du modèle sans optimisation.")
            self.model.fit(X_train, y_train)
            self.best_model = self.model

        print("🎉 Entraînement terminé.")

        
        
    # -----------------------------
    # Matrice de Confusion
    # -----------------------------
    def matrice_de_confusion(self, y_test, preds):
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matrice de Confusion - {self.model_name}")
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
            print(f"⚠️ ROC-AUC non supporté pour le modèle {self.model_name}")
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
            plt.title(f"ROC Curve - {self.model_name}")
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
        plt.title(f"ROC Multiclass - {self.model_name}")
        plt.legend()
        plt.grid(True)
        plt.show()


    # -----------------------------
    # Evaluation globale
    # -----------------------------
    def evaluate(self, X_test, y_test):

        print(f"Evaluating model {self.model_name}...")

        y_pred = self.model.predict(X_test)
        
        # Accuracy, Precision, Recall, F1
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Matrice de confusion
        self.matrice_de_confusion(y_test, y_pred)

        # ROC-AUC
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
