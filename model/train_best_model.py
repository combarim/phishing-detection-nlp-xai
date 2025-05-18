import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Chargement des données
DATA_PATH = "data/emails_cleaned.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

# Nettoyage des labels
df["label"] = df["label"].astype(str).str.strip().str.lower()
valid_labels = ["phishing", "spam", "legitimate", "ham", "safe email", "0", "1"]
df = df[df["label"].isin(valid_labels)]

df["label"] = df["label"].replace({
    "phishing": 1, "spam": 1, "1": 1,
    "legitimate": 0, "ham": 0, "safe email": 0, "0": 0
}).astype(int)

# Préparation du texte
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["text"] = df["subject"] + " " + df["body"]

X = df["text"]
y = df["label"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Liste des modèles à tester
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Naive Bayes": MultinomialNB()
}

results = []
os.makedirs("confusion_matrices", exist_ok=True)

for name, model in models.items():
    print(f"\n--- Entraînement et évaluation du modèle : {name} ---")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("Rapport de classification:")
    print(classification_report(y_test, y_pred))

    # Sauvegarde des résultats
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        "model": name,
        "accuracy": acc,
        "precision_0": report["0"]["precision"],
        "recall_0": report["0"]["recall"],
        "f1_0": report["0"]["f1-score"],
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "f1_1": report["1"]["f1-score"],
    })

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Légitime", "Phishing"],
                yticklabels=["Légitime", "Phishing"])
    plt.title(f"Matrice de confusion - {name}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.tight_layout()

    # Sauvegarde de la figure
    fig_path = f"confusion_matrices/conf_matrix_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"✅ Matrice de confusion sauvegardée dans {fig_path}")

# Résumé des résultats
df_results = pd.DataFrame(results)
print("\n=== Comparaison des modèles ===")
print(df_results[["model", "accuracy", "precision_1", "recall_1", "f1_1"]].sort_values(by="f1_1", ascending=False))

# Sauvegarde du meilleur modèle
best_model_name = df_results.sort_values(by="f1_1", ascending=False).iloc[0]["model"]
print(f"\nMeilleur modèle selon le F1-score phishing : {best_model_name}")

best_model = models[best_model_name]
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/best_model_comparatif.pkl")
joblib.dump(vectorizer, "model/vectorizer_comparatif.pkl")
print("✅ Modèle et vectorizer sauvegardés dans 'model/best_model_comparatif.pkl' et 'model/vectorizer_comparatif.pkl'")
