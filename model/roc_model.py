import joblib
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Charger les modèles sauvegardés
model = joblib.load("model/best_model_comparatif.pkl")
vectorizer = joblib.load("model/vectorizer_comparatif.pkl")

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
})
df["label"] = df["label"].infer_objects(copy=False).astype(int)

# Préparation du texte
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["text"] = df["subject"] + " " + df["body"]

X = df["text"]
y = df["label"]

# Transformation du texte en vecteurs
X_vectorized = vectorizer.transform(X)

# Prédictions de probabilités
y_proba = model.predict_proba(X_vectorized)[:, 1]

# Calcul des points de la courbe ROC
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

# Tracer la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Random Forest (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taux de faux positifs (False Positive Rate)")
plt.ylabel("Taux de vrais positifs (True Positive Rate)")
plt.title("Courbe ROC - Modèle Random Forest")
plt.legend(loc="lower right")
plt.grid()
plt.show()
