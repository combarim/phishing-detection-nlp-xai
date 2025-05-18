import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# 📂 Chargement des données UNE SEULE FOIS
DATA_PATH = "data/emails_cleaned.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

# 🧹 Nettoyage et conversion des labels (bien avant tout autre traitement)
df["label"] = df["label"].astype(str).str.strip().str.lower()

# ⚠️ Garde uniquement les labels que tu veux gérer
valid_labels = ["phishing", "spam", "legitimate", "ham", "safe email", "0", "1"]
df = df[df["label"].isin(valid_labels)]

# 🔁 Conversion des labels en binaire
df["label"] = df["label"].replace({
    "phishing": 1, "spam": 1, "1": 1,
    "legitimate": 0, "ham": 0, "safe email": 0, "0": 0
}).astype(int)  # assure-toi bien que c'est int

# Affiche la distribution finale (doit être uniquement 0 et 1)
print("Répartition des classes après conversion :")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True) * 100)

# 📨 Préparation du texte (subject + body)
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["text"] = df["subject"] + " " + df["body"]

# 🎯 Séparation des variables
X = df["text"]
y = df["label"]

# 🧪 Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Nombre d'exemples dans le train : {X_train.shape[0]}")
print(f"Nombre d'exemples dans le test : {X_test.shape[0]}")

# ✨ Vectorisation TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🧠 Entraînement du modèle
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 📈 Évaluation
y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Rapport de classification:\n", classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Légitime", "Phishing"], yticklabels=["Légitime", "Phishing"])
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de confusion")
plt.show()

# Courbe ROC + AUC
y_score = model.predict_proba(X_test_vec)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="darkorange")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC")
plt.legend(loc="lower right")
plt.show()

# Courbe Precision-Recall
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_score)
plt.figure(figsize=(7,6))
plt.plot(recall, precision, color="blue")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Courbe Precision-Recall")
plt.show()

# 💾 Sauvegarde du modèle et du vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("✅ Modèle et vectorizer sauvegardés.")
