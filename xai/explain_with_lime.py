import joblib
import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

# Charger le modèle et le vectorizer
model = joblib.load("model/best_model_comparatif.pkl")
vectorizer = joblib.load("model/vectorizer_comparatif.pkl")

# Créer un pipeline texte → vecteur → prédiction
pipeline = make_pipeline(vectorizer, model)

# Charger les données pour tester une explication
df = pd.read_csv("data/emails_cleaned.csv", low_memory=False)

# Nettoyer comme dans le modèle
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(["phishing", "spam", "legitimate", "ham", "safe email", "0", "1"])]
df["label"] = df["label"].replace({
    "phishing": 1, "spam": 1, "1": 1,
    "legitimate": 0, "ham": 0, "safe email": 0, "0": 0
}).astype(int)
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["text"] = df["subject"] + " " + df["body"]

# Prendre un exemple de phishing ou légitime
text_to_explain = df[df["label"] == 1]["text"].iloc[5]  # phishing
# text_to_explain = df[df["label"] == 0]["text"].iloc[0]  # légitime

# Créer l'explainer LIME
class_names = ["legitimate", "phishing"]
explainer = LimeTextExplainer(class_names=class_names)

# Expliquer la prédiction
exp = explainer.explain_instance(text_to_explain, pipeline.predict_proba, num_features=10)

# Afficher l’explication dans le navigateur
# exp.show_in_notebook()  # ou
exp.save_to_file("lime_explanation.html")

print("✅ Explication générée avec LIME")
