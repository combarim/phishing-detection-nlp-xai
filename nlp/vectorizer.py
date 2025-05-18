import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# === Configuration ===
INPUT_PATH = "data/emails_cleaned.csv"
VECTORIZER_PATH = "model/vectorizer.pkl"

def train_vectorizer(input_path=INPUT_PATH, save_path=VECTORIZER_PATH, max_features=5000):
    # Charger les données nettoyées
    df = pd.read_csv(input_path)

    # On suppose que le corps du mail est dans la colonne "body"
    texts = df["body"].fillna("")

    # Initialiser le vectorizer TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",  # ignore les mots comme "the", "is", etc.
        max_features=max_features  # limite le nombre de mots importants
    )

    # Apprentissage du vocabulaire
    X = vectorizer.fit_transform(texts)

    # Sauvegarder le vectorizer
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(vectorizer, save_path)

    print(f"✅ Vectorizer entraîné et sauvegardé dans {save_path}")
    return vectorizer

def load_vectorizer(path=VECTORIZER_PATH):
    return joblib.load(path)

def vectorize_text(texts, vectorizer):
    return vectorizer.transform(texts)

# === Lancement direct pour entraîner ===
if __name__ == "__main__":
    train_vectorizer()
