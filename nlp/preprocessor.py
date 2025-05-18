import pandas as pd
import re as regex
from bs4 import BeautifulSoup
import warnings
from bs4 import MarkupResemblesLocatorWarning

# Supprimer le warning BeautifulSoup
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def clean_text(text):
    if pd.isna(text):
        return ""

    # Supprimer HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # Supprimer les URLs
    text = regex.sub(r"http\S+|www\S+|https\S+", '', text)

    # Supprimer les emails
    text = regex.sub(r'\S+@\S+', '', text)

    # Supprimer caractères spéciaux
    text = regex.sub(r"[^a-zA-Z0-9\s]", '', text)

    # Espaces propres
    text = regex.sub(r"\s+", ' ', text)

    return text.lower().strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["subject", "body", "label"]].copy()
    df["subject"] = df["subject"].fillna("").apply(clean_text)
    df["body"] = df["body"].fillna("").apply(clean_text)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    return df


if __name__ == "__main__":
    input_path = "data/emails_merged.csv"
    output_path = "data/emails_cleaned.csv"

    try:
        df = pd.read_csv(input_path, low_memory=False)
        df_clean = clean_dataframe(df)
        df_clean.to_csv(output_path, index=False)
        print(f"✅ Données nettoyées sauvegardées dans {output_path} ({len(df_clean)} lignes)")
    except Exception as e:
        print(f"❌ Erreur lors du nettoyage : {e}")
