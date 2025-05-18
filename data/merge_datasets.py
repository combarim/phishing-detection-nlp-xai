import pandas as pd
import os

# Dossier des fichiers CSV
DATA_DIR = "data/raw"
OUTPUT_FILE = "data/emails_merged.csv"

# Liste des fichiers avec les colonnes à mapper
datasets_info = [
    {"filename": "CEAS_08.csv", "columns": {"subject": "subject", "body": "body", "label": "label"}},
    {"filename": "Enron.csv", "columns": {"subject": "subject", "body": "body", "label": "label"}},
    {"filename": "Ling.csv", "columns": {"subject": "subject", "body": "body", "label": "label"}},
    {"filename": "Nazario.csv", "columns": {"subject": "subject", "body": "body", "label": "label"}},
    {"filename": "Nigerian_Fraud.csv", "columns": {"subject": "subject", "body": "body", "label": "label"}},
    {"filename": "Phishing_Email.csv", "columns": {"Email Text": "body", "Email Type": "label"}, "add_subject": True},
    {"filename": "phishing_email_1.csv", "columns": {"text_combined": "body", "label": "label"}, "add_subject": True},
    {"filename": "SpamAssasin.csv", "columns": {"subject": "subject", "body": "body", "label": "label"}},
    {"filename": "enron_data_fraud_labeled.csv", "columns": {"Subject": "subject", "Body": "body", "Label": "label"}},
]

# Fusion des fichiers
all_dfs = []

for info in datasets_info:
    path = os.path.join(DATA_DIR, info["filename"])
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)

        # Ajouter la colonne 'subject' si nécessaire
        if info.get("add_subject"):
            df["subject"] = ""

        df = df.rename(columns=info["columns"])

        # On conserve uniquement les colonnes standardisées
        df = df[["subject", "body", "label"]]
        all_dfs.append(df)
        print(f"✅ Chargé : {info['filename']} ({len(df)} lignes)")

    except Exception as e:
        print(f"❌ Erreur avec {info['filename']}: {e}")

# Fusion et export
if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Fusion terminée : {len(merged_df)} emails enregistrés dans {OUTPUT_FILE}")
else:
    print("❗ Aucune donnée chargée.")
