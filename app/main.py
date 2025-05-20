import joblib


def load_model():
    return joblib.load("../model/best_model_comparatif.pkl")

def load_vectorizer():
    return joblib.load("../model/vectorizer_comparatif.pkl")

def predict_email(model, email_df):
    prediction = model.predict(email_df)
    proba = model.predict_proba(email_df)
    return prediction[0], proba[0]