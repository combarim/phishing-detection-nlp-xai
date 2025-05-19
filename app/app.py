import base64

from flask import Flask, render_template, request, jsonify, Response

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from main import predict_email, load_model, load_vectorizer
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline


app = Flask(__name__)
model = load_model()
vectorizer = load_vectorizer()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")



import os

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    email_object = data.get("email_object", "")
    email_text = data.get("email_text", "")

    email = email_object + " " + email_text
    email_df = pd.DataFrame([email], columns=["text"])
    email_vec = vectorizer.transform(email_df)

    result = predict_email(model, email_vec)
    prediction = "Email légitime" if result[0] == 1 else "Phishing"
    percent = result[1][result[0]] * 100

    pipeline = make_pipeline(vectorizer, model)
    class_names = ["Email légitime", "Phishing"]
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(email, pipeline.predict_proba, num_features=10)

    # Récupérer le HTML sous forme de chaîne de caractères
    html = exp.as_html()

    # Encodage base64 pour l'intégrer dans un data URI
    html_base64 = base64.b64encode(html.encode()).decode()

    return jsonify({
        "label": prediction,
        "percent": percent,
        "lime_html_base64": html_base64
    })


if __name__ == "__main__":
    app.run(debug=True)
