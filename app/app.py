from flask import Flask, render_template, request, jsonify

app = Flask(__name__)



@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")



@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    email_object = data.get("email_object", "")
    email_text = data.get("email_text", "")

    result = {
        "label": "Phishing",
        "percent": 72
    }

    return jsonify(result)



if __name__ == "__main__":
    app.run(debug=True)
