from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("data_set.csv")

# Extract questions and answers
questions = df["Question"].values
answers = df["Answer"].values

# NLP model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"]

    # Transform query
    user_vec = vectorizer.transform([user_msg])
    similarity = cosine_similarity(user_vec, X)

    # Best match
    idx = similarity.argmax()
    response = answers[idx]

    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)
