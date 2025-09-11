from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data_set.csv")  # Columns: Question, Answer

questions = df["Question"].values
answers = df["Answer"].values

# Train NLP Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# -----------------------------
# Response Function
# -----------------------------
def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)

    idx = similarity.argmax()
    return answers[idx]

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.form["message"]
    bot_reply = get_response(user_input)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
