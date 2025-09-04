import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

# -----------------------------
# 1. Load CSV
# -----------------------------
data = pd.read_csv("data_set.csv")  # Columns: User Query, Intent, Response

# -----------------------------
# 2. Train NLP model
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['User Query'])   # <-- fixed column name
y = data['Intent']

model = MultinomialNB()
model.fit(X, y)

# -----------------------------
# 3. Response logic
# -----------------------------
def get_response(user_input):
    text = user_input.lower().strip()
    X_test = vectorizer.transform([text])
    intent = model.predict(X_test)[0]

    # Pick a random response from matching intent
    responses = data[data['Intent'] == intent]['Response'].tolist()
    return random.choice(responses) if responses else "Sorry, I don't understand."

# -----------------------------
# 4. Flask Routes
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
