import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request, jsonify
import random
import re

app = Flask(__name__)

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("data_set.csv")

# Normalize intent labels
data['Intent'] = data['Intent'].str.lower().str.strip()

# -----------------------------
# 2. Train NLP Model
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['User Query'])
y = data['Intent']

model = MultinomialNB()
model.fit(X, y)

# -----------------------------
# 3. Synonyms Dictionary
# -----------------------------
synonyms = {
    "canteen": "mess",
    "food court": "mess",
    "dining": "mess",
    "hostel charges": "hostel fee",
    "college fees": "college fee",
    "semester charges": "semester fee",
    "rohini": "rohini college",
    "r c e t": "rohini college",
}

def normalize(text):
    """Lowercase + synonym replacement"""
    text = text.lower()
    for k, v in synonyms.items():
        text = re.sub(r"\b" + re.escape(k) + r"\b", v, text)
    return text.strip()

# -----------------------------
# 4. Response Logic
# -----------------------------
def get_response(user_input):
    norm_text = normalize(user_input)
    X_test = vectorizer.transform([norm_text])

    # Predict intent probabilities
    probs = model.predict_proba(X_test)[0]
    intent_index = probs.argmax()
    confidence = probs[intent_index]
    intent = model.classes_[intent_index]

    # Confidence threshold
    if confidence < 0.45:
        return "Sorry, I’m not sure. Can you rephrase your question about Rohini College?"

    # Get response for the predicted intent
    responses = data[data['Intent'] == intent]['Response'].tolist()
    if responses:
        return random.choice(responses)
    else:
        return "Sorry, I don't have information about that yet."

# -----------------------------
# 5. Flask Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")  # Create index.html inside /templates/

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.form.get("message", "")
    bot_reply = get_response(user_input)
    return jsonify({"reply": bot_reply})

# -----------------------------
# 6. Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
