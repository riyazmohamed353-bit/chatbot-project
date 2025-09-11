import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request, jsonify
import re

app = Flask(__name__)

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Make sure your CSV is named exactly 'dataset.csv' and Responses are quoted (") if they contain commas.
data = pd.read_csv("data_set.csv")  

# normalize intent labels
data['Intent'] = data['Intent'].str.lower().str.strip()

# -----------------------------
# 2. Train NLP Model
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['User Query'].astype(str))
y = data['Intent']

model = MultinomialNB()
model.fit(X, y)

# -----------------------------
# 3. Synonyms / Normalization
# -----------------------------
synonyms = {
    "canteen": "mess",
    "food court": "mess",
    "hostel charges": "hostel fee",
    "college fees": "college fee",
    "semester charges": "semester fee",
    "rcet": "rohini college",
    "rohini": "rohini college",
}

def normalize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    # replace synonyms as whole words
    for k, v in synonyms.items():
        text = re.sub(r"\b" + re.escape(k) + r"\b", v, text)
    # remove extra spaces
    text = re.sub(r"\s+", " ", text)
    return text

# -----------------------------
# 4. Response Logic
# -----------------------------
CONFIDENCE_THRESHOLD = 0.45  # tune between 0.35 - 0.6 as needed

def get_response(user_input: str) -> str:
    text = normalize(user_input)
    X_test = vectorizer.transform([text])

    probs = model.predict_proba(X_test)[0]
    intent_index = probs.argmax()
    confidence = probs[intent_index]
    intent = model.classes_[intent_index]

    if confidence < CONFIDENCE_THRESHOLD:
        return "Sorry, I’m not sure. Can you rephrase your question?"

    # Always return the first (clean) response for that intent (avoids duplicate/random answers)
    responses = data[data['Intent'] == intent]['Response'].tolist()
    return responses[0] if responses else "Sorry, I don't have information about that yet."

# -----------------------------
# 5. Flask Routes (UI)
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.form.get("message", "")
    reply = get_response(user_input)
    return jsonify({"reply": reply})

# -----------------------------
# 6. Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
