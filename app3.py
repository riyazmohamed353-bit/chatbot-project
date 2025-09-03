from flask import Flask, render_template, request, jsonify
import csv, random, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# -----------------------------
# 1. Load CSV
# -----------------------------
queries = []
intents = []
responses_by_intent = {}

with open("data_set.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        queries.append(row["User Query"].lower())
        intents.append(row["Intent"])
        
        if row["Intent"] not in responses_by_intent:
            responses_by_intent[row["Intent"]] = []
        responses_by_intent[row["Intent"]].append(row["Response"])

# -----------------------------
# 2. Train NLP model
# -----------------------------
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(queries, intents)

# -----------------------------
# 3. Rule + ML Response
# -----------------------------
def get_response(user_input):
    text = user_input.lower().strip()

    # --- Rule-based keywords ---
    if "fee" in text:
        return random.choice(responses_by_intent.get("college_fees", ["Fees info not found"]))
    if "semester" in text:
        return random.choice(responses_by_intent.get("semester_fees", ["Semester fees not found"]))
    if "hi" in text or "hello" in text:
        return random.choice(responses_by_intent.get("greeting", ["Hello!"]))

    # --- Else use ML ---
    intent = model.predict([text])[0]
    return random.choice(responses_by_intent[intent])

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
