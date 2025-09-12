# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Flask App --------------------
app = Flask(__name__)

# -------------------- Load Dataset --------------------
data = pd.read_csv("data_set.csv")

# Convert CSV into dictionary {question: [answers]}
qa_dict = {}
for _, row in data.iterrows():
    question = str(row["Question"]).strip().lower()
    answer = str(row["Answer"]).strip()
    if question in qa_dict:
        qa_dict[question].append(answer)
    else:
        qa_dict[question] = [answer]

questions = list(qa_dict.keys())

# -------------------- TF-IDF Model --------------------
# Use unigrams + bigrams to improve matching for short phrases
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
question_vectors = vectorizer.fit_transform(questions)

# -------------------- Routes --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_message = request.form.get("message", "").strip().lower()

    # Basic validation
    if len(user_message) < 1:
        return jsonify({"reply": "⚠️ Please type a message."})

    # 1) Exact (or near-exact) match first — useful for greetings/short phrases
    if user_message in qa_dict:
        reply = random.choice(qa_dict[user_message])
        print(f"[DEBUG] Exact match for '{user_message}' -> reply selected")
        return jsonify({"reply": reply})

    # 2) TF-IDF similarity fallback
    user_vector = vectorizer.transform([user_message])
    scores = cosine_similarity(user_vector, question_vectors)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    # Debug log (visible in Render logs / console)
    print(f"[DEBUG] User: '{user_message}'  BestMatch: '{questions[best_idx]}'  Score: {best_score:.4f}")

    # Threshold (tune as needed; TF-IDF scores are usually lower than embeddings)
    THRESHOLD = 0.25

    if best_score >= THRESHOLD:
        reply = random.choice(qa_dict[questions[best_idx]])
    else:
        # use CSV fallback row if exists, else default string
        reply = qa_dict.get("fallback", ["⚠️ Sorry, I didn’t understand that. Could you rephrase your question?"])[0]

    return jsonify({"reply": reply})

# -------------------- Run --------------------
if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
