from flask import Flask, request, jsonify, render_template
import pandas as pd
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
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# -------------------- Routes --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_message = request.form["message"].strip().lower()

    # Reject very short or nonsense input
    if len(user_message) < 3:
        return jsonify({"reply": "⚠️ Please ask a complete question."})

    # Convert user query into TF-IDF vector
    user_vector = vectorizer.transform([user_message])

    # Compute cosine similarity
    scores = cosine_similarity(user_vector, question_vectors)[0]
    best_idx = scores.argmax()
    best_score = scores[best_idx]

    # Check similarity threshold
    if best_score >= 0.3:  # lower threshold for TF-IDF
        reply = qa_dict[questions[best_idx]][0]  # take first answer
    else:
        reply = (
            "❓ Sorry, I don't have info on that.\n"
            "👉 Try asking about college details."
        )

    return jsonify({"reply": reply})

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(debug=True)
