from flask import Flask, request, jsonify, render_template
import pandas as pd
from sentence_transformers import SentenceTransformer, util

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

# -------------------- Embedding Model --------------------
# Small, fast model (downloads first time)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings for dataset questions
question_embeddings = model.encode(questions, convert_to_tensor=True)

# -------------------- Routes --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_message = request.form["message"].strip()

    # Reject very short or nonsense input
    if len(user_message) < 3:
        return jsonify({"reply": "⚠️ Please ask a complete question."})

    # Encode user query
    user_embedding = model.encode(user_message, convert_to_tensor=True)

    # Compute cosine similarity
    scores = util.cos_sim(user_embedding, question_embeddings)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    # Check similarity threshold
    if best_score >= 0.75:
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
