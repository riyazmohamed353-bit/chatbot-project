from flask import Flask, request, jsonify, render_template
import pandas as pd
from fuzzywuzzy import process

# Initialize Flask app
app = Flask(__name__)

# Load dataset
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

questions = list(qa_dict.keys())  # list of all questions

# Home route → loads index.html
@app.route("/")
def home():
    return render_template("index.html")

# Chatbot route
@app.route("/get", methods=["POST"])
def chatbot_response():
    user_message = request.form["message"].strip().lower()

    # Fuzzy match with threshold
    best_match, score = process.extractOne(user_message, questions)

    if score >= 70:  # adjust threshold if needed
        reply = qa_dict[best_match][0]  # pick first available answer
    else:
        reply = "❓ Sorry, I don't have info on that. Please contact the admin office."

    return jsonify({"reply": reply})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
