"""
Emotion Detection Chatbot - Flask app.

Loads the trained TF-IDF vectorizer + Naive Bayes model and uses them
to classify the emotion of whatever text a user submits.
"""

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

EMOTION_RESPONSES = {
    "happiness": "That's wonderful to hear! I'm glad things are going well. 😊",
    "sadness": "I'm sorry you're feeling this way. I'm here if you want to talk. 💙",
    "anger": "That sounds really frustrating. It's okay to feel that way. 😤",
    "neutral": "Got it, noted!",
}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("message", "")
        if user_text.strip():
            vector = vectorizer.transform([user_text])
            prediction = model.predict(vector)[0]
            probabilities = model.predict_proba(vector)[0]
            confidence = max(probabilities) * 100

            result = {
                "emotion": prediction,
                "confidence": round(confidence, 1),
                "response": EMOTION_RESPONSES.get(prediction, ""),
            }

    return render_template("index.html", result=result, user_text=user_text)


if __name__ == "__main__":
    app.run(debug=True)