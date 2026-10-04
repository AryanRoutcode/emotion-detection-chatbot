# Emotion Detection Chatbot

A text-based emotion classifier that detects **happiness**, **sadness**, **anger**, or **neutral** tone from a sentence, using a classic NLP pipeline, wrapped in a simple Flask web app.

## How it works

1. **TF-IDF Vectorization** — converts input text into a numeric vector, weighting distinctive words more heavily than common ones.
2. **Multinomial Naive Bayes classifier** — trained on the vectorized text to predict one of four emotions.
3. **Flask web app** — lets a user type a sentence and see the predicted emotion with a confidence score.

## Tech stack

- Python
- scikit-learn (TF-IDF, train/test split, Naive Bayes, evaluation metrics)
- Flask
- joblib (model persistence)

## Project structure
emotion-chatbot/
├── generate_data.py # Builds the training dataset
├── data.py # Generated training data (created by generate_data.py)
├── train_model.py # Trains the TF-IDF + Naive Bayes model
├── app.py # Flask app serving live predictions
├── templates/
│ └── index.html # Simple web form UI
├── model.joblib # Saved trained model
├── vectorizer.joblib # Saved TF-IDF vectorizer
└── requirements.txt


## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

python generate_data.py      # generates data.py
python train_model.py        # trains and saves the model
python app.py                # runs the web app at http://127.0.0.1:5000
```

## A note on the data 

The training data in this repo is **synthetically generated** using sentence templates combined with word pools per emotion (see `generate_data.py`) — it is not scraped or collected from real users. This was a deliberate choice for a focused rebuild of an earlier project whose original dataset no longer exists.

Because of this, the model's test accuracy (commonly ~95%+ on the held-out test set) partly reflects shared structure between the generated train and test sentences, rather than performance on genuinely diverse, real-world text. On freshly hand-written sentences outside the training templates, predictions are still generally correct, but with more realistic, lower confidence scores.

## Possible improvements

- Replace synthetic data with a real, larger, human-labeled emotion dataset
- Try additional classifiers (Logistic Regression, SVM) for comparison
- Expand beyond 4 emotion categories
