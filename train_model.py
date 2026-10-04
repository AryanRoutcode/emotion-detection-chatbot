"""
Trains the Emotion Detection classifier.

Pipeline: TF-IDF vectorization -> train/test split -> Naive Bayes
classifier. Run this once to produce model.joblib and vectorizer.joblib,
which app.py loads to make live predictions.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from data import TRAINING_DATA

texts = [t for t, label in TRAINING_DATA]
labels = [label for t, label in TRAINING_DATA]

# Step 1: TF-IDF vectorization - turns each sentence into a vector of
# numbers, weighting distinctive words more heavily than common ones.
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)

# Step 2: train/test split - hold back 25% of the data purely for
# testing, so we evaluate on examples the model never trained on.
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

# Step 3: train a Multinomial Naive Bayes classifier - a standard,
# simple, well-understood choice for TF-IDF text classification.
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: evaluate on the held-out test set.
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Test accuracy: {accuracy:.2%}")
print()
print(classification_report(y_test, predictions))

# Step 5: save the trained model and vectorizer so app.py can load
# them without retraining every time.
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
print("Saved model.joblib and vectorizer.joblib")