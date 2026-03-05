import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from src.preprocess import clean_text


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "spam.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_PATH, encoding="latin-1")

# Keep only needed columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']


# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# Clean messages
df['message'] = df['message'].apply(clean_text)


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)


# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Train model
model = MultinomialNB()

model.fit(X_train_vec, y_train)


# Evaluate
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# Save model
pickle.dump(model, open(os.path.join(MODEL_DIR, "model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb"))

print("Model saved successfully")