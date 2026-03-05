import os
import pickle

from src.preprocess import clean_text


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")


# Load model
model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))


def predict_sms(text):

    text = clean_text(text)

    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)[0]

    if prediction == 1:
        return "Spam"
    else:
        return "Ham"