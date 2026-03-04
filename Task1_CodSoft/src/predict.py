# for loading the trained model and vectorizer 
# from disk for making predictions.
import pickle 
from src.preprocess import clean_text

# Load model and vectorizer from disk. This allows us to use the trained model and vectorizer without having to retrain them every time we want to make a prediction.
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Function to predict genre based on input text. It first cleans the input text, then transforms it into a vector using the loaded vectorizer, and finally uses the loaded model to predict the genre.
def predict_genre(text):
    text = clean_text(text)
    text_vector = vectorizer.transform([text])
    genre = model.predict(text_vector)[0]
    return genre
