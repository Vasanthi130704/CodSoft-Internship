# import necessary libraries for data manipulation, model training, and evaluation. 
# We import a custom function clean_text from the preprocess module 
# to clean the text data before training the model.
import pandas as pd
# for saving the trained model and vectorizer to disk, which allows us to load 
# them later for making predictions without having to retrain the model every time.
import pickle 
# for converting text data into numerical vectors that can be used by machine learning models.
from sklearn.feature_extraction.text import TfidfVectorizer 
# for training a Support Vector Machine (SVM) model, 
# which is a popular machine learning algorithm for classification tasks.
# LinearSVC is faster than svc and works well for text classification tasks.
from sklearn.svm import LinearSVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# for cleaning the text data before training the model. 
# We are using the clean_text function from the preprocess module to remove unwanted characters, stop words, and perform other text preprocessing steps to improve the quality of the text data for training the model.
from src.preprocess import clean_text 

print("Reading dataset...")

data = pd.read_csv(
    "dataset/train_data.txt",
    sep=":::",
    engine="python",
    header=None
)

data.columns = ["ID","TITLE","GENRE","DESCRIPTION"]

# IMPORTANT CHANGE: combine title + description
data["text"] = data["TITLE"] + " " + data["DESCRIPTION"]

# Clean text
data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["GENRE"]

print("Converting text to numbers...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X = vectorizer.fit_transform(X)

# split the dataset into training and testing sets. 
# We use 80% of the data for training the model and 20% for testing its performance. 
# The random_state parameter is set to 42 to ensure that the split is reproducible, 
# meaning that the same split will be generated every time the code is run, 
# which is important for consistent evaluation of the model's performance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training SVM model...")
# Adjust C value for better performance since the dataset is relatively small. 
# A higher C value can help the model fit the training data better, 
# but it may also lead to overfitting. 
# It's important to find a balance that works well for the specific dataset and task at hand.
model = LinearSVC(C=1.5,class_weight='balanced') 

model.fit(X_train, y_train)

pred = model.predict(X_test)
print("New Accuracy:", accuracy_score(y_test, pred))

# save the trained model and vectorizer to disk using pickle. 
# This allows us to load the model and vectorizer later for making predictions without having to retrain the model every time.
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model saved! to model")
