import os 
# The os module provides a way of using operating system dependent functionality, 
# such as reading or writing to the file system, which is essential for loading data and saving models 
import pickle 
# The pickle module is used for serializing and deserializing Python objects, 
# allowing us to save the trained model and scaler to disk for later use.
from sklearn.model_selection import train_test_split 
# train_test_split is a function from scikit-learn that splits the dataset into 
# training and testing sets, 
from sklearn.ensemble import RandomForestClassifier 
# RandomForestClassifier is a machine learning algorithm that builds 
# multiple decision trees and merges them together to get a more accurate and stable prediction.
from src.preprocess import load_data, preprocess_data
# These functions are imported from the preprocess module,
# which we created to handle data loading and preprocessing tasks.
import pandas as pd
# pandas is a powerful data manipulation library that provides data structures and functions

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATA_PATH = os.path.join(BASE_DIR, "dataset", "fraudTrain.csv")

print("Loading dataset...")

# Load only 20,000 rows instead of full dataset
df = pd.read_csv(DATA_PATH, nrows=500)

print("Dataset loaded:", df.shape)

# Preprocess
X, y, scaler, feature_names = preprocess_data(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
# Model folder path
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Ensure folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "features.pkl")

# Save files
pickle.dump(model, open(MODEL_PATH, "wb"))
pickle.dump(scaler, open(SCALER_PATH, "wb"))
pickle.dump(feature_names, open(FEATURE_PATH, "wb"))

print("✅ Model trained and saved successfully!")