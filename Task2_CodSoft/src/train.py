import os
import pickle
import pandas as pd
import numpy as np

from math import radians, sin, cos, sqrt, atan2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


# HAVERSINE FUNCTION TO CALCULATE DISTANCE BETWEEN TWO POINTS ON EARTH
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "fraudTrain.csv")

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# Encode gender
df["gender"] = df["gender"].map({"M": 0, "F": 1})

#FEATURE ENGINEERING
# Distance using HAVERSINE formula
df["distance"] = df.apply(
    lambda row: haversine(
        row["lat"],
        row["long"],
        row["merch_lat"],
        row["merch_long"]
    ),
    axis=1
)

# Transaction hour
df["hour"] = pd.to_datetime(df["unix_time"], unit="s").dt.hour

# Selected Features
features = [
    "amt",
    "gender",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "unix_time",
    "distance",
    "hour"
]

X = df[features]
y = df["is_fraud"]

print("\nClass distribution BEFORE balancing:")
print(y.value_counts())


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Handle Imbalanced Data
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nClass distribution AFTER SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Model Training
print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train_smote)

# Model Evaluation
y_pred = model.predict(X_test_scaled)

print("\nModel Performance:")
print(classification_report(y_test, y_pred))

# Save Model
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

pickle.dump(model, open(os.path.join(MODEL_DIR, "model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb"))
pickle.dump(features, open(os.path.join(MODEL_DIR, "feature.pkl"), "wb"))

print("\n✅ Model trained and saved successfully!")