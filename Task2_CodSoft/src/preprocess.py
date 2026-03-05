import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path):
    """
    Load dataset from CSV file
    """
    return pd.read_csv(path)


def preprocess_data(df):
    """
    Preprocess the fraud dataset:
    - Drop unnecessary columns
    - Encode categorical features
    - Create new fraud detection features
    - Scale features
    """

    # Drop unnecessary columns
    df = df.drop([
        "trans_date_trans_time",
        "cc_num",
        "first",
        "last",
        "street",
        "city",
        "state",
        "zip",
        "dob",
        "trans_num"
    ], axis=1, errors="ignore")

    # Feature Engineering
    # Distance between customer and merchant
    df["distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"])**2 +
        (df["long"] - df["merch_long"])**2
    )

    # Transaction hour
    df["hour"] = pd.to_datetime(df["unix_time"], unit="s").dt.hour

    # Encode categorical data
    categorical_cols = ["merchant", "category", "gender", "job"]

    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le


    # Features & target
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    feature_names = X.columns

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, feature_names, encoders
