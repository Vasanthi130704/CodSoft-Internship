import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler 
# LabelEncoder is used to convert categorical variables into numerical format,
# while StandardScaler is used to standardize features by removing the mean and scaling to unit variance, which is important for many machine learning algorithms to perform well.

def load_data(path): # This function takes a file path as input and uses pandas to read a CSV file from that path, returning the resulting DataFrame. It is used to load the dataset for training and prediction.
    return pd.read_csv(path)

def preprocess_data(df): # This function takes a DataFrame as input and performs several preprocessing steps to prepare the data for machine learning. It drops unnecessary columns, encodes categorical variables, separates features and target variable, and scales the features using StandardScaler. It returns the processed features, target variable, fitted scaler, and the names of the feature columns.
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
    ], axis=1)

    # Encode categorical columns
    categorical_cols = ["merchant", "category", "gender", "job"]

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Features & target
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns