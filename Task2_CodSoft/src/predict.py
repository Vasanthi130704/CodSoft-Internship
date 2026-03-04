import pickle 
# The pickle module is used for serializing and deserializing Python objects, 
#allowing us to save the trained model and scaler to disk for later use.
import numpy as np 
# NumPy is a fundamental package for scientific computing in Python,
# providing support for arrays, matrices, and a collection of mathematical functions to operate on these data

model = pickle.load(open("../model/model.pkl", "rb")) # Load the trained machine learning model from a file using pickle, allowing us to make predictions on new data.
scaler = pickle.load(open("../model/scaler.pkl", "rb")) # Load the fitted scaler object from a file using pickle, which will be used to preprocess new data in the same way as the training data.
feature_names = pickle.load(open("../model/features.pkl", "rb")) # Load the original feature names from a file using pickle, which will be used to ensure that the new data has the same structure as the training data when making predictions.

def predict_transaction(data_dict): # This function takes a dictionary of transaction data as input, preprocesses it to match the format of the training data, and uses the loaded model to make a prediction about whether the transaction is fraudulent or not.
    import pandas as pd 

    df = pd.DataFrame([data_dict]) # Convert the input dictionary into a pandas DataFrame
    df = pd.get_dummies(df) # Convert categorical variables into dummy/indicator variables, which is necessary for the model to process the data correctly.

    # Ensure same columns as training
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    scaled = scaler.transform(df)
    prediction = model.predict(scaled)

    return prediction[0] # Return the predicted class (0 for non-fraudulent, 1 for fraudulent) for the input transaction data.