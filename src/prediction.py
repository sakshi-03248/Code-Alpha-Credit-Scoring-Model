
import joblib
import pandas as pd

# Load trained model
model = joblib.load(
    "CodeAlpha_Credit_Scoring_Model/models/credit_scoring_model.pkl"
)

print("Credit scoring model loaded successfully!")


def predict_loan(data):
    data = pd.DataFrame([data])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    return {
        "Prediction": int(prediction),
        "Probability": round(float(probability), 4)
    }
