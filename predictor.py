import joblib
import numpy as np


class BreastCancerPredictor:

    def __init__(self, model_path="breast_cancer_model.pkl", scaler_path="scaler.pkl"):
        # Load model and scaler ONCE
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def _validate_input(self, input_data):
        """
        Validates input data format and length
        """
        if not isinstance(input_data, list):
            raise ValueError("Input data must be a list of 30 numerical values.")

        if len(input_data) != 30:
            raise ValueError("Input data must contain exactly 30 features.")

        return np.array(input_data).reshape(1, -1)

    def predict_single_case(self, input_data):
        """
        Predicts diagnosis for a single patient
        """

        # 1️- Validate input
        input_array = self._validate_input(input_data)

        # 2️- Scale input using training scaler
        scaled_input = self.scaler.transform(input_array)

        # 3- Predict class
        prediction = self.model.predict(scaled_input)[0]

        # 4️- Predict probability
        probability = self.model.predict_proba(scaled_input)[0][prediction]

        # 5️- Return clean result
        return {
            "diagnosis": "Benign" if prediction == 1 else "Malignant",
            "confidence": round(probability * 100, 2),
            "prediction_label": int(prediction)
        }
