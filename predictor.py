import joblib
import numpy as np
import pandas as pd


class BreastCancerPredictor:
    def __init__(
        self,
        model_path="breast_cancer_model.pkl",
        feature_names_path="feature_names.pkl"
    ):
        """
        Load the full saved machine-learning pipeline.

        The pipeline already contains:
        - StandardScaler
        - Logistic Regression model

        Therefore, no separate scaler file is needed.
        """

        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(feature_names_path)

    def _validate_input(self, input_data):
        """
        Validate that the API receives exactly 30 valid numerical values.
        """

        if not isinstance(input_data, list):
            raise ValueError(
                "Input data must be a list containing 30 numerical values."
            )

        if len(input_data) != len(self.feature_names):
            raise ValueError(
                f"Input data must contain exactly "
                f"{len(self.feature_names)} features."
            )

        try:
            numeric_values = [float(value) for value in input_data]
        except (TypeError, ValueError):
            raise ValueError(
                "Every feature value must be numeric."
            )

        if not np.isfinite(numeric_values).all():
            raise ValueError(
                "Feature values cannot contain NaN or infinity."
            )

        # Column names and order must exactly match training data.
        input_df = pd.DataFrame(
            [numeric_values],
            columns=self.feature_names
        )

        return input_df

    def predict_single_case(self, input_data):
        """
        Predict breast-cancer diagnosis for one patient.
        """

        # 1. Validate raw incoming feature values
        input_df = self._validate_input(input_data)

        # 2. The saved pipeline scales input internally, then predicts.
        prediction = int(self.model.predict(input_df)[0])

        # 3. Get probability of the predicted class
        probabilities = self.model.predict_proba(input_df)[0]
        class_index = list(self.model.classes_).index(prediction)
        confidence = float(probabilities[class_index]) * 100

        # Dataset labels:
        # 0 = Malignant
        # 1 = Benign
        return {
            "diagnosis": "Benign" if prediction == 1 else "Malignant",
            "confidence": round(confidence, 2),
            "prediction_label": prediction
        }