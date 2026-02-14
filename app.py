from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import BreastCancerPredictor

app = Flask(__name__)
CORS(app)

predictor = BreastCancerPredictor()

@app.route("/")
def home():
    return jsonify({
        "message": "Breast Cancer Prediction API is running",
        "usage": {
            "POST /predict": "Send JSON with 30 features"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' field"}), 400

    try:
        result = predictor.predict_single_case(data["features"])
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception:
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)