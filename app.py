from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from pytesseract import TesseractNotFoundError

from predictor import BreastCancerPredictor
from report_ocr import ReportOCR


app = Flask(__name__)
CORS(app)

# Maximum upload size: 8 MB
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

predictor = BreastCancerPredictor()
report_ocr = ReportOCR()


@app.route("/")
def home():
    return jsonify({
        "message": "Breast Cancer Prediction API is running",
        "usage": {
            "POST /predict": "Send JSON with 30 features",
            "POST /extract-report": "Upload a compatible JPG, PNG, or PDF report"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "features" not in data:
        return jsonify({
            "error": "Missing 'features' field"
        }), 400

    try:
        result = predictor.predict_single_case(data["features"])
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            "error": "Internal server error"
        }), 500


@app.route("/extract-report", methods=["POST"])
def extract_report():
    if "report" not in request.files:
        return jsonify({
            "error": "Missing file. Upload it using the 'report' field."
        }), 400

    uploaded_file = request.files["report"]

    if uploaded_file.filename == "":
        return jsonify({
            "error": "No file selected."
        }), 400

    try:
        result = report_ocr.extract(
            uploaded_file.read(),
            uploaded_file.filename
        )

        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except TesseractNotFoundError:
        return jsonify({
            "error": "Tesseract OCR is not installed or could not be found."
        }), 500

    except Exception as e:
        print(f"OCR error: {e}")
        return jsonify({
            "error": "Could not process the uploaded report."
        }), 500


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return jsonify({
        "error": "File is too large. Maximum allowed size is 8 MB."
    }), 413


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)