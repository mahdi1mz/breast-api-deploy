import os
import re
import shutil
from pathlib import Path
from difflib import SequenceMatcher

import cv2
import fitz
import numpy as np
import pytesseract
from pytesseract import Output

from feature_schema import FEATURE_ORDER


class ReportOCR:
    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}

    def __init__(self):
        self._configure_tesseract()

    def _configure_tesseract(self):
        """
        Works locally on Windows and later inside Render Docker.
        """
        windows_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        if os.path.exists(windows_tesseract):
            pytesseract.pytesseract.tesseract_cmd = windows_tesseract

        elif shutil.which("tesseract"):
            pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

        else:
            raise RuntimeError(
                "Tesseract OCR executable was not found on this system."
            )

    def extract(self, file_bytes: bytes, filename: str) -> dict:
        # Safety check: the model requires exactly 30 ordered features.
        if len(FEATURE_ORDER) != 30 or len(set(FEATURE_ORDER)) != 30:
            raise RuntimeError(
                "FEATURE_ORDER must contain exactly 30 unique feature names."
            )

        extension = Path(filename).suffix.lower()

        if extension not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                "Only PNG, JPG, JPEG, and PDF files are supported."
            )

        image = self._decode_file(file_bytes, extension)

        candidates = []

        for processed_image in self._create_ocr_variants(image):
            text = pytesseract.image_to_string(
                processed_image,
                config="--oem 3 --psm 6",
                lang="eng"
            )

            features, scores = self._parse_features(text)
            confidence = self._average_ocr_confidence(processed_image)

            candidates.append({
                "features": features,
                "scores": scores,
                "confidence": confidence,
                "text": text
            })

        # Choose the OCR result that extracted the most valid features.
        best = max(
            candidates,
            key=lambda item: (
                len(item["features"]),
                item["confidence"] if item["confidence"] is not None else 0
            )
        )

        missing_features = [
            feature for feature in FEATURE_ORDER
            if feature not in best["features"]
        ]

        warnings = []

        if best["confidence"] is not None and best["confidence"] < 70:
            warnings.append(
                "OCR confidence is low. Review every extracted value carefully."
            )

        if missing_features:
            warnings.append(
                f"{len(missing_features)} feature(s) were not detected."
            )

        if len(best["features"]) == 0:
            warnings.append(
                "No compatible WDBC-style features were detected. "
                "The uploaded report may use an unsupported format."
            )

        # Only create an API-ready ordered list when every required feature exists.
        if len(missing_features) == 0:
            ordered_features = [
                best["features"][feature]
                for feature in FEATURE_ORDER
            ]
        else:
            ordered_features = []

        return {
            "status": "needs_review",
            "detected_count": len(best["features"]),
            "all_features_detected": len(missing_features) == 0,

            # Named values: frontend uses these to autofill its manual fields.
            "features": best["features"],

            # Ordered values: same order required by your existing /predict route.
            "ordered_features": ordered_features,

            "feature_match_scores": best["scores"],
            "missing_features": missing_features,
            "ocr_average_confidence": best["confidence"],
            "warnings": warnings
        }

    def _decode_file(self, file_bytes: bytes, extension: str):
        """
        Converts an uploaded image or first PDF page into an OpenCV image.
        """
        if extension == ".pdf":
            document = fitz.open(stream=file_bytes, filetype="pdf")

            if document.page_count == 0:
                document.close()
                raise ValueError("The uploaded PDF contains no pages.")

            page = document.load_page(0)

            # Higher resolution improves OCR quality.
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2.5, 2.5),
                alpha=False
            )

            image_bytes = pixmap.tobytes("png")
            document.close()

            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        else:
            image_array = np.frombuffer(file_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not read the uploaded file.")

        return image

    def _create_ocr_variants(self, image):
        """
        Produces two versions because some documents work better in grayscale,
        while others work better after thresholding.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        _, thresholded = cv2.threshold(
            denoised,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        thresholded = cv2.copyMakeBorder(
            thresholded,
            20,
            20,
            20,
            20,
            cv2.BORDER_CONSTANT,
            value=255
        )

        return [gray, thresholded]

    def _parse_features(self, text: str):
        """
        Extracts every known feature directly from the full OCR text.

        Works even when a two-column report is read as one long line.
        Example:
        mean radius: 17.99 radius error: 1.095
        """
        normalized_text = self._normalize_text(text)

        extracted = {}
        match_scores = {}

        for feature in FEATURE_ORDER:
            label_pattern = re.escape(feature).replace(r"\ ", r"\s+")

            pattern = (
                rf"{label_pattern}"
                rf"\s*[:=\-]?\s*"
                rf"([-+]?(?:\d+(?:[.,]\d+)?|\.\d+))"
            )

            match = re.search(pattern, normalized_text, flags=re.IGNORECASE)

            if not match:
                continue

            raw_value = match.group(1).replace(",", ".")

            try:
                extracted[feature] = float(raw_value)
                match_scores[feature] = 1.0
            except ValueError:
                continue

        return extracted, match_scores

    @staticmethod
    def _normalize_text(value: str) -> str:
        value = value.lower()
        value = value.replace("—", "-").replace("–", "-")
        value = re.sub(r"[^a-z0-9.\- ]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @staticmethod
    def _similarity_score(text_a: str, text_b: str) -> float:
        return SequenceMatcher(
            None,
            text_a,
            text_b
        ).ratio()

    @staticmethod
    def _average_ocr_confidence(image):
        data = pytesseract.image_to_data(
            image,
            config="--oem 3 --psm 6",
            lang="eng",
            output_type=Output.DICT
        )

        confidence_values = []

        for word, confidence in zip(data["text"], data["conf"]):
            try:
                confidence = float(confidence)
            except ValueError:
                continue

            if word.strip() and confidence >= 0:
                confidence_values.append(confidence)

        if not confidence_values:
            return None

        return round(
            sum(confidence_values) / len(confidence_values),
            2
        )