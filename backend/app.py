"""
app.py
------
Flask REST API for PolluSense AQI Prediction System.

Endpoints:
  POST /predict   — Forecast AQI for a city (1–7 days)
  GET  /cities    — List of supported cities
  GET  /health    — Health check

Run:
    python app.py
"""

import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

from model import SUPPORTED_CITIES, validate_city, validate_days
from predict import get_forecast_result, predict_single

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Allow all origins for local development (restrict in production)
CORS(app, resources={r"/*": {"origins": "*"}})


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok", "service": "PolluSense API"}), 200


@app.route("/cities", methods=["GET"])
def get_cities():
    """Return the list of supported cities."""
    return jsonify({"cities": SUPPORTED_CITIES}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Forecast AQI for a city over 1–7 days.

    Request body (JSON):
        {
            "city": "Delhi",
            "days": 7
        }

    Response (JSON):
        {
            "city": "Delhi",
            "predictions": [120.5, 135.2, ...],
            "categories":  ["Unhealthy", ...],
            "dates":       ["2024-04-20", ...],
            "alert":       "⚠️ UNHEALTHY: ...",   // or null
            "forecast": [
                {"day": 1, "date": "...", "aqi": 120.5, "category": "Unhealthy", "alert": null},
                ...
            ]
        }
    """
    # ── Parse & validate input ────────────────────────────────────────────────
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON."}), 400

    body = request.get_json(silent=True) or {}

    # City
    city_raw = body.get("city")
    if not city_raw or not isinstance(city_raw, str):
        return jsonify({"error": "'city' is required and must be a string."}), 400

    # Days
    days_raw = body.get("days")
    if days_raw is None:
        return jsonify({"error": "'days' is required."}), 400

    try:
        days = int(days_raw)
    except (TypeError, ValueError):
        return jsonify({"error": "'days' must be an integer."}), 400

    # Validate
    try:
        city = validate_city(city_raw)
        validate_days(days)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    # ── Run prediction ────────────────────────────────────────────────────────
    try:
        result = get_forecast_result(city, days)
        return jsonify(result), 200

    except FileNotFoundError as e:
        return jsonify({
            "error": "Model files not found. Please train the models first.",
            "detail": str(e)
        }), 503

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error during prediction.",
            "detail": str(e)
        }), 500


@app.route("/predict/single", methods=["POST"])
def predict_single_route():
    """
    Predict AQI for a specific city on a specific date.

    Request body (JSON):
        {
            "city": "Delhi",
            "date": "2023-12-15"
        }

    Response (JSON):
        {
            "city":     "Delhi",
            "date":     "2023-12-15",
            "aqi":      185.3,
            "category": "Unhealthy",
            "alert":    "⚠️ UNHEALTHY: ..."
        }
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON."}), 400

    body = request.get_json(silent=True) or {}
    city_raw = body.get("city")
    date_str = body.get("date")

    if not city_raw or not isinstance(city_raw, str):
        return jsonify({"error": "'city' is required."}), 400
    if not date_str or not isinstance(date_str, str):
        return jsonify({"error": "'date' is required (format: YYYY-MM-DD)."}), 400

    try:
        city = validate_city(city_raw)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    try:
        result = predict_single(city, date_str)
        return jsonify(result), 200

    except FileNotFoundError as e:
        return jsonify({
            "error": "Model files not found. Please train the models first.",
            "detail": str(e)
        }), 503

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error during prediction.",
            "detail": str(e)
        }), 500


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  PolluSense AQI Prediction API")
    print("  Running on: http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
