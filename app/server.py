import os
import sys
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS

# -------------------------------------------------------------------
# Ensure project root is in Python path
# -------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -------------------------------------------------------------------
# Import internal modules
# -------------------------------------------------------------------
from models.hybrid_model import build_hybrid
from preprocessing.preprocess import preprocess_input_data

# -------------------------------------------------------------------
# Flask App Setup
# -------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------------------------------
# File Paths
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_hybrid_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# -------------------------------------------------------------------
# Global Variables
# -------------------------------------------------------------------
model = None
scaler = None


# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def load_resources():
    """Load model and scaler, or initialize fallback versions."""
    global model, scaler

    # Load or initialize model
    if os.path.exists(MODEL_PATH):
        print("[INFO] Loading trained model...")
        model = load_model(MODEL_PATH)
    else:
        print("[WARN] No trained model found. Initializing new hybrid model...")
        model = build_hybrid(seq_len=128, channels=4)

    # Load or warn about missing scaler
    if os.path.exists(SCALER_PATH):
        print("[INFO] Loading scaler...")
        scaler = joblib.load(SCALER_PATH)
    else:
        print("[WARN] Scaler not found. Predictions may be less accurate.")


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({"message": "Medical Intrusion Detection API is running."}), 200


@app.route("/detect", methods=["POST"])
def detect():
    """Predict intrusion from incoming medical sensor data."""
    global model, scaler

    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' in request body."}), 400

        # -------------------------------------------------------
        # Step 1: Parse & reshape input
        # -------------------------------------------------------
        features = np.array(data["features"], dtype=float)
        if features.ndim == 1:
            channels = 4
            seq_len = len(features) // channels
            features = features.reshape(1, seq_len, channels)
        elif features.ndim == 2:
            seq_len, channels = features.shape
            features = features.reshape(1, seq_len, channels)
        elif features.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected input shape {features.shape}")

        print(f"[INFO] Input reshaped to {features.shape}")

        # -------------------------------------------------------
        # Step 2: Preprocessing
        # -------------------------------------------------------
        try:
            features = preprocess_input_data(features)
        except Exception as e:
            print("[WARN] Preprocessing skipped:", e)

        # -------------------------------------------------------
        # Step 3: Scaling
        # -------------------------------------------------------
        if scaler is not None:
            flat = features.reshape(features.shape[0], -1)
            expected = scaler.mean_.shape[0]
            actual = flat.shape[1]
            if expected == actual:
                flat_scaled = scaler.transform(flat)
                features = flat_scaled.reshape(features.shape)
            else:
                print(f"[WARN] Skipping scaling (expected {expected}, got {actual}).")
        else:
            print("[WARN] Proceeding without scaling.")

        # -------------------------------------------------------
        # Step 4: Pad or truncate to (128,4)
        # -------------------------------------------------------
        seq_len, channels = features.shape[1], features.shape[2]
        if seq_len < 128:
            pad_len = 128 - seq_len
            features = np.pad(features, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
        elif seq_len > 128:
            features = features[:, :128, :]

        # -------------------------------------------------------
        # Step 5: Predict
        # -------------------------------------------------------
        preds = model.predict(features, verbose=0)
        pred_val = preds[0][0] if preds.ndim > 1 else preds[0]
        confidence = float(pred_val)
        result = "attack" if confidence > 0.5 else "normal"
        print(f"[INFO] Prediction: {result} (conf={confidence:.3f})")

        # -------------------------------------------------------
        # Step 6: SHAP explainability (optional lightweight)
        # -------------------------------------------------------
        try:
            import shap
            background = np.repeat(features, 10, axis=0)  # small background
            explainer = shap.Explainer(model, background)
            shap_values = explainer(features)
            mean_abs = np.abs(shap_values.values[0]).mean(axis=0)
            top_features = np.argsort(mean_abs)[-3:][::-1].tolist()
        except Exception as e:
            print("[WARN] SHAP explainability skipped:", e)
            top_features = []

        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "explanation": top_features
        }), 200

    except Exception as e:
        print("[ERROR] Detection failed:", str(e))
        return jsonify({"error": str(e)}), 500



@app.route("/retrain", methods=["POST"])
def retrain():
    """Retrain hybrid model with new labeled data."""
    global model, scaler

    try:
        data = request.get_json()
        if not data or "X" not in data or "y" not in data:
            return jsonify({"error": "Missing 'X' or 'y' in request body."}), 400

        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)

        # Preprocess new data
        X = preprocess_input_data(X)

        if scaler is not None:
            flat = X.reshape(X.shape[0], -1)
            flat = scaler.transform(flat)
            X = flat.reshape(X.shape)

        model.fit(X, y, epochs=5, batch_size=16, verbose=1)
        model.save(MODEL_PATH)

        return jsonify({"message": "Model retrained and saved successfully."}), 200

    except Exception as e:
        print("[ERROR] Retraining failed:", str(e))
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------
# Run Server
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Starting Medical Intrusion Detection API Server...")
    load_resources()
    app.run(host="0.0.0.0", port=8000, debug=True)
