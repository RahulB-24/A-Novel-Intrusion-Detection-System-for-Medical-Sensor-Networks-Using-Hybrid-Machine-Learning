"""
Explainability Script (SHAP KernelExplainer)
-------------------------------------------
Generates SHAP explanations for the trained Hybrid CNN-BiLSTM model
and saves feature-importance plots under /explainability/outputs/.

Compatible with TensorFlow ≥2.13 and Python 3.11.
"""

import os
import numpy as np
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
EXP_DIR = os.path.join(ROOT, "explainability")
OUT_DIR = os.path.join(EXP_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ROOT, "models", "checkpoints", "hybrid_final.h5")

# -------------------------------------------------------------
# Main SHAP computation
# -------------------------------------------------------------
def compute_and_save_shap():
    print("🔹 Loading model and data ...")
    model = load_model(MODEL_PATH)
    X = np.load(os.path.join(DATA_DIR, "processed_timeseries.npy"))
    seq_len, channels = X.shape[1], X.shape[2]

    background = X[np.random.choice(X.shape[0], size=25, replace=False)]
    test_samples = X[np.random.choice(X.shape[0], size=5, replace=False)]

    background_flat = background.reshape(background.shape[0], -1)
    test_flat = test_samples.reshape(test_samples.shape[0], -1)

    # ---------------------------------------------------------
    # Prediction wrapper
    # ---------------------------------------------------------
    def predict_fn(x):
        """x -> (n_samples, seq_len*channels)"""
        x = x.reshape(-1, seq_len, channels)
        preds = model.predict(x, verbose=0).squeeze()

        if preds.ndim == 0:
            preds = np.array([preds])
        if preds.ndim == 1:
            preds = np.vstack([1 - preds, preds]).T
        return preds

    # ---------------------------------------------------------
    print("🔹 Running SHAP KernelExplainer ... (may take a few minutes)")
    explainer = shap.KernelExplainer(predict_fn, background_flat[:10])
    shap_values = explainer.shap_values(test_flat, nsamples=100)

    shap_values = np.array(shap_values)
    # Handle (classes, samples, features)
    if shap_values.ndim == 3:
        shap_values = shap_values[1]  # class 1 (attack)
    shap_abs = np.abs(shap_values).mean(axis=0)

    # ---------------------------------------------------------
    # Safe reshaping: only reshape if size matches seq_len*channels
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 4))

    if shap_abs.size == seq_len * channels:
        shap_abs = shap_abs.reshape(seq_len, channels).mean(axis=0)
        plt.bar(range(channels), shap_abs, color="teal")
        plt.xticks(range(channels), ["HeartRate", "Temp", "SpO2", "ECG"])
        plt.xlabel("Sensor Channel")
        plt.ylabel("Mean |SHAP|")
        plt.title("Average SHAP Contribution per Channel")
    else:
        # fallback summary for top features
        top_k = min(10, shap_abs.size)
        top_idx = np.argsort(shap_abs)[-top_k:][::-1]
        plt.bar(range(top_k), shap_abs[top_idx], color="teal")
        plt.xticks(range(top_k), [f"F{i}" for i in top_idx])
        plt.xlabel("Top Features")
        plt.ylabel("Mean |SHAP|")
        plt.title("Average SHAP Contribution (Top Features)")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "shap_avg_channels.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✅ SHAP analysis complete.\nSaved plot: {out_path}")

# -------------------------------------------------------------
if __name__ == "__main__":
    compute_and_save_shap()
