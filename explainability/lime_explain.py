# explainability/lime_explain.py
import os
import numpy as np
from lime import lime_tabular
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "explainability", "outputs")
os.makedirs(OUT, exist_ok=True)

MODEL_PATH = os.path.join(ROOT, "models", "checkpoints", "hybrid_final.h5")
SCALER = os.path.join(ROOT, "preprocessing", "scaler.joblib")

def lime_for_flattened():
    # LIME needs a 2D tabular input. We'll load flattened processed features
    X = np.load(os.path.join(DATA_DIR, "processed_flattened.npy"))
    y = np.load(os.path.join(DATA_DIR, "labels.npy"))
    # load model (we'll use a small wrapper: average time dimension to get features)
    # For demonstration, we'll build a simple function that maps flattened -> model prediction
    model = load_model(MODEL_PATH)
    def predict_fn(flat_batch):
        # flat_batch: (B, seq_len*channels)
        B = flat_batch.shape[0]
        seq_len = int(flat_batch.shape[1] // 4)
        batch = flat_batch.reshape(B, seq_len, 4)
        preds = model.predict(batch, verbose=0).squeeze()
        return np.vstack([1-preds, preds]).T  # lime expects probabilities for each class
    explainer = lime_tabular.LimeTabularExplainer(X, mode='classification')
    i = np.random.randint(0, X.shape[0])
    exp = explainer.explain_instance(X[i], predict_fn, num_features=10)
    fig = exp.as_pyplot_figure()
    fpath = os.path.join(OUT, f"lime_sample_{i}.png")
    fig.savefig(fpath)
    print("Saved LIME plot to", fpath)

if __name__ == '__main__':
    lime_for_flattened()
