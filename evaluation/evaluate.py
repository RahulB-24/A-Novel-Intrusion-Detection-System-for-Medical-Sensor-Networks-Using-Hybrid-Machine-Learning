# evaluation/evaluate.py
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from time import perf_counter
from tensorflow.keras.models import load_model
from models.hybrid_model import build_hybrid


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "models", "checkpoints")
OUT = os.path.join(ROOT, "evaluation", "outputs")
os.makedirs(OUT, exist_ok=True)

MODEL_PATH = os.path.join(CKPT_DIR, "hybrid_final.h5")

def load_data_flat():
    X_ts = np.load(os.path.join(DATA_DIR, "processed_timeseries.npy"))
    y = np.load(os.path.join(DATA_DIR, "labels.npy"))
    return X_ts, y

def evaluate_model(model_path=MODEL_PATH):
    print("Loading model:", model_path)
    model = load_model(model_path)
    X, y = load_data_flat()
    # evaluate predictions and latency
    start = perf_counter()
    preds = model.predict(X, batch_size=64, verbose=0).squeeze()
    end = perf_counter()
    latency_ms = (end - start) / X.shape[0] * 1000.0
    y_pred = (preds >= 0.5).astype(int)
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auroc': roc_auc_score(y, preds),
        'avg_latency_ms': latency_ms
    }
    print(metrics)
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(OUT, "confusion_matrix.png"))
    plt.close()
    fpr, tpr, _ = roc_curve(y, preds)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.savefig(os.path.join(OUT, "roc_curve.png"))
    plt.close()
    print("Saved plots to", OUT)
    return metrics

if __name__ == '__main__':
    evaluate_model()
