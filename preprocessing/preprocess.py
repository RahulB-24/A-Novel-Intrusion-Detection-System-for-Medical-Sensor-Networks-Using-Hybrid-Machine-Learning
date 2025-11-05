import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from joblib import load

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")

def feature_engineer(X):
    feats = []
    for s in X:
        f = []
        for ch in range(s.shape[1]):
            a = s[:, ch]
            f += [a.mean(), a.std(), a.min(), a.max(),
                  (np.abs(np.fft.rfft(a)) ** 2).mean()]
        feats.append(f)
    return np.array(feats)

def split_scale(feats, y):
    from sklearn.model_selection import train_test_split
    Xtr, Xte, Ytr, Yte = train_test_split(feats, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    np.save(os.path.join(DATA, "train.npy"), Xtr)
    np.save(os.path.join(DATA, "test.npy"), Xte)
    np.save(os.path.join(DATA, "train_labels.npy"), Ytr)
    np.save(os.path.join(DATA, "test_labels.npy"), Yte)
    from joblib import dump
    dump(sc, os.path.join(ROOT, "preprocessing", "scaler.joblib"))
    print("Saved scaled train/test sets.")

# ✅ ADD THIS FUNCTION BELOW (runtime preprocessing)
def preprocess_input_data(raw_data):
    """
    Preprocesses raw input (e.g., sensor readings) before inference.
    Loads the saved scaler and applies feature engineering.
    """
    scaler_path = os.path.join(ROOT, "preprocessing", "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found. Run preprocess.py once to generate it.")

    scaler = load(scaler_path)

    # Ensure input shape consistency
    if isinstance(raw_data, list):
        raw_data = np.array(raw_data)

    if raw_data.ndim == 2:
        raw_data = np.expand_dims(raw_data, axis=0)

    feats = feature_engineer(raw_data)
    scaled_feats = scaler.transform(feats)
    return scaled_feats

if __name__ == "__main__":
    X = np.load(os.path.join(DATA, "processed_timeseries.npy"))
    y = np.load(os.path.join(DATA, "labels.npy"))
    feats = feature_engineer(X)
    split_scale(feats, y)
