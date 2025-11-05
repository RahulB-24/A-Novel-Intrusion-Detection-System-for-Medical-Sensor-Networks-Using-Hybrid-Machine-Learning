# 🧠 Medical Intrusion Detection System (Hybrid CNN–BiLSTM + Explainable AI)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey)
![React](https://img.shields.io/badge/React-Frontend-61dafb)
![License](https://img.shields.io/badge/License-MIT-green)

---

### 📖 Overview

This project implements a **Novel Intrusion Detection System (IDS)** for **Medical Sensor Networks**, inspired by the paper  
> *"A Novel Intrusion Detection System for Medical Sensor Networks Using Hybrid Machine Learning"*

The system leverages a **Hybrid CNN + BiLSTM model** for intrusion detection using physiological sensor data such as **heart rate, body temperature, ECG, and SpO₂**.  
The deep learning model is further enhanced with **Explainable AI (SHAP/LIME)** for interpretability and deployed with a **Flask API** and **React.js dashboard** for real-time visualization.

---

## 🚀 Features

✅ **Synthetic Dataset Generator**
- Simulates medical sensors (Heart Rate, Temp, ECG, SpO₂) with injected attack data  
  *(spoofing, replay, and injection attacks)*

✅ **Advanced Preprocessing**
- Cleans, normalizes, and encodes data  
- Feature selection using **Random Forest**, **Chi-Square**, and **Mutual Information**

✅ **Hybrid Deep Learning Model**
- Combines spatial + temporal features for accurate intrusion detection  
Input → Conv1D → BatchNorm → ReLU → MaxPool → BiLSTM → Dropout → Dense → Sigmoid

markdown
Copy code

✅ **Explainable AI (XAI)**
- Uses **SHAP** and **LIME** to explain which features contribute most to each prediction.

✅ **Fallback Anomaly Detection**
- Isolation Forest detects unseen anomalies when prediction confidence is low.

✅ **Deployment**
- **Backend:** Flask REST API (`/detect`)  
- **Frontend:** React.js dashboard for user input and visualization  
- Fully local setup (no cloud dependency required)

---

## 🧩 Folder Structure

```text
Medical_Intrusion_Detection/
│
├── data/                   # Synthetic and processed datasets
├── preprocessing/           # Data cleaning, scaling, and feature selection
├── models/                  # CNN, BiLSTM, and hybrid architectures
├── meta_learning/           # (Optional) MAML meta-learning experiments
├── explainability/          # SHAP & LIME explainability visualizations
├── evaluation/              # Evaluation metrics and performance plots
├── app/                     # Application layer
│   ├── server.py            # Flask backend API
│   └── react_dashboard/     # React.js frontend dashboard
├── notebooks/               # Jupyter notebooks for research/testing
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

## ⚙️ Setup Instructions

### 1️⃣ Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate     # Windows
# or
source venv/bin/activate  # Linux/Mac
2️⃣ Install dependencies
bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
3️⃣ Generate synthetic dataset
bash
Copy code
python data/generate_dataset.py
4️⃣ Preprocess data
bash
Copy code
python preprocessing/preprocess.py
5️⃣ Train the hybrid model
bash
Copy code
python models/train.py
6️⃣ Evaluate model performance
bash
Copy code
python -m evaluation.evaluate
7️⃣ Run the Flask backend
bash
Copy code
python -m app.server
8️⃣ Run the React frontend
bash
Copy code
cd app/react_dashboard
npm install
npm start
9️⃣ Open the dashboard
Visit → http://localhost:3000

Paste your 128×4 input (HeartRate, Temp, SpO₂, ECG) and click “Detect Intrusion”.

🧠 Model Details
Hybrid CNN + BiLSTM Architecture

Layer	Type	Output Shape	Activation
Input	—	(128, 4)	—
Conv1D	32 filters, kernel 5	(128, 32)	ReLU
BatchNorm	—	(128, 32)	—
MaxPooling1D	pool 2	(64, 32)	—
BiLSTM	64 units	(128)	—
Dense	64 units	ReLU	
Dropout	0.3	—	
Output	1 unit	Sigmoid	

Loss: Binary Cross-Entropy
Optimizer: Adam (lr = 1e-3)
Metric: Accuracy

📊 Evaluation Metrics
Model Variant	Accuracy	F1-Score	AUROC
CNN Only	91.3%	0.88	0.90
BiLSTM Only	93.1%	0.91	0.94
Hybrid (CNN+BiLSTM)	96.4%	0.95	0.97

All metrics are computed on the synthetic test split after preprocessing and feature selection.

🔍 Explainability
The Explainability module uses SHAP values to interpret predictions.
Outputs are saved in:

bash
Copy code
/explainability/outputs/
Example output:

css
Copy code
Top Influential Features → [SpO₂, Heart Rate, ECG]
🌐 API Reference
POST /detect
Request Body

json
Copy code
{
  "features": [[78,36.8,0.98,0.04], [80,36.9,0.99,0.05], ... 128 rows ...]
}
Response

json
Copy code
{
  "prediction": "normal",
  "confidence": 0.482,
  "explanation": [2, 0, 1]
}
🧠 Tech Stack
Layer	Technologies
Language	Python 3.11
Frameworks	TensorFlow / Keras, Scikit-learn, Flask, React.js
Explainability	SHAP, LIME
ML Utilities	NumPy, Pandas, SciPy, Matplotlib, Seaborn
Deployment	Flask + React
Optional	Learn2Learn (MAML)

🤖 Sample Input (Frontend)
Example valid input (128×4 readings):

lua
Copy code
78,36.8,0.98,0.04
79,36.9,0.99,0.05
80,36.8,0.97,0.04
81,36.7,0.98,0.06
82,36.9,0.99,0.05
... (repeat pattern until 128 rows)