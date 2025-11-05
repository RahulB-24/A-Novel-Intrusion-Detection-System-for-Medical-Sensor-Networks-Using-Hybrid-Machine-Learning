import numpy as np
import requests
import json

# Generate synthetic (128, 4) test input
data = np.random.rand(128, 4) * [100, 40, 1, 0.1]  # roughly like your ranges

# Convert to list for JSON
payload = data.tolist()

response = requests.post("http://127.0.0.1:8000/detect", 
                         json={"data": payload})
print(response.json())
