import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleDetect = async () => {
  try {
    setLoading(true);

    // Sanitize and split values
    const cleaned = input
      .trim()
      .replace(/\n/g, ",")
      .replace(/,+/g, ",")
      .replace(/^,|,$/g, "");

    const values = cleaned.split(",").map(Number).filter(v => !isNaN(v));

    // Check if total count fits 4-feature structure
    if (values.length % 4 !== 0) {
      alert("Number of values must be a multiple of 4 (HeartRate, Temp, SpO2, ECG per timestep).");
      setLoading(false);
      return;
    }

    // Convert to 2D array [[4 features], [4 features], ...]
    const seq = [];
    for (let i = 0; i < values.length; i += 4) {
      seq.push(values.slice(i, i + 4));
    }

    // Pad or truncate to exactly 128 timesteps
    while (seq.length < 128) seq.push([0, 0, 0, 0]);
    if (seq.length > 128) seq.splice(128);

    const res = await axios.post("http://127.0.0.1:8000/detect", { features: seq });

    console.log("Response:", res.data);
    setResult(res.data);
  } catch (err) {
    console.error("Error contacting backend:", err);
    alert("There was a problem connecting to the backend. Check console.");
  } finally {
    setLoading(false);
  }
};



  return (
    <div className="App">
      <h1>Medical Intrusion Detection Dashboard</h1>
      <textarea
        placeholder="Enter comma-separated feature values..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />
      <button onClick={handleDetect} disabled={loading}>
        {loading ? "Analyzing..." : "Detect Intrusion"}
      </button>

      {result && (
        <div className="output">
          <h2>Prediction: {result.prediction.toUpperCase()}</h2>
          <p>Confidence: {result.confidence.toFixed(3)}</p>
          <p>Top Influential Features: {result.explanation.join(", ")}</p>
        </div>
      )}
    </div>
  );
}

export default App;
