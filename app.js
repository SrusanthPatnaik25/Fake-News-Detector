import { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [source, setSource] = useState("bbc");
  const [headlines, setHeadlines] = useState([]);
  const [inputText, setInputText] = useState("");
  const [prediction, setPrediction] = useState(null);

  const backendURL = "https://<your-username>-<your-spacename>.hf.space";

  const fetchHeadlines = async () => {
    const res = await fetch(`${backendURL}/api/predict/fetch_headlines`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    });
    const data = await res.json();
    setHeadlines(data);
  };

  const analyzeText = async (text) => {
    const res = await fetch(`${backendURL}/api/predict/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    setPrediction(data);
  };

  useEffect(() => {
    fetchHeadlines();
  }, [source]);

  return (
    <div className={darkMode ? "App dark" : "App"}>
      <header>
        <h1>📰 Fake News Detector Dashboard</h1>
        <button onClick={() => setDarkMode(!darkMode)}>
          🌙 {darkMode ? "Light Mode" : "Dark Mode"}
        </button>
      </header>

      <section>
        <input
          type="text"
          placeholder="Enter news headline or article..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
        />
        <button onClick={() => analyzeText(inputText)}>Analyze</button>

        {prediction && (
          <div className="prediction">
            <strong>{prediction.prediction}</strong>
            <p>Confidence: {prediction.confidence}</p>
          </div>
        )}
      </section>

      <section>
        <label>Select News Source: </label>
        <select value={source} onChange={(e) => setSource(e.target.value)}>
          <option value="bbc">BBC</option>
          <option value="toi">Times of India</option>
        </select>
        <button onClick={fetchHeadlines}>Refresh Headlines</button>

        <ul>
          {headlines.map((h, i) => (
            <li key={i}>
              {h} <button onClick={() => analyzeText(h)}>Analyze</button>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

export default App;
