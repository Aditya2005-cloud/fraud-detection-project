import React, { useState } from "react";
import "./index.css";

const INITIAL_FEATURES = {
  v1: "",
  v2: "",
  v3: "",
  v4: "",
  v5: "",
  v6: "",
  v7: "",
  v8: "",
};

export default function App() {
  const [time, setTime] = useState("");
  const [amount, setAmount] = useState("");
  const [features, setFeatures] = useState(INITIAL_FEATURES);

  function loadLegitSample() {
    setTime("32941.04426927148");
    setAmount("59.67");
    setFeatures({
      v1: "-1.460678",
      v2: "0.030159",
      v3: "2.675455",
      v4: "1.287264",
      v5: "-0.40634",
      v6: "0.374663",
      v7: "0.255996",
      v8: "-0.047776",
    });
  }

  function loadFraudSample() {
    setTime("406.0");
    setAmount("1879.55");
    setFeatures({
      v1: "1.836",
      v2: "-0.553",
      v3: "3.214",
      v4: "1.902",
      v5: "-2.107",
      v6: "0.842",
      v7: "-0.321",
      v8: "1.104",
    });
  }

  function handleFeatureChange(key, value) {
    setFeatures((prev) => ({ ...prev, [key]: value }));
  }

  function clearForm() {
    setTime("");
    setAmount("");
    setFeatures(INITIAL_FEATURES);
  }

  function handleSubmit(e) {
    e.preventDefault();
    // Demo only – no backend. Could show toast here.
    alert("Risk check submitted (demo only).");
  }

  return (
    <div className="page-shell">
      <div className="brand-floating">
        <div className="brand-row">
          <div className="brand-logo-wrap">
            <div className="brand-logo-pulse">
              <img
                src="/fraud-logo.png"
                alt="Fraud Detection logo"
                className="brand-logo"
              />
            </div>
          </div>
          <div className="brand-text">
            <h1 className="brand-title text-glow">FRAUD DETECTION</h1>
            <p className="brand-subtitle">Risk Scoring &amp; Analytics</p>
          </div>
        </div>
      </div>

      <div className="page-content">
        <header className="panel panel-header">
          <div className="header-top">
            <h2 className="title text-glow">Transaction risk scoring</h2>
            <p className="subtitle">
              Transaction risk score (0–100) from ensemble model. Load a sample or paste
              your own features to simulate a prediction.
            </p>
          </div>
        </header>

        <main className="panel">
          <form onSubmit={handleSubmit} className="fd-form">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Transaction input</p>
                <h2 className="section-title">Card & PCA features</h2>
              </div>
            </div>

            <div className="fd-row fd-row-buttons">
              <button
                type="button"
                className="btn-secondary"
                onClick={loadLegitSample}
              >
                Load legitimate sample
              </button>
              <button
                type="button"
                className="btn-secondary"
                onClick={loadFraudSample}
              >
                Load fraud sample
              </button>
            </div>

            <div className="fd-row fd-row-two">
              <div className="form-field">
                <label className="field-label">Time (sec)</label>
                <input
                  className="input-field"
                  value={time}
                  onChange={(e) => setTime(e.target.value)}
                  placeholder="e.g. 32941.04426927148"
                />
              </div>
              <div className="form-field">
                <label className="field-label">Amount</label>
                <input
                  className="input-field"
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  placeholder="e.g. 59.67"
                />
              </div>
            </div>

            <button type="button" className="fd-toggle-link">
              Hide V1–V8 (PCA features)
            </button>

            <div className="fd-grid">
              {Object.keys(features).map((key) => (
                <div key={key} className="form-field">
                  <label className="field-label">{key.toUpperCase()}</label>
                  <input
                    className="input-field"
                    value={features[key]}
                    onChange={(e) => handleFeatureChange(key, e.target.value)}
                  />
                </div>
              ))}
            </div>

            <div className="fd-row fd-row-footer">
              <button
                type="button"
                className="btn-secondary"
                onClick={clearForm}
              >
                Clear
              </button>
              <button type="submit" className="btn-primary">
                Check risk
              </button>
            </div>
          </form>
        </main>
      </div>
    </div>
  );
}
