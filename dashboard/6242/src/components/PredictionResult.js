import React from 'react';
import './components.css';

function PredictionResult({ winner, probability, WinnerLogoComponent }) {
  if (!winner) return null;

  return (
    <div className="Prediction-container">
      <h2>Prediction</h2>
      <div className="Winner-info">
        {WinnerLogoComponent && <WinnerLogoComponent size={60} />}
        <p className="Winner-text">{winner}</p>
      </div>
      <p className="Probability-text">Win Probability: {probability}%</p>
    </div>
  );
}

export default PredictionResult;
