import React from 'react';
import './components.css';

function PredictionResult({ winner, probability, WinnerLogoComponent, predictedSpread, matchup }) {
  if (!winner) return null;

  // Format spread display
  let spreadText = '';
  if (predictedSpread !== null && predictedSpread !== undefined && matchup) {
    const absSpread = Math.abs(predictedSpread).toFixed(1);
    const favored = predictedSpread > 0 ? matchup.home_team : matchup.away_team;
    spreadText = `${favored} by ${absSpread}`;
  }

  return (
    <div className="Prediction-container">
      <h2>Prediction</h2>
      <div className="Winner-info">
        {WinnerLogoComponent && <WinnerLogoComponent size={60} />}
        <p className="Winner-text">{winner}</p>
      </div>
      <p className="Probability-text">Win Probability: {probability}%</p>
      {spreadText && <p className="Spread-text">Predicted Spread: {spreadText}</p>}
    </div>
  );
}

export default PredictionResult;
