import React from 'react';
import './components.css';

function PredictionResult({ winner, probability, WinnerLogoComponent, predictedSpread, consistent, matchup }) {
  if (!winner) return null;

  const isHomeWinner = winner === matchup.home_team;
  const homeProb = isHomeWinner ? probability : 100 - probability;
  const awayProb = 100 - homeProb;
  const winnerColor = isHomeWinner ? '#007bff' : '#dc3545';

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
        <p className="Winner-text" style={{ color: winnerColor }}>
          {winner}
        </p>
      </div>

      {spreadText && (
        <div className="Spread-container">
          {consistent === false ? (
            <p className="Spread-text warning" title="The win probability model and spread model disagree. This indicates a very close game.">
              <strong>Warning: Model Divergence:</strong> Spread favors {spreadText}
            </p>
          ) : (
            <p className="Spread-text">Predicted Spread: {spreadText}</p>
          )}
        </div>
      )}

      <div className="TugOfWar-bar">
        <div 
          className="TugOfWar-segment home" 
          style={{ width: `${homeProb}%` }}
        >
          {homeProb > 15 && <span>{homeProb.toFixed(0)}%</span>}
        </div>
        <div 
          className="TugOfWar-segment away" 
          style={{ width: `${awayProb}%` }}
        >
           {awayProb > 15 && <span>{awayProb.toFixed(0)}%</span>}
        </div>
      </div>

    </div>
  );
}

export default PredictionResult;