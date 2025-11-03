import React from 'react';
import './components.css';

function PredictionResult({ winner, probability, WinnerLogoComponent, predictedSpread, matchup }) {
  if (!winner) return null;

  const isHomeWinner = winner === matchup.home_team;
  const homeProb = isHomeWinner ? probability : 100 - probability;
  const awayProb = 100 - homeProb;

  let spreadText = '';
  if (predictedSpread !== null && predictedSpread !== undefined && matchup) {
    const absSpread = Math.abs(predictedSpread).toFixed(1);
    const favored = predictedSpread > 0 ? matchup.home_team : matchup.away_team;
    spreadText = `${favored} by ${absSpread}`;
  }

  return (
    <div className="Prediction-container">
      <h2>Prediction</h2>
      
      {/* Winner and Spread Text */}
      <div className="Winner-info">
        {WinnerLogoComponent && <WinnerLogoComponent size={60} />}
        <p className="Winner-text">{winner}</p>
      </div>
      {spreadText && <p className="Spread-text">{spreadText}</p>}

      {/* Probability Bar */}
      <div className="TugOfWar-bar">
        <div 
          className="TugOfWar-segment home" 
          style={{ width: `${homeProb}%` }}
        >
          <span>{homeProb.toFixed(0)}%</span>
        </div>
        <div 
          className="TugOfWar-segment away" 
          style={{ width: `${awayProb}%` }}
        >
          <span>{awayProb.toFixed(0)}%</span>
        </div>
      </div>

    </div>
  );
}

export default PredictionResult;