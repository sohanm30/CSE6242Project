import React from 'react';
import './Modal.css';
import PredictionResult from './PredictionResult';
import TeamMetrics from './TeamMetrics';

function ResultsModal({ results, onClose }) {
  if (!results) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Analysis Results</h2>
          <button onClick={onClose} className="close-button">&times;</button>
        </div>
        <div className="modal-body">
          <PredictionResult 
            winner={results.prediction.winner} 
            probability={results.prediction.probability} 
            WinnerLogoComponent={results.prediction.WinnerLogoComponent}
            predictedSpread={results.prediction.predicted_spread}
            matchup={results.matchup}
          />
          <TeamMetrics metrics={results.metrics} homeTeam={results.matchup?.home_team} awayTeam={results.matchup?.away_team} />
        </div>
      </div>
    </div>
  );
}

export default ResultsModal;
