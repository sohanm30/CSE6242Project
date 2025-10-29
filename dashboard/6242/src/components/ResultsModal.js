import React from 'react';
import './Modal.css';
import PredictionResult from './PredictionResult';
import ShapExplanation from './ShapExplanation';
import KeyPlayers from './KeyPlayers';
import TeamMetrics from './TeamMetrics';
import PathToVictory from './PathToVictory';

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
          <PredictionResult winner={results.prediction.winner} probability={results.prediction.probability} WinnerLogoComponent={results.prediction.WinnerLogoComponent} />
          <ShapExplanation features={results.explanation} />
          <TeamMetrics metrics={results.metrics} />
          <KeyPlayers homePlayers={results.keyPlayers.home} awayPlayers={results.keyPlayers.away} />
          <PathToVictory teamName={results.pathToVictory.teamName} steps={results.pathToVictory.steps} />
        </div>
      </div>
    </div>
  );
}

export default ResultsModal;
