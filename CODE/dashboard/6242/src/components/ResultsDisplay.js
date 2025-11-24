import React from 'react';
import './ResultsDisplay.css';
import PredictionResult from './PredictionResult';
import TeamMetrics from './TeamMetrics';
import ShapExplanation from './ShapExplanation';
import PathToVictory from './PathToVictory';

function ResultsDisplay({ results }) {
  if (!results) {
    return null;
  }

  return (
    <div className="Results-Container">
      <h2>Analysis Results</h2>

      {/* Prediction */}
      <PredictionResult
        winner={results.prediction.winner}
        probability={results.prediction.probability}
        WinnerLogoComponent={results.prediction.WinnerLogoComponent}
        predictedSpread={results.prediction.predicted_spread}
        consistent={results.prediction.consistent}
        matchup={results.matchup}
      />

      {/* Tale of the Tape */}
      <TeamMetrics
        metrics={results.metrics}
        homeTeam={results.matchup?.home_team}
        awayTeam={results.matchup?.away_team}
      />

      {/* Path to Victory */}
      <PathToVictory 
        features={results.explanations} 
        winner={results.prediction.winner}
        homeTeam={results.matchup?.home_team}
      /> 

      {/* Keys to the Game */}
      <ShapExplanation features={results.explanations || []} />
      
    </div>
  );
}

export default ResultsDisplay;