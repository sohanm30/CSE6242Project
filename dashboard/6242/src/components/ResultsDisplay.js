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
        matchup={results.matchup}
      />
  
      {/* Keys to the Game 
      <ShapExplanation features={results.explanation} />
      */}
      
      {/* Tale of the Tape */}
      <TeamMetrics 
        metrics={results.metrics} 
        homeTeam={results.matchup?.home_team} 
        awayTeam={results.matchup?.away_team} 
      />
      
      {/* Path to Victory
      <PathToVictory 
        teamName={results.pathToVictory.teamName} 
        steps={results.pathToVictory.steps} 
      /> 
      */}
    </div>
  );
}

export default ResultsDisplay;