import { useState } from 'react';
import './App.css';
import { TEAM_MAP, TEAM_NAMES } from './data/teams';
import ResultsDisplay from './components/ResultsDisplay';
import MatchupDisplay from './components/MatchupDisplay';

function App() {
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleHomeChange = (event) => {
    setHomeTeam(event.target.value);
    setResults(null); 
  };

  const handleAwayChange = (event) => {
    setAwayTeam(event.target.value);
    setResults(null); 
  };

  const handleAnalyze = async () => {
    setResults(null);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
      });

      if (!response.ok) {
        const err = await response.json();
        alert(`Error: ${err.error}`);
        return;
      }

      const data = await response.json();

      const winner = data.prediction.winner;
      const winnerProb = Math.round(data.prediction.confidence * 100);

      const homeStats = data.season_stats?.home;
      const awayStats = data.season_stats?.away;

      const metrics = homeStats && awayStats ? [
        { name: 'Record', homeValue: `${homeStats.wins}-${homeStats.losses}`, awayValue: `${awayStats.wins}-${awayStats.losses}` },
        { name: 'PPG', homeValue: homeStats.ppg.toFixed(1), awayValue: awayStats.ppg.toFixed(1) },
        { name: 'FG%', homeValue: `${(homeStats.fg_pct * 100).toFixed(1)}%`, awayValue: `${(awayStats.fg_pct * 100).toFixed(1)}%` },
        { name: '3P%', homeValue: `${(homeStats.three_pct * 100).toFixed(1)}%`, awayValue: `${(awayStats.three_pct * 100).toFixed(1)}%` },
        { name: 'Rebounds', homeValue: homeStats.rebounds.toFixed(1), awayValue: awayStats.rebounds.toFixed(1) },
        { name: 'Assists', homeValue: homeStats.assists.toFixed(1), awayValue: awayStats.assists.toFixed(1) },
      ] : []; 

      const formattedData = {
        prediction: {
          winner: winner,
          probability: winnerProb,
          WinnerLogoComponent: TEAM_MAP[winner]?.component || null,
          predicted_spread: data.prediction.predicted_spread,
          consistent: data.prediction.consistent,
        },
        matchup: data.matchup,
        probabilities: data.probabilities,
        metrics: metrics,
        explanations: data.explanations,
      };

      setResults(formattedData);
    } catch (error) {
      console.error('Prediction failed:', error);
      alert('Failed to get prediction. Make sure Flask server is running on port 5001.');
    } finally {
      setIsLoading(false);
    }
  };

  const awayTeamOptions = TEAM_NAMES.filter(team => team !== homeTeam);
  const homeTeamOptions = TEAM_NAMES.filter(team => team !== awayTeam);

  const HomeLogo = homeTeam ? TEAM_MAP[homeTeam].component : null;
  const AwayLogo = awayTeam ? TEAM_MAP[awayTeam].component : null;

  return (
    <div className="App">
      <header className="App-header">
        <h1>NBA GamePlan: Matchup Predictor</h1>
      </header>

      <div className="Dashboard">
        <div className="Team-selectors">
          <div className="Team-selector-container">
            <div className="Team-selector">
              <label htmlFor="home-team">Home Team</label>
              <select id="home-team" className="Team-dropdown" value={homeTeam} onChange={handleHomeChange}>
                <option value="">Select a Team</option>
                {homeTeamOptions.map(team => (
                  <option key={`home-${team}`} value={team}>{team}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="Team-selector-container">
            <div className="Team-selector">
              <label htmlFor="away-team">Away Team</label>
              <select id="away-team" className="Team-dropdown" value={awayTeam} onChange={handleAwayChange}>
                <option value="">Select a Team</option>
                {awayTeamOptions.map(team => (
                  <option key={`away-${team}`} value={team}>{team}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
        
        <button 
          className="Analyze-button" 
          onClick={handleAnalyze} 
          disabled={!homeTeam || !awayTeam || isLoading}
        >
          {isLoading ? 'Analyzing...' : 'Analyze Matchup'}
        </button>
      </div>

      <MatchupDisplay
        homeTeam={homeTeam}
        awayTeam={awayTeam}
        HomeLogo={HomeLogo}
        AwayLogo={AwayLogo}
      />

      {isLoading && (
        <div className="Loading-container">
          <div className="Loading-spinner"></div>
          <p>Crunching the numbers...</p>
        </div>
      )}

      <ResultsDisplay results={results} />
    </div>
  );
}

export default App;