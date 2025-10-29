import { useState } from 'react';
import './App.css';
import ResultsModal from './components/ResultsModal';
import { TEAM_MAP, TEAM_NAMES } from './data/teams';
import { FAKE_RESULTS } from './data/fakeData';

function App() {
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [results, setResults] = useState(null);

  const handleHomeChange = (event) => {
    setHomeTeam(event.target.value);
  };

  const handleAwayChange = (event) => {
    setAwayTeam(event.target.value);
  };

  const handleAnalyze = () => {
    const fakeData = {
      ...FAKE_RESULTS,
      prediction: {
        ...FAKE_RESULTS.prediction,
        winner: homeTeam,
        WinnerLogoComponent: TEAM_MAP[homeTeam].component,
      },
      // In a real app, you'd dynamically set the loser for Path to Victory
      pathToVictory: {
        ...FAKE_RESULTS.pathToVictory,
        teamName: awayTeam,
      }
    }
    setResults(fakeData);
  };

  const handleCloseModal = () => {
    setResults(null);
  }

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
            {HomeLogo && <HomeLogo size={50} />}
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
            {AwayLogo && <AwayLogo size={50} />}
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
        <button className="Analyze-button" onClick={handleAnalyze} disabled={!homeTeam || !awayTeam}>
          Analyze Matchup
        </button>
      </div>

      {results && <ResultsModal results={results} onClose={handleCloseModal} />}
    </div>
  );
}

export default App;