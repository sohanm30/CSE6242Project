import React from 'react';
import './components.css';

function KeyPlayers({ homePlayers, awayPlayers }) {
  return (
    <div className="key-players-container">
      <h4>Key Players</h4>
      <div className="players-columns">
        <div className="team-players">
          <h5>Home</h5>
          <ul>
            {homePlayers.map(player => <li key={player}>{player}</li>)}
          </ul>
        </div>
        <div className="team-players">
          <h5>Away</h5>
          <ul>
            {awayPlayers.map(player => <li key={player}>{player}</li>)}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default KeyPlayers;
