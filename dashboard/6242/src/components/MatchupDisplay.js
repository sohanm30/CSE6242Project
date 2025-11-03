import React from 'react';
import './MatchupDisplay.css';

function MatchupDisplay({ homeTeam, awayTeam, HomeLogo, AwayLogo }) {
  if (!homeTeam && !awayTeam) {
    return null;
  }

  return (
    <div className="Matchup-Display">
      <div className="Matchup-Team-Container">
        {HomeLogo && <HomeLogo size={200} />}
        {homeTeam && <h2 className="Team-Name">{homeTeam}</h2>}
      </div>
      
      <div className="Matchup-VS">
        {HomeLogo && AwayLogo && <span>VS</span>}
      </div>
      
      <div className="Matchup-Team-Container">
        {AwayLogo && <AwayLogo size={200} />}
        {awayTeam && <h2 className="Team-Name">{awayTeam}</h2>}
      </div>
    </div>
  );
}

export default MatchupDisplay;