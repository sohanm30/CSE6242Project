import React from 'react';
import './components.css';

function PathToVictory({ features, winner, homeTeam }) {
  if (!features || features.length === 0) return null;

  const formatFeatureName = (name) => {
    return name
      .replace(/([A-Z])/g, ' $1')
      .replace(/([a-zA-Z])(\d)/g, '$1 $2')
      .replace(/(\d)([a-zA-Z])/g, '$1 $2')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase())
      .replace(/Ewm/g, '')
      .replace(/Roll10/g, '')
      .replace(/Avg/g, '')
      .replace(/Pct/g, '%')
      .replace(/Fg/g, 'FG')
      .replace(/Ppg/g, 'PPG')
      .replace(/Opp/g, 'Opponent')
      .replace(/\s+/g, ' ')
      .trim();
  };

  const generateStrategy = (item) => {
    const isHomeWinner = winner === homeTeam;
    const factorFavorsHome = item.impact > 0;
    const factorFavorsAway = item.impact < 0;
    const helpsWinner = (isHomeWinner && factorFavorsHome) || (!isHomeWinner && factorFavorsAway);
    const metricName = formatFeatureName(item.feature);

    if (helpsWinner) {
      const phrases = [
        `Capitalize on the advantage in ${metricName}`,
        `Maintain dominance in ${metricName}`,
        `Leverage superior ${metricName}`,
        `Continue to exploit the edge in ${metricName}`
      ];
      return phrases[metricName.length % phrases.length];
    } else {
      const phrases = [
        `Overcome the opponent's edge in ${metricName}`,
        `Tighten up defense against ${metricName}`,
        `Mitigate the impact of the opponent's ${metricName}`,
        `Survive the opponent's strong ${metricName}`
      ];
      return phrases[metricName.length % phrases.length];
    }
  };

  const topFactors = features.slice(0, 3);

  return (
    <div className="path-container">
      <h4>Path to Victory for {winner}</h4>
      <p style={{fontSize: '0.9rem', color: '#555', marginBottom: '15px'}}>
        Based on our AI analysis, these are the critical factors determining this outcome:
      </p>
      <ul className="path-steps">
        {topFactors.map((item, index) => (
          <li key={index} className="path-step-item">
            <span className="path-icon">✔</span>
            {generateStrategy(item)}.
          </li>
        ))}
      </ul>
    </div>
  );
}

export default PathToVictory;