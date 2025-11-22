import React from 'react';
import './components.css';

function TeamMetrics({ metrics, homeTeam, awayTeam }) {

  const parseMetricValue = (valStr) => {
    if (typeof valStr !== 'string') return 0;
    
    if (valStr.includes('-')) {
      const [w, l] = valStr.split('-').map(Number);
      if (!isNaN(w) && !isNaN(l) && (w + l) > 0) {
        return w / (w + l);
      }
    }

    if (valStr.includes('%')) {
      return parseFloat(valStr.replace('%', ''));
    }

    return parseFloat(valStr);
  };

  return (
    <div className="metrics-container">
      <h4>Tale of the Tape</h4>
      <table className="metrics-table">
        <thead>
          <tr>
            <th style={{textAlign: 'left'}}>Metric</th>
            <th>{homeTeam || 'Home'}</th>
            <th>{awayTeam || 'Away'}</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map(metric => {
            const homeVal = parseMetricValue(metric.homeValue);
            const awayVal = parseMetricValue(metric.awayValue);
            const max = Math.max(homeVal, awayVal) || 1;
            const homeWidth = (homeVal / max) * 100;
            const awayWidth = (awayVal / max) * 100;
            const homeColor = homeVal >= awayVal ? '#007bff' : '#ccc';
            const awayColor = awayVal >= homeVal ? '#dc3545' : '#ccc';

            return (
              <tr key={metric.name}>
                <td className="metric-name">{metric.name}</td>
                
                <td className="metric-cell">
                  <span className="metric-text">{metric.homeValue}</span>
                  {!isNaN(homeVal) && (
                    <div className="metric-bar-bg">
                      <div className="metric-bar" style={{width: `${homeWidth}%`, backgroundColor: homeColor}}></div>
                    </div>
                  )}
                </td>

                <td className="metric-cell">
                  <span className="metric-text">{metric.awayValue}</span>
                  {!isNaN(awayVal) && (
                    <div className="metric-bar-bg">
                      <div className="metric-bar" style={{width: `${awayWidth}%`, backgroundColor: awayColor}}></div>
                    </div>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default TeamMetrics;