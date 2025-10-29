import React from 'react';
import './components.css';

function TeamMetrics({ metrics }) {
  return (
    <div className="metrics-container">
      <h4>Tale of the Tape</h4>
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Home</th>
            <th>Away</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map(metric => (
            <tr key={metric.name}>
              <td>{metric.name}</td>
              <td>{metric.homeValue}</td>
              <td>{metric.awayValue}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default TeamMetrics;
