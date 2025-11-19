import React from 'react';
import './components.css';

function ShapExplanation({ features }) {
  if (!features || features.length === 0) return null;

  // Convert snake_case feature names to readable format
  const formatFeatureName = (name) => {
    return name
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase())
      .replace(/Ewm/g, '(Recent)')
      .replace(/Avg/g, '(Avg)')
      .replace(/Pct/g, '%')
      .replace(/Fg/g, 'FG')
      .replace(/Ppg/g, 'PPG')
      .replace(/Opp/g, 'Opponent');
  };

  return (
    <div className="Shap-container">
      <h3>Keys to the Game</h3>
      <p style={{ fontSize: '0.9em', color: '#666', marginBottom: '15px' }}>
        Top factors influencing the prediction. <strong style={{ color: 'green' }}>Positive values</strong> favor the home team, <strong style={{ color: 'red' }}>negative values</strong> favor the away team.
      </p>
      <ul className="Shap-list">
        {features.map((item, index) => (
          <li key={index} className="Shap-item">
            <span className="Feature-name">{formatFeatureName(item.feature)}</span>
            <span className={`Feature-impact ${item.type === 'positive' ? 'positive' : 'negative'}`}>
              {item.impact > 0 ? '+' : ''}{item.impact.toFixed(3)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default ShapExplanation;
