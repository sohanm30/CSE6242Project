import React from 'react';
import './components.css';

function ShapExplanation({ features }) {
  if (!features || features.length === 0) return null;

  return (
    <div className="Shap-container">
      <h3>Keys to the Game</h3>
      <ul className="Shap-list">
        {features.map((feature, index) => (
          <li key={index} className="Shap-item">
            <span className="Feature-name">{feature.name}</span>
            <span className={`Feature-impact ${feature.impact > 0 ? 'positive' : 'negative'}`}>
              {feature.impact > 0 ? '+' : ''}{feature.impact.toFixed(2)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default ShapExplanation;
