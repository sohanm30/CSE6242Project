import React from 'react';
import './components.css';

function PathToVictory({ teamName, steps }) {
  return (
    <div className="path-container">
      <h4>{`Path to Victory for ${teamName}`}</h4>
      <ul className="path-steps">
        {steps.map((step, index) => (
          <li key={index}>{step}</li>
        ))}
      </ul>
    </div>
  );
}

export default PathToVictory;
