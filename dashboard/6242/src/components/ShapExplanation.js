import React, { useState } from 'react';
import './components.css';

function ShapExplanation({ features }) {
  const [hoveredItem, setHoveredItem] = useState(null);

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

  const formatValue = (val) => {
    if (typeof val === 'number') {
      return val.toFixed(2);
    }
    return val;
  };

  const filteredFeatures = features.filter(f => f.feature.toLowerCase() !== 'home');
  const maxImpact = Math.max(...filteredFeatures.map(f => Math.abs(f.impact))) || 1;

  const handleMouseEnter = (e, item) => {
    setHoveredItem({
      item: item,
      x: e.clientX,
      y: e.clientY,
    });
  };

  const handleMouseMove = (e) => {
    if (hoveredItem) {
      setHoveredItem(prev => ({
        ...prev,
        x: e.clientX,
        y: e.clientY,
      }));
    }
  };

  const handleMouseLeave = () => {
    setHoveredItem(null);
  };

  return (
    <div className="Shap-container">
      <h3>Keys to the Game</h3>
      
      <p style={{ fontSize: '0.9em', color: '#666', marginBottom: '20px' }}>
        Top factors driving the prediction. 
        <span style={{color: '#007bff', fontWeight: 'bold', marginLeft: '5px'}}>Left (Blue)</span> favors Home, 
        <span style={{color: '#dc3545', fontWeight: 'bold', marginLeft: '5px'}}>Right (Red)</span> favors Away.
        <br/>
        <span style={{ fontSize: '0.85em', color: '#999', fontStyle: 'italic' }}>
          Tip: Hover over bars for detailed stats
        </span>
      </p>
      
      <div className="Shap-chart" onMouseLeave={handleMouseLeave}>
        {filteredFeatures.map((item, index) => {
          const isHomeAdvantage = item.impact > 0;
          const widthPct = (Math.abs(item.impact) / maxImpact) * 100;
          
          return (
            <div key={index} className="chart-row">
              <div className="chart-label">
                {formatFeatureName(item.feature)}
              </div>

              <div className="bar-container">
                <div className="bar-side left">
                  {isHomeAdvantage && (
                    <div 
                      className="bar home" 
                      style={{ width: `${widthPct}%` }}
                      onMouseEnter={(e) => handleMouseEnter(e, item)}
                      onMouseMove={handleMouseMove}
                      onMouseLeave={handleMouseLeave}
                    ></div>
                  )}
                </div>

                <div className="center-line"></div>

                <div className="bar-side right">
                  {!isHomeAdvantage && (
                    <div 
                      className="bar away" 
                      style={{ width: `${widthPct}%` }}
                      onMouseEnter={(e) => handleMouseEnter(e, item)}
                      onMouseMove={handleMouseMove}
                      onMouseLeave={handleMouseLeave}
                    ></div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {hoveredItem && (
        <div 
          className="Shap-custom-tooltip"
          style={{ 
            top: hoveredItem.y - 10, 
            left: hoveredItem.x + 15 
          }}
        >
          <div className="tooltip-header">
            {formatFeatureName(hoveredItem.item.feature)}
          </div>
          <div className="tooltip-row">
            <span>Impact:</span>
            <span className={hoveredItem.item.impact > 0 ? 'val-home' : 'val-away'}>
              {hoveredItem.item.impact > 0 ? '+' : ''}{hoveredItem.item.impact.toFixed(3)}
            </span>
          </div>
          <div className="tooltip-row">
            <span>Actual Value:</span>
            <span className="val-neutral">
              {formatValue(hoveredItem.item.value)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export default ShapExplanation;