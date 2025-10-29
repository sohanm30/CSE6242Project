export const FAKE_RESULTS = {
  prediction: {
    winner: 'Golden State Warriors',
    probability: 68,
  },
  explanation: [
    { name: '3-Point Percentage', impact: 0.15 },
    { name: 'Turnover Rate', impact: 0.12 },
    { name: 'Offensive Rebounding', impact: -0.08 },
    { name: 'Pace', impact: 0.05 },
    { name: 'Defensive Efficiency', impact: -0.03 },
  ],
  keyPlayers: {
    home: ['Stephen Curry', 'Klay Thompson', 'Draymond Green'],
    away: ['LeBron James', 'Anthony Davis', 'Russell Westbrook'],
  },
  metrics: [
    { name: 'Overall Record', homeValue: '53-29', awayValue: '33-49' },
    { name: 'Points Per Game', homeValue: '117.7', awayValue: '112.1' },
    { name: 'FG%', homeValue: '47.8%', awayValue: '46.9%' },
    { name: '3P%', homeValue: '38.0%', awayValue: '34.7%' },
  ],
  pathToVictory: {
    teamName: 'Los Angeles Lakers',
    steps: [
      'Improve 3-point shooting to over 37%.',
      'Limit turnovers to less than 12.',
      'Win the offensive rebound battle by at least 5.',
    ],
  },
};
