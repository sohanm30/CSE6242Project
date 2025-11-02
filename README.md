# NBA GamePlan: Matchup Predictor

A machine learning-powered NBA game prediction system with an interactive dashboard. Predicts win probabilities and point spreads for NBA matchups using XGBoost models trained on modern NBA data (2020-2025).

## Project Structure

```
CSE6242Project/
├── app.py                          # Flask API server
├── requirements-api.txt            # Python dependencies for API
├── models/                         # Trained ML models
│   ├── xgb_modern_classification.json
│   └── xgb_modern_regression.json
├── data/                          # Processed NBA data
│   └── processed_modern_nba_with_players.csv
├── model_scripts/                 # Training and preprocessing scripts
│   ├── 02c_modern_preprocessing.py
│   ├── 02d_add_player_features_modern.py
│   ├── 03b_train_modern_model.py
│   └── 04_predict_game_modern.py
└── dashboard/6242/                # React frontend
    ├── src/
    ├── public/
    └── package.json
```

## Prerequisites

- **Python 3.8+** (with pandas, numpy, xgboost, flask, flask-cors)
- **Node.js 14+** and npm
- Trained models in `models/` directory
- Processed data in `data/` directory

## Setup Instructions

### 1. Backend Setup (Flask API)

```bash
# From project root directory
cd /path/to/CSE6242Project

# Install Python dependencies
pip install -r requirements-api.txt

# Verify models and data exist
ls models/xgb_modern_*.json
ls data/processed_modern_nba_with_players.csv
```

### 2. Frontend Setup (React Dashboard)

```bash
# Navigate to dashboard directory
cd dashboard/6242

# Install Node dependencies (one-time setup)
npm install
```

## Running the Application

### Start the Flask API Server

```bash
# From project root directory
python app.py
```

The API will start on **http://localhost:5001**

You should see:
```
 * Running on http://127.0.0.1:5001
 * Running on http://10.x.x.x:5001
```

### Start the React Dashboard

**In a separate terminal:**

```bash
# From dashboard directory
cd dashboard/6242
npm start
```

The dashboard will open automatically at **http://localhost:3000**

## Using the Application

1. **Select Teams**: Choose a home team and away team from the dropdowns
2. **Analyze Matchup**: Click the "Analyze Matchup" button
3. **View Results**: The modal displays:
   - Predicted winner with win probability
   - Predicted point spread
   - Tale of the Tape (2025-26 season stats comparison)

## API Endpoints

### `GET /health`
Health check and system information

### `GET /teams`
Returns list of available NBA teams

**Response:**
```json
{
  "teams": ["76ers", "Bucks", "Bulls", ...]
}
```

### `POST /predict`
Generate prediction for a matchup

**Request Body:**
```json
{
  "home_team": "Boston Celtics",
  "away_team": "Los Angeles Lakers"
}
```

**Response:**
```json
{
  "matchup": {
    "home_team": "Celtics",
    "away_team": "Lakers"
  },
  "probabilities": {
    "home": 0.657,
    "away": 0.343
  },
  "prediction": {
    "winner": "Celtics",
    "confidence": 0.657,
    "predicted_spread": 4.7,
    "consistent": true
  },
  "season_stats": {
    "home": {
      "wins": 3,
      "losses": 1,
      "ppg": 118.5,
      "fg_pct": 0.475,
      "three_pct": 0.384,
      "rebounds": 45.2,
      "assists": 26.3
    },
    "away": { ... }
  }
}
```

## Features

- **ML Models**: XGBoost classification (win probability) and regression (point spread)
- **Modern NBA Data**: Trained on 2020-2025 seasons with EWMA features
- **Player Features**: Top 5 player stats per team integrated
- **Season Stats**: Real-time 2025-26 season statistics (games after Oct 21, 2025)
- **Team Aliases**: Supports full team names ("Los Angeles Lakers") or short names ("Lakers")
- **CORS Enabled**: Frontend can connect from any origin

## Troubleshooting

### API won't start
- Verify models exist: `ls models/xgb_modern_*.json`
- Check data exists: `ls data/processed_modern_nba_with_players.csv`
- Ensure port 5001 is available: `lsof -i :5001`

### Frontend errors
- Ensure API is running on port 5001
- Check browser console for CORS errors
- Verify `npm install` completed successfully

### "Unknown team" errors
- Use team names from `GET /teams` endpoint
- Full names work: "Boston Celtics", "Los Angeles Lakers"
- Short names work: "Celtics", "Lakers"

### No season stats showing (0-0 records)
- Data may not include 2025-26 season games yet
- Check date filter in `app.py` line 351: `>= "2025-10-21"`
- Verify `gameDate` column in CSV is properly formatted

## Model Training (Optional)

To retrain models with updated data:

```bash
# Run complete pipeline
python run_modern_pipeline.py

# Or run individual scripts:
python model_scripts/02c_modern_preprocessing.py
python model_scripts/02d_add_player_features_modern.py
python model_scripts/03b_train_modern_model.py
```

## Tech Stack

- **Backend**: Flask, XGBoost, pandas, numpy
- **Frontend**: React 19, react-nba-logos
- **ML**: XGBoost (classification + regression)
- **Data**: NBA API data (2020-2025)

## Team

Team 29 - CSE 6242 Project