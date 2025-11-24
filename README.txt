NBA GamePlan: Matchup Predictor
Team 29 - CSE 6242 Project

DESCRIPTION:

NBA GamePlan is a machine learning-powered NBA game prediction system with an 
interactive web dashboard. The system predicts win probabilities and point 
spreads for NBA matchups using XGBoost models trained on modern NBA data 
(2020-2025 seasons).

Key Features:
- Real-time game predictions with win probability and point spread
- Interactive React dashboard for easy team selection and result visualization
- SHAP explanations showing key factors influencing predictions
- Season statistics comparison
- Flask REST API for backend predictions
- XGBoost models trained on exponentially weighted moving averages (EWMA) 
  features and player-level statistics

The system uses two XGBoost models:
1. Classification model: Predicts win/loss with probability
2. Regression model: Predicts point spread

Models are trained on 2020-2025 NBA data with features including team statistics,
opponent strength, head-to-head history, player performance metrics, and 
schedule context (rest days, back-to-back games, etc.).

INSTALLATION:

Prerequisites:
- Python 3.8 or higher
- Node.js 14+ and npm
- pip (Python package manager)

Step 1: Install Python Dependencies
From the project root directory (CODE/), run:

    pip install -r requirements-api.txt

This installs: Flask, flask-cors, pandas, numpy, xgboost, shap

Step 2: Install Frontend Dependecdncies
Navigate to the dashboard directory and install Node.js dependencies:

    cd dashboard/6242
    npm install --legacy-peer-deps

Note: Use --legacy-peer-deps flag due to React version compatibility.

Step 3: Verify Installation
Verify that required files exist:

    ls models/xgb_modern_*.json
    ls data/processed_modern_nba_with_players.csv

You should see both model files and the processed data file.

EXECUTION

The application consists of two components that must run simultaneously:
1. Flask API backend (port 5001)
2. React frontend (port 3000)

Step 1: Start the Flask API Server
Open a terminal and navigate to the project root (CODE/ directory):

    python app.py

You should see output indicating the server is running on http://localhost:5001

Step 2: Start the React Dashboard
Open a NEW terminal window and navigate to the dashboard directory:

    cd CODE/dashboard/6242
    npm start

The React development server will start and automatically open your browser 
at http://localhost:3000

Step 3: Use the Application
1. In the browser, select a HOME team from the dropdown menu
2. Select an AWAY team from the dropdown menu
3. Click the "Analyze Matchup" button
4. View the prediction results including:
   - Predicted winner with win probability
   - Predicted point spread
   - Season statistics comparison
   - SHAP feature explanations

Demo Example:
- Home Team: Boston Celtics
- Away Team: Los Angeles Lakers
- Click "Analyze Matchup" to see prediction results

Note: Both servers must be running for the application to work. The frontend 
communicates with the backend API at http://localhost:5001.


OPTIONAL: Retraining Models

To retrain models with updated data:

    python run_modern_pipeline.py

This runs the complete pipeline:
1. Preprocessing (02c_modern_preprocessing.py)
2. Player features (02d_add_player_features_modern.py)
3. Model training (03b_train_modern_model.py)

Individual scripts can also be run separately from the model_scripts/ directory.

