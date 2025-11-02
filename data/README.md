# Data Directory

## Raw Data Files (Keep in GitHub)

These files should be committed to the repo:
- `TeamStatistics.csv` - 33 MB
- `PlayerStatistics.csv` - 303 MB
- `Games.csv` - 9.5 MB
- `Players.csv` - 524 KB
- `TeamHistories.csv` - 6.8 KB
- `LeagueSchedule24_25.csv` - 144 KB
- `LeagueSchedule25_26.csv` - 182 KB

**Source:** Kaggle - Historical NBA data and player box scores

## Generated Files (In .gitignore)

These are created by running `run_modern_pipeline.py`:
- `processed_modern_nba.csv` - Preprocessed team data
- `processed_modern_nba_with_players.csv` - With player features
- `current_player_teams.csv` - Player-team mappings

Your teammates will regenerate these files when they run the pipeline.
