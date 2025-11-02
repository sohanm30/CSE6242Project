"""
NBA GamePlan - Modern NBA Preprocessing (2020-2025 Focus)
Team 29: Focus on recent seasons with exponentially weighted features and player stats
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("NBA GAMEPLAN - MODERN NBA PREPROCESSING (2020-2025)")
print("="*80)

# ============================================================================
# 1. LOAD AND FILTER TO MODERN ERA
# ============================================================================
print("\n[1] LOADING DATA (2020-2025 FOCUS)...")
print("-"*80)

# Load team statistics
team_stats = pd.read_csv('data/TeamStatistics.csv')
print(f"✓ Loaded {len(team_stats):,} total team game records")

# Parse dates
team_stats['gameDate'] = pd.to_datetime(team_stats['gameDate'], format='ISO8601', errors='coerce', utc=True)
team_stats['gameDate'] = team_stats['gameDate'].dt.tz_localize(None)
team_stats = team_stats.dropna(subset=['gameDate']).copy()

# Extract season
team_stats['year'] = team_stats['gameDate'].dt.year
team_stats['month'] = team_stats['gameDate'].dt.month
team_stats['season'] = team_stats['year']
team_stats.loc[team_stats['month'] >= 7, 'season'] = team_stats.loc[team_stats['month'] >= 7, 'year'] + 1

# FOCUS ON MODERN NBA: 2020-2025 only
team_stats = team_stats[team_stats['season'] >= 2020].copy()
print(f"✓ Filtered to 2020-2025: {len(team_stats):,} records")

# Get active teams (teams that played in 2024-25 season)
active_teams = team_stats[team_stats['season'] >= 2024]['teamName'].unique()
print(f"✓ Found {len(active_teams)} active teams in 2024-25 season")

# Filter to only active teams
team_stats = team_stats[team_stats['teamName'].isin(active_teams)].copy()
print(f"✓ Kept only active teams: {len(team_stats):,} records")

print(f"\nActive teams: {sorted(active_teams)[:10]}... ({len(active_teams)} total)")

# ============================================================================
# 2. DATA CLEANING
# ============================================================================
print("\n[2] CLEANING DATA...")
print("-"*80)

# Remove records with missing critical data
initial_count = len(team_stats)
team_stats = team_stats.dropna(subset=['teamScore', 'opponentScore', 'win'])
print(f"✓ Removed {initial_count - len(team_stats)} records with missing scores/outcomes")

# Fill missing advanced stats with 0
advanced_cols = ['pointsFastBreak', 'pointsFromTurnovers', 'pointsInThePaint',
                 'pointsSecondChance', 'benchPoints', 'biggestLead']
for col in advanced_cols:
    if col in team_stats.columns:
        team_stats[col] = team_stats[col].fillna(0)

print(f"✓ Final dataset: {len(team_stats):,} clean records")

# ============================================================================
# 3. BASIC FEATURES
# ============================================================================
print("\n[3] CREATING BASIC FEATURES...")
print("-"*80)

team_stats['home'] = team_stats['home'].astype(int)
team_stats['point_diff'] = team_stats['teamScore'] - team_stats['opponentScore']

# Efficiency metrics
team_stats['efg_pct'] = (team_stats['fieldGoalsMade'] + 0.5 * team_stats['threePointersMade']) / team_stats['fieldGoalsAttempted']
team_stats['efg_pct'] = team_stats['efg_pct'].fillna(0)

team_stats['ts_pct'] = team_stats['teamScore'] / (2 * (team_stats['fieldGoalsAttempted'] + 0.44 * team_stats['freeThrowsAttempted']))
team_stats['ts_pct'] = team_stats['ts_pct'].fillna(0)

team_stats['possessions'] = team_stats['fieldGoalsAttempted'] + 0.44 * team_stats['freeThrowsAttempted'] - team_stats['reboundsOffensive'] + team_stats['turnovers']
team_stats['tov_rate'] = team_stats['turnovers'] / team_stats['possessions']
team_stats['tov_rate'] = team_stats['tov_rate'].fillna(0)

team_stats['oreb_pct'] = team_stats['reboundsOffensive'] / (team_stats['reboundsOffensive'] + team_stats['reboundsDefensive'])
team_stats['oreb_pct'] = team_stats['oreb_pct'].fillna(0)

team_stats['pace'] = team_stats['possessions']

print(f"✓ Created 5 efficiency metrics")

# ============================================================================
# 4. EXPONENTIALLY WEIGHTED MOVING AVERAGES (Recent games matter more!)
# ============================================================================
print("\n[4] CREATING EXPONENTIALLY WEIGHTED MOVING AVERAGES...")
print("-"*80)

# Sort by team and date
team_stats = team_stats.sort_values(['teamId', 'gameDate']).reset_index(drop=True)

# Features to create EWMA for
rolling_features = [
    'teamScore', 'assists', 'reboundsTotal', 'steals', 'blocks', 'turnovers',
    'fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage',
    'pointsInThePaint', 'pointsFastBreak', 'benchPoints',
    'efg_pct', 'ts_pct', 'tov_rate', 'pace'
]

# Use EWMA with span=10 (more weight on recent games)
for feature in rolling_features:
    if feature in team_stats.columns:
        col_name = f'{feature}_ewm'
        team_stats[col_name] = team_stats.groupby('teamId')[feature].transform(
            lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
        )

print(f"✓ Created {len(rolling_features)} exponentially weighted features")

# Win percentage (EWMA)
team_stats['win_pct_ewm'] = team_stats.groupby('teamId')['win'].transform(
    lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
)

# Streak (before current game)
team_stats['streak'] = 0
for team_id in team_stats['teamId'].unique():
    team_mask = team_stats['teamId'] == team_id
    team_games = team_stats[team_mask].copy()

    streak = []
    current_streak = 0
    for win in team_games['win'].values:
        streak.append(current_streak)
        if win == 1:
            current_streak = max(1, current_streak + 1) if current_streak >= 0 else 1
        else:
            current_streak = min(-1, current_streak - 1) if current_streak <= 0 else -1

    team_stats.loc[team_mask, 'streak'] = streak

print("✓ Created win percentage and streak features (EWMA weighted)")

# ============================================================================
# 5. REST DAYS & SCHEDULE
# ============================================================================
print("\n[5] CREATING REST DAYS & SCHEDULE FEATURES...")
print("-"*80)

team_stats['days_rest'] = team_stats.groupby('teamId')['gameDate'].diff().dt.days.fillna(3)
team_stats['is_back_to_back'] = (team_stats['days_rest'] <= 1).astype(int)

team_stats['season_part'] = pd.cut(
    team_stats.groupby(['teamId', 'season']).cumcount(),
    bins=[0, 27, 55, 82],
    labels=['early', 'mid', 'late'],
    include_lowest=True
)
team_stats['season_part'] = team_stats['season_part'].astype(str)

print("✓ Created rest days and schedule features")

# ============================================================================
# 6. OPPONENT FEATURES (EWMA)
# ============================================================================
print("\n[6] CREATING OPPONENT STRENGTH FEATURES...")
print("-"*80)

# Create lookup of team EWMA stats
team_lookup = team_stats.groupby(['teamId', 'gameDate']).agg({
    'teamScore_ewm': 'first',
    'assists_ewm': 'first',
    'reboundsTotal_ewm': 'first',
    'fieldGoalsPercentage_ewm': 'first',
    'win_pct_ewm': 'first'
}).reset_index()

# Merge opponent stats
team_stats = team_stats.merge(
    team_lookup.rename(columns={
        'teamId': 'opponentTeamId',
        'teamScore_ewm': 'opp_score_ewm',
        'assists_ewm': 'opp_assists_ewm',
        'reboundsTotal_ewm': 'opp_rebounds_ewm',
        'fieldGoalsPercentage_ewm': 'opp_fg_pct_ewm',
        'win_pct_ewm': 'opp_win_pct_ewm'
    }),
    on=['opponentTeamId', 'gameDate'],
    how='left'
)

print("✓ Created opponent strength features (EWMA weighted)")

# ============================================================================
# 7. HEAD-TO-HEAD HISTORY
# ============================================================================
print("\n[7] CREATING HEAD-TO-HEAD FEATURES...")
print("-"*80)

team_stats['matchup_id'] = team_stats.apply(
    lambda x: f"{min(x['teamId'], x['opponentTeamId'])}_{max(x['teamId'], x['opponentTeamId'])}",
    axis=1
)

team_stats['h2h_win_pct'] = 0.5
for matchup in team_stats['matchup_id'].unique():
    matchup_mask = team_stats['matchup_id'] == matchup
    matchup_games = team_stats[matchup_mask].sort_values('gameDate').reset_index(drop=False)

    for i in range(len(matchup_games)):
        original_idx = matchup_games.loc[i, 'index']

        if i > 0:
            prev_games = matchup_games.iloc[:i]
            recent_prev = prev_games.tail(5)
            current_team = matchup_games.loc[i, 'teamId']
            wins = (recent_prev['teamId'] == current_team) & (recent_prev['win'] == 1)
            team_stats.loc[original_idx, 'h2h_win_pct'] = wins.sum() / len(recent_prev) if len(recent_prev) > 0 else 0.5

print("✓ Created head-to-head history features")

# ============================================================================
# 8. SAVE PREPROCESSED DATA
# ============================================================================
print("\n[8] SAVING MODERN NBA PREPROCESSED DATA...")
print("-"*80)

feature_columns = [
    'gameId', 'gameDate', 'season', 'teamId', 'opponentTeamId',
    'teamCity', 'teamName', 'opponentTeamCity', 'opponentTeamName',
    'home', 'win', 'teamScore', 'opponentScore', 'point_diff',

    # Current game stats (for reference)
    'assists', 'reboundsTotal', 'steals', 'blocks', 'turnovers',
    'fieldGoalsPercentage', 'threePointersPercentage',
    'pointsInThePaint', 'pointsFastBreak', 'benchPoints',

    # Engineered features (EWMA)
    'days_rest', 'is_back_to_back', 'month', 'season_part', 'streak',
    'win_pct_ewm', 'h2h_win_pct',

    # Team EWMA stats
    'teamScore_ewm', 'assists_ewm', 'reboundsTotal_ewm',
    'steals_ewm', 'blocks_ewm', 'turnovers_ewm',
    'fieldGoalsPercentage_ewm', 'threePointersPercentage_ewm',
    'pointsInThePaint_ewm', 'pointsFastBreak_ewm', 'benchPoints_ewm',
    'efg_pct_ewm', 'ts_pct_ewm', 'tov_rate_ewm', 'pace_ewm',

    # Opponent features (EWMA)
    'opp_score_ewm', 'opp_assists_ewm', 'opp_rebounds_ewm',
    'opp_fg_pct_ewm', 'opp_win_pct_ewm'
]

feature_columns = [col for col in feature_columns if col in team_stats.columns]
processed_data = team_stats[feature_columns].copy()

# Remove games without sufficient history
processed_data = processed_data.dropna(subset=['teamScore_ewm', 'win_pct_ewm'])

output_file = 'data/processed_modern_nba.csv'
processed_data.to_csv(output_file, index=False)
print(f"✓ Saved {len(processed_data):,} processed records to {output_file}")

print("\n" + "="*80)
print("MODERN NBA PREPROCESSING COMPLETE")
print("="*80)
print(f"\nTotal Features: {len(feature_columns)}")
print(f"Total Games: {len(processed_data):,}")
print(f"Date Range: {processed_data['gameDate'].min()} to {processed_data['gameDate'].max()}")
print(f"Seasons: {processed_data['season'].min()}-{processed_data['season'].max()}")
print(f"Active Teams: {len(processed_data['teamName'].unique())}")
print(f"\n✓ Ready for player feature integration and modeling!")
print(f"\n💡 KEY IMPROVEMENTS:")
print(f"  - Focused on 2020-2025 (modern NBA era)")
print(f"  - Only active teams (no Bobcats, etc.)")
print(f"  - Exponentially weighted averages (recent games matter more!)")
print(f"  - Next: Run player features script for final enhancement")
