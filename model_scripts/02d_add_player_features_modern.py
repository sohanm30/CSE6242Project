"""
NBA GamePlan - Add Player Features (Modern NBA)
Team 29: Integrate top player stats into modern team data
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("NBA GAMEPLAN - ADDING PLAYER FEATURES (MODERN NBA)")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA...")
print("-"*80)

# Load player statistics
print("Loading PlayerStatistics.csv...")
player_stats = pd.read_csv('data/PlayerStatistics.csv')
print(f"✓ Loaded {len(player_stats):,} player game records")

# Load modern processed team data
team_data = pd.read_csv('data/processed_modern_nba.csv')
team_data['gameDate'] = pd.to_datetime(team_data['gameDate'])
print(f"✓ Loaded {len(team_data):,} processed team games (2020-2025)")

# ============================================================================
# 2. FILTER PLAYER DATA TO MODERN ERA
# ============================================================================
print("\n[2] FILTERING PLAYER DATA TO 2020-2025...")
print("-"*80)

# Parse dates
player_stats['gameDate'] = pd.to_datetime(player_stats['gameDate'], format='ISO8601', errors='coerce', utc=True)
player_stats['gameDate'] = player_stats['gameDate'].dt.tz_localize(None)
player_stats = player_stats.dropna(subset=['gameDate'])

# Filter to 2020+
player_stats['year'] = player_stats['gameDate'].dt.year
player_stats = player_stats[player_stats['year'] >= 2020].copy()
print(f"✓ Filtered to 2020+: {len(player_stats):,} player records")

# Fill missing values
player_stats['numMinutes'] = player_stats['numMinutes'].fillna(0)
player_stats['points'] = player_stats['points'].fillna(0)
player_stats['assists'] = player_stats['assists'].fillna(0)
player_stats['reboundsTotal'] = player_stats['reboundsTotal'].fillna(0)
player_stats['plusMinusPoints'] = player_stats['plusMinusPoints'].fillna(0)

# Only keep players who actually played
player_stats = player_stats[player_stats['numMinutes'] > 0].copy()
print(f"✓ Kept players with minutes > 0: {len(player_stats):,} records")

# ============================================================================
# 3. CALCULATE PLAYER ROLLING AVERAGES (EWMA)
# ============================================================================
print("\n[3] CALCULATING PLAYER EXPONENTIALLY WEIGHTED AVERAGES...")
print("-"*80)

player_stats = player_stats.sort_values(['personId', 'gameDate']).reset_index(drop=True)

stat_columns = ['points', 'assists', 'reboundsTotal', 'numMinutes', 'plusMinusPoints']

print(f"Calculating EWMA for {len(stat_columns)} player stats...")

for col in stat_columns:
    player_stats[f'{col}_ewm'] = player_stats.groupby('personId')[col].transform(
        lambda x: x.shift(1).ewm(span=5, min_periods=1).mean()
    )

print(f"✓ Created EWMA for player stats")

# ============================================================================
# 4. AGGREGATE TOP PLAYERS TO TEAM LEVEL (FASTER VERSION)
# ============================================================================
print("\n[4] AGGREGATING TOP PLAYERS TO TEAM LEVEL...")
print("-"*80)

print("Processing games in batches for speed...")

# Initialize lists
top5_points_avg = []
top5_assists_avg = []
top5_rebounds_avg = []
top5_plusminus_avg = []
star_points_max = []

# Group player stats by team and date for faster lookup
player_team_date = player_stats.groupby(['playerteamName', 'gameDate']).apply(
    lambda x: x.nlargest(5, 'numMinutes_ewm')[['points_ewm', 'assists_ewm', 'reboundsTotal_ewm', 'plusMinusPoints_ewm']].mean()
).reset_index()

total_games = len(team_data)

for idx, row in team_data.iterrows():
    if idx % 1000 == 0:
        print(f"  Processing game {idx:,} / {total_games:,} ({idx/total_games*100:.1f}%)...")

    game_date = row['gameDate']
    team_name = row['teamName']

    # Look for recent player data (within 10 days before game)
    date_window_start = game_date - timedelta(days=10)

    recent_players = player_stats[
        (player_stats['playerteamName'] == team_name) &
        (player_stats['gameDate'] < game_date) &
        (player_stats['gameDate'] >= date_window_start)
    ].copy()

    if len(recent_players) == 0:
        # No recent data - use defaults
        top5_points_avg.append(20.0)  # Default average
        top5_assists_avg.append(5.0)
        top5_rebounds_avg.append(7.0)
        top5_plusminus_avg.append(0.0)
        star_points_max.append(25.0)
        continue

    # Get top 5 players by recent minutes
    top_players = recent_players.groupby('personId').agg({
        'numMinutes_ewm': 'mean',
        'points_ewm': 'mean',
        'assists_ewm': 'mean',
        'reboundsTotal_ewm': 'mean',
        'plusMinusPoints_ewm': 'mean'
    }).reset_index()

    top_players = top_players.nlargest(5, 'numMinutes_ewm')

    # Aggregate
    top5_points_avg.append(top_players['points_ewm'].mean())
    top5_assists_avg.append(top_players['assists_ewm'].mean())
    top5_rebounds_avg.append(top_players['reboundsTotal_ewm'].mean())
    top5_plusminus_avg.append(top_players['plusMinusPoints_ewm'].mean())
    star_points_max.append(top_players['points_ewm'].max())

print(f"✓ Aggregated player features for all {len(team_data):,} games")

# ============================================================================
# 5. ADD TO TEAM DATA
# ============================================================================
print("\n[5] ADDING PLAYER FEATURES TO TEAM DATA...")
print("-"*80)

team_data['top5_points_avg'] = top5_points_avg
team_data['top5_assists_avg'] = top5_assists_avg
team_data['top5_rebounds_avg'] = top5_rebounds_avg
team_data['top5_plusminus_avg'] = top5_plusminus_avg
team_data['star_points_max'] = star_points_max

print(f"✓ Added 5 player-level features:")
print(f"  - top5_points_avg: Average points from top 5 players")
print(f"  - top5_assists_avg: Average assists from top 5 players")
print(f"  - top5_rebounds_avg: Average rebounds from top 5 players")
print(f"  - top5_plusminus_avg: Average +/- from top 5 players")
print(f"  - star_points_max: Best scorer's recent average")

# ============================================================================
# 6. SAVE ENHANCED DATA
# ============================================================================
print("\n[6] SAVING ENHANCED MODERN NBA DATA...")
print("-"*80)

output_file = 'data/processed_modern_nba_with_players.csv'
team_data.to_csv(output_file, index=False)
print(f"✓ Saved enhanced data to {output_file}")

print("\n✓ Player Feature Summary:")
print(team_data[['top5_points_avg', 'top5_assists_avg', 'top5_rebounds_avg', 'star_points_max']].describe())

print("\n" + "="*80)
print("PLAYER FEATURES ADDED SUCCESSFULLY")
print("="*80)
print(f"\n✓ Modern NBA dataset with player features ready:")
print(f"  - Total games: {len(team_data):,}")
print(f"  - Total features: {len(team_data.columns)}")
print(f"  - Player features: 5")
print(f"  - Seasons: 2020-2025")
print(f"  - Active teams only")
print(f"  - Output: {output_file}")
