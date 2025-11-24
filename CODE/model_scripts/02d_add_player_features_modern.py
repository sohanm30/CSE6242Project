import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

player_stats = pd.read_csv('data/PlayerStatistics.csv')

team_data = pd.read_csv('data/processed_modern_nba.csv')
team_data['gameDate'] = pd.to_datetime(team_data['gameDate'])

player_stats['gameDate'] = pd.to_datetime(player_stats['gameDate'], format='ISO8601', errors='coerce', utc=True)
player_stats['gameDate'] = player_stats['gameDate'].dt.tz_localize(None)
player_stats = player_stats.dropna(subset=['gameDate'])

player_stats['year'] = player_stats['gameDate'].dt.year
player_stats = player_stats[player_stats['year'] >= 2020].copy()

player_stats['numMinutes'] = player_stats['numMinutes'].fillna(0)
player_stats['points'] = player_stats['points'].fillna(0)
player_stats['assists'] = player_stats['assists'].fillna(0)
player_stats['reboundsTotal'] = player_stats['reboundsTotal'].fillna(0)
player_stats['plusMinusPoints'] = player_stats['plusMinusPoints'].fillna(0)

player_stats = player_stats[player_stats['numMinutes'] > 0].copy()

player_stats = player_stats.sort_values(['personId', 'gameDate']).reset_index(drop=True)

stat_columns = ['points', 'assists', 'reboundsTotal', 'numMinutes', 'plusMinusPoints']

for col in stat_columns:
    player_stats[f'{col}_ewm'] = player_stats.groupby('personId')[col].transform(
        lambda x: x.shift(1).ewm(span=5, min_periods=1).mean()
    )

top5_points_avg = []
top5_assists_avg = []
top5_rebounds_avg = []
top5_plusminus_avg = []
star_points_max = []

player_team_date = player_stats.groupby(['playerteamName', 'gameDate']).apply(
    lambda x: x.nlargest(5, 'numMinutes_ewm')[['points_ewm', 'assists_ewm', 'reboundsTotal_ewm', 'plusMinusPoints_ewm']].mean()
).reset_index()

total_games = len(team_data)

for idx, row in team_data.iterrows():
    game_date = row['gameDate']
    team_name = row['teamName']

    date_window_start = game_date - timedelta(days=10)

    recent_players = player_stats[
        (player_stats['playerteamName'] == team_name) &
        (player_stats['gameDate'] < game_date) &
        (player_stats['gameDate'] >= date_window_start)
    ].copy()

    if len(recent_players) == 0:
        top5_points_avg.append(20.0)
        top5_assists_avg.append(5.0)
        top5_rebounds_avg.append(7.0)
        top5_plusminus_avg.append(0.0)
        star_points_max.append(25.0)
        continue

    top_players = recent_players.groupby('personId').agg({
        'numMinutes_ewm': 'mean',
        'points_ewm': 'mean',
        'assists_ewm': 'mean',
        'reboundsTotal_ewm': 'mean',
        'plusMinusPoints_ewm': 'mean'
    }).reset_index()

    top_players = top_players.nlargest(5, 'numMinutes_ewm')

    top5_points_avg.append(top_players['points_ewm'].mean())
    top5_assists_avg.append(top_players['assists_ewm'].mean())
    top5_rebounds_avg.append(top_players['reboundsTotal_ewm'].mean())
    top5_plusminus_avg.append(top_players['plusMinusPoints_ewm'].mean())
    star_points_max.append(top_players['points_ewm'].max())

team_data['top5_points_avg'] = top5_points_avg
team_data['top5_assists_avg'] = top5_assists_avg
team_data['top5_rebounds_avg'] = top5_rebounds_avg
team_data['top5_plusminus_avg'] = top5_plusminus_avg
team_data['star_points_max'] = star_points_max

output_file = 'data/processed_modern_nba_with_players.csv'
team_data.to_csv(output_file, index=False)
