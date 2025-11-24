import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

team_stats = pd.read_csv('data/TeamStatistics.csv')

team_stats['gameDate'] = pd.to_datetime(team_stats['gameDate'], format='ISO8601', errors='coerce', utc=True)
team_stats['gameDate'] = team_stats['gameDate'].dt.tz_localize(None)
team_stats = team_stats.dropna(subset=['gameDate']).copy()

team_stats['year'] = team_stats['gameDate'].dt.year
team_stats['month'] = team_stats['gameDate'].dt.month
team_stats['season'] = team_stats['year']
team_stats.loc[team_stats['month'] >= 7, 'season'] = team_stats.loc[team_stats['month'] >= 7, 'year'] + 1

team_stats = team_stats[team_stats['season'] >= 2020].copy()

active_teams = team_stats[team_stats['season'] >= 2024]['teamName'].unique()
team_stats = team_stats[team_stats['teamName'].isin(active_teams)].copy()

initial_count = len(team_stats)
team_stats = team_stats.dropna(subset=['teamScore', 'opponentScore', 'win'])
advanced_cols = ['pointsFastBreak', 'pointsFromTurnovers', 'pointsInThePaint',
                 'pointsSecondChance', 'benchPoints', 'biggestLead']
for col in advanced_cols:
    if col in team_stats.columns:
        team_stats[col] = team_stats[col].fillna(0)

team_stats['home'] = team_stats['home'].astype(int)
team_stats['point_diff'] = team_stats['teamScore'] - team_stats['opponentScore']

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

team_stats = team_stats.sort_values(['teamId', 'gameDate']).reset_index(drop=True)

rolling_features = [
    'teamScore', 'assists', 'reboundsTotal', 'steals', 'blocks', 'turnovers',
    'fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage',
    'pointsInThePaint', 'pointsFastBreak', 'benchPoints',
    'efg_pct', 'ts_pct', 'tov_rate', 'pace'
]

for feature in rolling_features:
    if feature in team_stats.columns:
        col_name = f'{feature}_ewm'
        team_stats[col_name] = team_stats.groupby('teamId')[feature].transform(
            lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
        )

team_stats['win_pct_ewm'] = team_stats.groupby('teamId')['win'].transform(
    lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
)

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

team_stats['days_rest'] = team_stats.groupby('teamId')['gameDate'].diff().dt.days.fillna(3)
team_stats['is_back_to_back'] = (team_stats['days_rest'] <= 1).astype(int)

team_stats['season_part'] = pd.cut(
    team_stats.groupby(['teamId', 'season']).cumcount(),
    bins=[0, 27, 55, 82],
    labels=['early', 'mid', 'late'],
    include_lowest=True
)
team_stats['season_part'] = team_stats['season_part'].astype(str)

team_lookup = team_stats.groupby(['teamId', 'gameDate']).agg({
    'teamScore_ewm': 'first',
    'assists_ewm': 'first',
    'reboundsTotal_ewm': 'first',
    'fieldGoalsPercentage_ewm': 'first',
    'win_pct_ewm': 'first'
}).reset_index()

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

feature_columns = [
    'gameId', 'gameDate', 'season', 'teamId', 'opponentTeamId',
    'teamCity', 'teamName', 'opponentTeamCity', 'opponentTeamName',
    'home', 'win', 'teamScore', 'opponentScore', 'point_diff',

    'assists', 'reboundsTotal', 'steals', 'blocks', 'turnovers',
    'fieldGoalsPercentage', 'threePointersPercentage',
    'pointsInThePaint', 'pointsFastBreak', 'benchPoints',

    'days_rest', 'is_back_to_back', 'month', 'season_part', 'streak',
    'win_pct_ewm', 'h2h_win_pct',

    'teamScore_ewm', 'assists_ewm', 'reboundsTotal_ewm',
    'steals_ewm', 'blocks_ewm', 'turnovers_ewm',
    'fieldGoalsPercentage_ewm', 'threePointersPercentage_ewm',
    'pointsInThePaint_ewm', 'pointsFastBreak_ewm', 'benchPoints_ewm',
    'efg_pct_ewm', 'ts_pct_ewm', 'tov_rate_ewm', 'pace_ewm',

    'opp_score_ewm', 'opp_assists_ewm', 'opp_rebounds_ewm',
    'opp_fg_pct_ewm', 'opp_win_pct_ewm'
]

feature_columns = [col for col in feature_columns if col in team_stats.columns]
processed_data = team_stats[feature_columns].copy()

processed_data = processed_data.dropna(subset=['teamScore_ewm', 'win_pct_ewm'])

output_file = 'data/processed_modern_nba.csv'
processed_data.to_csv(output_file, index=False)

