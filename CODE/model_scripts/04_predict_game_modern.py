import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    clf = xgb.XGBClassifier()
    clf.load_model('models/xgb_modern_classification.json')
    reg_spread = xgb.XGBRegressor()
    reg_spread.load_model('models/xgb_modern_regression.json')
except:
    clf = xgb.XGBClassifier()
    clf.load_model('models/xgb_classification.json')
    reg_spread = xgb.XGBRegressor()
    reg_spread.load_model('models/xgb_regression_spread.json')

try:
    team_data = pd.read_csv('data/processed_modern_nba_with_players.csv')
    has_player_features = True
except:
    try:
        team_data = pd.read_csv('data/processed_modern_nba.csv')
        has_player_features = False
    except:
        team_data = pd.read_csv('data/processed_team_data.csv')
        has_player_features = False

team_data['gameDate'] = pd.to_datetime(team_data['gameDate'])

try:
    feature_importance = pd.read_csv('results/feature_importance_modern.csv')
except:
    feature_importance = pd.read_csv('results/feature_importance.csv')

teams = sorted(team_data['teamName'].unique())

use_ewma = 'teamScore_ewm' in team_data.columns

if use_ewma:
    feature_suffix = '_ewm'
else:
    feature_suffix = '_roll10'

feature_columns = [
    'home', 'days_rest', 'is_back_to_back', 'month', 'streak',
    f'win_pct{feature_suffix}',
    f'teamScore{feature_suffix}', f'assists{feature_suffix}', f'reboundsTotal{feature_suffix}',
    f'steals{feature_suffix}', f'blocks{feature_suffix}', f'turnovers{feature_suffix}',
    f'fieldGoalsPercentage{feature_suffix}', f'threePointersPercentage{feature_suffix}',
    f'pointsInThePaint{feature_suffix}', f'pointsFastBreak{feature_suffix}', f'benchPoints{feature_suffix}',
    f'efg_pct{feature_suffix}', f'ts_pct{feature_suffix}', f'tov_rate{feature_suffix}', f'pace{feature_suffix}',
    f'opp_score{feature_suffix}', f'opp_assists{feature_suffix}', f'opp_rebounds{feature_suffix}',
    f'opp_fg_pct{feature_suffix}', f'opp_win_pct{feature_suffix}',
    'h2h_win_pct'
]

if has_player_features:
    player_features = ['top5_points_avg', 'top5_assists_avg', 'top5_rebounds_avg',
                      'top5_plusminus_avg', 'star_points_max']
    feature_columns.extend(player_features)

feature_columns.extend(['season_late', 'season_mid'])

def get_team_recent_stats(team_name, cutoff_date=None):
    if cutoff_date is None:
        cutoff_date = team_data['gameDate'].max()

    team_games = team_data[
        (team_data['teamName'] == team_name) &
        (team_data['gameDate'] <= cutoff_date)
    ].sort_values('gameDate', ascending=False)

    if len(team_games) == 0:
        return None

    return team_games.iloc[0]

def build_features_for_prediction(home_team, away_team):
    home_stats = get_team_recent_stats(home_team)
    away_stats = get_team_recent_stats(away_team)

    if home_stats is None or away_stats is None:
        return None

    features = {}

    features['home'] = 1
    features['days_rest'] = home_stats.get('days_rest', 2)
    features['is_back_to_back'] = int(home_stats.get('is_back_to_back', 0))
    features['month'] = datetime.now().month
    features['streak'] = home_stats.get('streak', 0)

    for stat in ['win_pct', 'teamScore', 'assists', 'reboundsTotal', 'steals', 'blocks',
                 'turnovers', 'fieldGoalsPercentage', 'threePointersPercentage',
                 'pointsInThePaint', 'pointsFastBreak', 'benchPoints',
                 'efg_pct', 'ts_pct', 'tov_rate', 'pace']:
        col_name = f'{stat}{feature_suffix}'
        features[col_name] = home_stats.get(col_name, 0)

    for stat in ['score', 'assists', 'rebounds', 'fg_pct', 'win_pct']:
        if stat == 'score':
            away_col = f'teamScore{feature_suffix}'
        elif stat == 'rebounds':
            away_col = f'reboundsTotal{feature_suffix}'
        elif stat == 'fg_pct':
            away_col = f'fieldGoalsPercentage{feature_suffix}'
        elif stat == 'win_pct':
            away_col = f'win_pct{feature_suffix}'
        else:
            away_col = f'{stat}{feature_suffix}'

        features[f'opp_{stat}{feature_suffix}'] = away_stats.get(away_col, 0)

    features['h2h_win_pct'] = home_stats.get('h2h_win_pct', 0.5)

    features['season_late'] = 0
    features['season_mid'] = 1

    if has_player_features:
        for pf in ['top5_points_avg', 'top5_assists_avg', 'top5_rebounds_avg',
                   'top5_plusminus_avg', 'star_points_max']:
            features[pf] = home_stats.get(pf, 0)

    return features, home_stats, away_stats

for i, team in enumerate(teams, 1):
    pass

home_input = input("\nEnter HOME team (number or name): ").strip()
try:
    home_idx = int(home_input) - 1
    home_team = teams[home_idx]
except:
    matching = [t for t in teams if home_input.lower() in t.lower()]
    if matching:
        home_team = matching[0]
    else:
        home_team = teams[0]

away_input = input("Enter AWAY team (number or name): ").strip()
try:
    away_idx = int(away_input) - 1
    away_team = teams[away_idx]
except:
    matching = [t for t in teams if away_input.lower() in t.lower()]
    if matching:
        away_team = matching[0]
    else:
        away_team = teams[1]

features_dict, home_stats, away_stats = build_features_for_prediction(home_team, away_team)

if features_dict is None:
    exit()

X_pred = pd.DataFrame([features_dict])[feature_columns]

home_win_prob = clf.predict_proba(X_pred)[0, 1]
away_win_prob = 1 - home_win_prob

predicted_spread = reg_spread.predict(X_pred)[0]

if home_win_prob > 0.5:
    predicted_winner = home_team
    confidence = home_win_prob
else:
    predicted_winner = away_team
    confidence = away_win_prob

feature_imp = feature_importance.set_index('feature')['importance_classification']

top_features = feature_importance.head(15)

comparisons = [
    ('Recent Win %', f'win_pct{feature_suffix}', f'win_pct{feature_suffix}'),
    ('Points per Game', f'teamScore{feature_suffix}', f'teamScore{feature_suffix}'),
    ('FG%', f'fieldGoalsPercentage{feature_suffix}', f'fieldGoalsPercentage{feature_suffix}'),
    ('Assists', f'assists{feature_suffix}', f'assists{feature_suffix}'),
    ('Rebounds', f'reboundsTotal{feature_suffix}', f'reboundsTotal{feature_suffix}'),
]

if has_player_features:
    comparisons.append(('Star Scorer PPG', 'star_points_max', 'star_points_max'))
    comparisons.append(('Top 5 PPG Avg', 'top5_points_avg', 'top5_points_avg'))

home_importance = feature_imp.get('home', 0)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
teams_list = [home_team, away_team]
probs = [home_win_prob * 100, away_win_prob * 100]
colors = ['#1f77b4', '#ff7f0e']
ax1.barh(teams_list, probs, color=colors)
ax1.set_xlabel('Win Probability (%)', fontsize=12)
ax1.set_title('Win Probability', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 100)
ax1.axvline(50, color='gray', linestyle='--', alpha=0.5)
for i, (team, prob) in enumerate(zip(teams_list, probs)):
    ax1.text(prob + 2, i, f'{prob:.1f}%', va='center', fontweight='bold')

ax2 = axes[0, 1]
top_feats = feature_importance.head(10)
ax2.barh(range(len(top_feats)), top_feats['importance_classification'])
ax2.set_yticks(range(len(top_feats)))
ax2.set_yticklabels([f.replace('_', ' ').title() for f in top_feats['feature']], fontsize=9)
ax2.set_xlabel('Feature Importance')
ax2.set_title('Top 10 Model Features', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

ax3 = axes[1, 0]
stats_to_plot = [f'teamScore{feature_suffix}', f'assists{feature_suffix}',
                 f'fieldGoalsPercentage{feature_suffix}', f'reboundsTotal{feature_suffix}']
stat_labels = ['PPG', 'Assists', 'FG%', 'Rebounds']
home_vals = [home_stats.get(s, 0) for s in stats_to_plot]
away_vals = [away_stats.get(s, 0) for s in stats_to_plot]

x = np.arange(len(stat_labels))
width = 0.35
ax3.bar(x - width/2, home_vals, width, label=home_team, color='#1f77b4')
ax3.bar(x + width/2, away_vals, width, label=away_team, color='#ff7f0e')
ax3.set_xlabel('Statistics')
ax3.set_title('Key Stats Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(stat_labels)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
PREDICTION SUMMARY
{'='*50}

Matchup: {away_team} @ {home_team}

Winner: {predicted_winner}
Confidence: {confidence*100:.1f}%

Win Probability:
  {home_team}: {home_win_prob*100:.1f}%
  {away_team}: {away_win_prob*100:.1f}%

Point Spread:
  {abs(predicted_spread):.1f} points
  Favors: {home_team if predicted_spread > 0 else away_team}

Key Factors:
  • Home court advantage ({home_importance*100:.1f}% importance)
  • {home_team} win rate: {home_stats.get(f'win_pct{feature_suffix}', 0):.1%}
  • {away_team} win rate: {away_stats.get(f'win_pct{feature_suffix}', 0):.1%}

Model: XGBoost (2020-2025 trained)
Features: {len(feature_columns)} total
Data: Modern NBA only
"""
ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle(f'NBA GamePlan: {away_team} @ {home_team}',
             fontsize=16, fontweight='bold')
plt.tight_layout()

output_file = f'results/prediction_{home_team.replace(" ", "_")}_vs_{away_team.replace(" ", "_")}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')

plt.show()
