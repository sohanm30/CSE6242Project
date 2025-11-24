import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
import xgboost as xgb
from sklearn.inspection import permutation_importance
import warnings
import os
warnings.filterwarnings('ignore')

try:
    df = pd.read_csv('data/processed_modern_nba_with_players.csv')
    has_player_features = True
except FileNotFoundError:
    df = pd.read_csv('data/processed_modern_nba.csv')
    has_player_features = False

df['gameDate'] = pd.to_datetime(df['gameDate'])

feature_columns = [
    'home', 'days_rest', 'is_back_to_back', 'month', 'streak',
    'win_pct_ewm',
    'teamScore_ewm', 'assists_ewm', 'reboundsTotal_ewm',
    'steals_ewm', 'blocks_ewm', 'turnovers_ewm',
    'fieldGoalsPercentage_ewm', 'threePointersPercentage_ewm',
    'pointsInThePaint_ewm', 'pointsFastBreak_ewm', 'benchPoints_ewm',
    'efg_pct_ewm', 'ts_pct_ewm', 'tov_rate_ewm', 'pace_ewm',
    'opp_score_ewm', 'opp_assists_ewm', 'opp_rebounds_ewm',
    'opp_fg_pct_ewm', 'opp_win_pct_ewm',
    'h2h_win_pct'
]

if has_player_features:
    player_feature_cols = [
        'top5_points_avg', 'top5_assists_avg', 'top5_rebounds_avg',
        'top5_plusminus_avg', 'star_points_max'
    ]
    feature_columns.extend(player_feature_cols)

df_model = df.copy()
df_model['season_part'] = df_model['season_part'].fillna('mid')
season_dummies = pd.get_dummies(df_model['season_part'], prefix='season', drop_first=True)
df_model = pd.concat([df_model, season_dummies], axis=1)
feature_columns.extend(season_dummies.columns.tolist())

for col in feature_columns:
    if col in df_model.columns:
        df_model[col] = df_model[col].fillna(df_model[col].median())

df_model['win'] = df_model['win'].astype(int)
df_model['point_diff'] = df_model['teamScore'] - df_model['opponentScore']

train_mask = df_model['season'] < 2024
test_mask = df_model['season'] >= 2024

X_train = df_model[train_mask][feature_columns]
y_train_class = df_model[train_mask]['win']
y_train_spread = df_model[train_mask]['point_diff']

X_test = df_model[test_mask][feature_columns]
y_test_class = df_model[test_mask]['win']
y_test_spread = df_model[test_mask]['point_diff']

clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    base_score=0.5
)

clf.fit(X_train, y_train_class, verbose=False)

y_pred_class = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)
auc = roc_auc_score(y_test_class, y_pred_proba)

cm = confusion_matrix(y_test_class, y_pred_class)

reg_spread = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    base_score=0.0
)

reg_spread.fit(X_train, y_train_spread, verbose=False)

y_pred_spread = reg_spread.predict(X_test)

mae_spread = mean_absolute_error(y_test_spread, y_pred_spread)
rmse_spread = np.sqrt(mean_squared_error(y_test_spread, y_pred_spread))
r2_spread = r2_score(y_test_spread, y_pred_spread)

importance_clf = clf.feature_importances_
importance_reg = reg_spread.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance_classification': importance_clf,
    'importance_regression': importance_reg
}).sort_values('importance_classification', ascending=False)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

clf.save_model('models/xgb_modern_classification.json')
reg_spread.save_model('models/xgb_modern_regression.json')

feature_importance_df.to_csv('results/feature_importance_modern.csv', index=False)

predictions_df = pd.DataFrame({
    'gameId': df_model[test_mask]['gameId'].values,
    'gameDate': df_model[test_mask]['gameDate'].values,
    'teamName': df_model[test_mask]['teamName'].values,
    'opponentTeamName': df_model[test_mask]['opponentTeamName'].values,
    'actual_win': y_test_class.values,
    'predicted_win': y_pred_class,
    'win_probability': y_pred_proba,
    'actual_spread': y_test_spread.values,
    'predicted_spread': y_pred_spread
})
predictions_df.to_csv('results/test_predictions_modern.csv', index=False)

active_teams = sorted(df['teamName'].unique())
with open('results/active_teams.txt', 'w') as f:
    f.write('\n'.join(active_teams))

plt.figure(figsize=(10, 8))
top_features = feature_importance_df.head(15)
plt.barh(range(15), top_features['importance_classification'].values)
plt.yticks(range(15), top_features['feature'].values)
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features (Modern NBA Model)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('results/feature_importance_modern.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Modern NBA Model')
plt.tight_layout()
plt.savefig('results/confusion_matrix_modern.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test_class, y_pred_proba)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Modern NBA Model')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curve_modern.png', dpi=300, bbox_inches='tight')
plt.close()
