"""
NBA GamePlan - Modern Model Training (2020-2025 with Player Features)
Team 29: Train on recent NBA data with exponentially weighted features and player stats
"""

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
warnings.filterwarnings('ignore')

print("="*80)
print("NBA GAMEPLAN - MODERN MODEL TRAINING (2020-2025)")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING MODERN NBA DATA WITH PLAYER FEATURES...")
print("-"*80)

try:
    df = pd.read_csv('data/processed_modern_nba_with_players.csv')
    print("✓ Loaded data WITH player features")
    has_player_features = True
except FileNotFoundError:
    print("⚠ Player features not found, using team-only data")
    df = pd.read_csv('data/processed_modern_nba.csv')
    has_player_features = False

df['gameDate'] = pd.to_datetime(df['gameDate'])
print(f"✓ Loaded {len(df):,} games")
print(f"Date range: {df['gameDate'].min()} to {df['gameDate'].max()}")
print(f"Seasons: {df['season'].min()}-{df['season'].max()}")
print(f"Teams: {df['teamName'].nunique()}")

# ============================================================================
# 2. PREPARE FEATURES
# ============================================================================
print("\n[2] PREPARING FEATURES...")
print("-"*80)

# Feature columns using EWMA (exponentially weighted moving averages)
feature_columns = [
    # Team context
    'home', 'days_rest', 'is_back_to_back', 'month', 'streak',

    # Team EWMA features (recent games weighted more heavily)
    'win_pct_ewm',
    'teamScore_ewm', 'assists_ewm', 'reboundsTotal_ewm',
    'steals_ewm', 'blocks_ewm', 'turnovers_ewm',
    'fieldGoalsPercentage_ewm', 'threePointersPercentage_ewm',
    'pointsInThePaint_ewm', 'pointsFastBreak_ewm', 'benchPoints_ewm',
    'efg_pct_ewm', 'ts_pct_ewm', 'tov_rate_ewm', 'pace_ewm',

    # Opponent EWMA features
    'opp_score_ewm', 'opp_assists_ewm', 'opp_rebounds_ewm',
    'opp_fg_pct_ewm', 'opp_win_pct_ewm',

    # Head-to-head
    'h2h_win_pct'
]

# Add player features if available
if has_player_features:
    player_feature_cols = [
        'top5_points_avg', 'top5_assists_avg', 'top5_rebounds_avg',
        'top5_plusminus_avg', 'star_points_max'
    ]
    feature_columns.extend(player_feature_cols)
    print(f"✓ Including {len(player_feature_cols)} player features")

# Handle categorical features
df_model = df.copy()
df_model['season_part'] = df_model['season_part'].fillna('mid')
season_dummies = pd.get_dummies(df_model['season_part'], prefix='season', drop_first=True)
df_model = pd.concat([df_model, season_dummies], axis=1)
feature_columns.extend(season_dummies.columns.tolist())

# Fill NaN values
for col in feature_columns:
    if col in df_model.columns:
        df_model[col] = df_model[col].fillna(df_model[col].median())

# Targets
df_model['win'] = df_model['win'].astype(int)
df_model['point_diff'] = df_model['teamScore'] - df_model['opponentScore']

print(f"✓ Total features: {len(feature_columns)}")

# ============================================================================
# 3. TRAIN/TEST SPLIT
# ============================================================================
print("\n[3] CREATING TRAIN/TEST SPLIT...")
print("-"*80)

# Train: 2020-2023, Test: 2024-2025
train_mask = df_model['season'] < 2024
test_mask = df_model['season'] >= 2024

X_train = df_model[train_mask][feature_columns]
y_train_class = df_model[train_mask]['win']
y_train_spread = df_model[train_mask]['point_diff']

X_test = df_model[test_mask][feature_columns]
y_test_class = df_model[test_mask]['win']
y_test_spread = df_model[test_mask]['point_diff']

print(f"✓ Training set: {len(X_train):,} games (2020-2023)")
print(f"✓ Test set: {len(X_test):,} games (2024-2025)")
print(f"✓ Train win rate: {y_train_class.mean():.3f}")
print(f"✓ Test win rate: {y_test_class.mean():.3f}")

# ============================================================================
# 4. TRAIN CLASSIFICATION MODEL
# ============================================================================
print("\n[4] TRAINING BINARY CLASSIFICATION MODEL...")
print("-"*80)

clf = xgb.XGBClassifier(
    n_estimators=300,  # More trees for better performance
    max_depth=7,
    learning_rate=0.05,  # Lower learning rate for stability
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

print(f"\n✓ CLASSIFICATION RESULTS:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC AUC:   {auc:.4f}")

cm = confusion_matrix(y_test_class, y_pred_class)
print(f"\n  Confusion Matrix:")
print(f"  {cm}")

# ============================================================================
# 5. TRAIN REGRESSION MODEL
# ============================================================================
print("\n[5] TRAINING POINT SPREAD REGRESSION MODEL...")
print("-"*80)

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

print(f"\n✓ POINT SPREAD RESULTS:")
print(f"  MAE:  {mae_spread:.2f} points")
print(f"  RMSE: {rmse_spread:.2f} points")
print(f"  R²:   {r2_spread:.4f}")

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================
print("\n[6] ANALYZING FEATURE IMPORTANCE...")
print("-"*80)

importance_clf = clf.feature_importances_
importance_reg = reg_spread.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance_classification': importance_clf,
    'importance_regression': importance_reg
}).sort_values('importance_classification', ascending=False)

print("\nTop 15 Features (Classification):")
print(feature_importance_df.head(15)[['feature', 'importance_classification']].to_string(index=False))

# ============================================================================
# 7. SAVE MODELS AND RESULTS
# ============================================================================
print("\n[7] SAVING MODELS AND RESULTS...")
print("-"*80)

import os
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

clf.save_model('models/xgb_modern_classification.json')
reg_spread.save_model('models/xgb_modern_regression.json')
print("✓ Saved models")

feature_importance_df.to_csv('results/feature_importance_modern.csv', index=False)
print("✓ Saved feature importance")

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
print("✓ Saved predictions")

# Save metadata about active teams
active_teams = sorted(df['teamName'].unique())
with open('results/active_teams.txt', 'w') as f:
    f.write('\n'.join(active_teams))
print(f"✓ Saved {len(active_teams)} active teams")

# ============================================================================
# 8. CREATE VISUALIZATIONS
# ============================================================================
print("\n[8] CREATING VISUALIZATIONS...")
print("-"*80)

# Feature importance plot
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
print("✓ Saved feature importance plot")

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Modern NBA Model')
plt.tight_layout()
plt.savefig('results/confusion_matrix_modern.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved confusion matrix")

# ROC curve
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
print("✓ Saved ROC curve")

print("\n" + "="*80)
print("MODERN MODEL TRAINING COMPLETE")
print("="*80)
print(f"\n✓ Summary:")
print(f"  - Classification Accuracy: {accuracy:.1%}")
print(f"  - F1 Score: {f1:.3f}")
print(f"  - ROC AUC: {auc:.3f}")
print(f"  - Point Spread MAE: {mae_spread:.2f} points")
print(f"  - Point Spread RMSE: {rmse_spread:.2f} points")
print(f"\n✓ Improvements:")
print(f"  - Focused on 2020-2025 (modern NBA)")
print(f"  - Active teams only (no defunct teams)")
print(f"  - Exponentially weighted features (recent games matter more)")
if has_player_features:
    print(f"  - Player-level features integrated (top 5 players)")
print(f"\n✓ Models saved to: models/")
print(f"✓ Results saved to: results/")
