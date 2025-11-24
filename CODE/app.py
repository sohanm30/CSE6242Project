from __future__ import annotations
import os

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from difflib import get_close_matches
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent

clf: Optional[xgb.XGBClassifier] = None
reg_spread: Optional[xgb.XGBRegressor] = None
explainer_clf: Optional[shap.TreeExplainer] = None
explainer_reg: Optional[shap.TreeExplainer] = None
team_data: Optional[pd.DataFrame] = None
feature_importance: Optional[pd.DataFrame] = None
teams: list[str] = []
has_player_features: bool = False
use_ewma: bool = False
feature_suffix: str = ""
feature_columns: list[str] = []

TEAM_ALIASES: Dict[str, str] = {
    # East
    "atlanta hawks": "Hawks",
    "boston celtics": "Celtics",
    "brooklyn nets": "Nets",
    "charlotte hornets": "Hornets",
    "chicago bulls": "Bulls",
    "cleveland cavaliers": "Cavaliers",
    "detroit pistons": "Pistons",
    "indiana pacers": "Pacers",
    "miami heat": "Heat",
    "milwaukee bucks": "Bucks",
    "new york knicks": "Knicks",
    "orlando magic": "Magic",
    "philadelphia 76ers": "76ers",
    "toronto raptors": "Raptors",
    "washington wizards": "Wizards",

    # West
    "dallas mavericks": "Mavericks",
    "denver nuggets": "Nuggets",
    "golden state warriors": "Warriors",
    "houston rockets": "Rockets",
    "los angeles clippers": "Clippers",
    "los angeles lakers": "Lakers",
    "memphis grizzlies": "Grizzlies",
    "minnesota timberwolves": "Timberwolves",
    "new orleans pelicans": "Pelicans",
    "oklahoma city thunder": "Thunder",
    "phoenix suns": "Suns",
    "portland trail blazers": "Trail Blazers",
    "sacramento kings": "Kings",
    "san antonio spurs": "Spurs",
    "utah jazz": "Jazz",

    # Short forms
    "lakers": "Lakers",
    "la lakers": "Lakers",
    "clippers": "Clippers",
    "la clippers": "Clippers",
}


def canonicalize_team(name: str) -> Optional[str]:
    """Map an input name to a known team using aliases, case-insensitive match, substring, or close match."""
    n = (name or "").strip()
    if not n:
        return None

    if n in teams:
        return n

    low = n.lower()
    if low in TEAM_ALIASES and TEAM_ALIASES[low] in teams:
        return TEAM_ALIASES[low]

    for t in teams:
        if t.lower() == low:
            return t

    for t in teams:
        tl = t.lower()
        if low in tl or tl in low:
            return t

    candidates = get_close_matches(n, teams, n=1, cutoff=0.6)
    if candidates:
        return candidates[0]

    return None


def _load_models_and_data() -> None:
    global clf, reg_spread, explainer_clf, explainer_reg, team_data, feature_importance, teams
    global has_player_features, use_ewma, feature_suffix, feature_columns

    clf = xgb.XGBClassifier()
    reg_spread = xgb.XGBRegressor()

    cls_paths = [
        BASE_DIR / "models/xgb_modern_classification.json",
        BASE_DIR / "models/xgb_classification.json",
    ]
    reg_paths = [
        BASE_DIR / "models/xgb_modern_regression.json",
        BASE_DIR / "models/xgb_regression_spread.json",
    ]

    cls_loaded = False
    for p in cls_paths:
        if p.exists():
            clf.load_model(str(p))
            cls_loaded = True
            break
    if not cls_loaded:
        raise FileNotFoundError(
            "No classification model found in models/. Expected one of: "
            + ", ".join(str(p) for p in cls_paths)
        )

    reg_loaded = False
    for p in reg_paths:
        if p.exists():
            reg_spread.load_model(str(p))
            reg_loaded = True
            break
    if not reg_loaded:
        reg_spread = None 


    data_paths = [
        BASE_DIR / "data/processed_modern_nba_with_players.csv",
        BASE_DIR / "data/processed_modern_nba.csv",
        BASE_DIR / "data/processed_team_data.csv",
    ]

    for p in data_paths:
        if p.exists():
            team_data_local = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError(
            "No processed data found in data/. Expected one of: "
            + ", ".join(str(p) for p in data_paths)
        )

    if "gameDate" in team_data_local.columns:
        team_data_local["gameDate"] = pd.to_datetime(team_data_local["gameDate"], errors="coerce")

    nonlocal_has_players = all(
        col in team_data_local.columns
        for col in [
            "top5_points_avg",
            "top5_assists_avg",
            "top5_rebounds_avg",
            "top5_plusminus_avg",
            "star_points_max",
        ]
    )

    nonlocal_use_ewma = "teamScore_ewm" in team_data_local.columns

    suffix = "_ewm" if nonlocal_use_ewma else "_roll10"
    base_features = [
        "home",
        "days_rest",
        "is_back_to_back",
        "month",
        "streak",
        f"win_pct{suffix}",
        f"teamScore{suffix}",
        f"assists{suffix}",
        f"reboundsTotal{suffix}",
        f"steals{suffix}",
        f"blocks{suffix}",
        f"turnovers{suffix}",
        f"fieldGoalsPercentage{suffix}",
        f"threePointersPercentage{suffix}",
        f"pointsInThePaint{suffix}",
        f"pointsFastBreak{suffix}",
        f"benchPoints{suffix}",
        f"efg_pct{suffix}",
        f"ts_pct{suffix}",
        f"tov_rate{suffix}",
        f"pace{suffix}",
        f"opp_score{suffix}",
        f"opp_assists{suffix}",
        f"opp_rebounds{suffix}",
        f"opp_fg_pct{suffix}",
        f"opp_win_pct{suffix}",
        "h2h_win_pct",
    ]

    if nonlocal_has_players:
        base_features.extend(
            [
                "top5_points_avg",
                "top5_assists_avg",
                "top5_rebounds_avg",
                "top5_plusminus_avg",
                "star_points_max",
            ]
        )

    base_features.extend(["season_late", "season_mid"])  # order must match training

    team_data = team_data_local
    teams = sorted(team_data["teamName"].dropna().unique().tolist()) if "teamName" in team_data.columns else []
    has_player_features = nonlocal_has_players
    use_ewma = nonlocal_use_ewma
    feature_suffix = suffix
    feature_columns[:] = base_features

    if clf is not None and team_data_local is not None:
        try:
            missing_cols = [c for c in feature_columns if c not in team_data_local.columns]
            bg_source = team_data_local.copy()
            for c in missing_cols:
                if c in ['season_late', 'season_mid']:
                    bg_source[c] = 0
                else:
                    bg_source[c] = 0
            
            available_features = feature_columns # Now we have all of them
            background_data = bg_source[available_features].sample(50, random_state=42).fillna(0)
            
            def clf_predict(X):
                if isinstance(X, pd.DataFrame):
                return clf.predict_proba(X)[:, 1]
                
            explainer_clf = shap.Explainer(clf_predict, background_data)
        except Exception as e:
            print(f"Warning: Could not initialize classification explainer: {e}")
            explainer_clf = None
    else:
        explainer_clf = None

    if reg_spread is not None and team_data_local is not None:
        try:
            missing_cols = [c for c in feature_columns if c not in team_data_local.columns]
            bg_source = team_data_local.copy()
            for c in missing_cols:
                if c in ['season_late', 'season_mid']:
                    bg_source[c] = 0
                else:
                    bg_source[c] = 0
            
            available_features = feature_columns
            background_data = bg_source[available_features].sample(50, random_state=42).fillna(0)
            
            def reg_predict(X):
                return reg_spread.predict(X)
                
            explainer_reg = shap.Explainer(reg_predict, background_data)
        except Exception as e:
            print(f"Warning: Could not initialize regression explainer: {e}")
            explainer_reg = None
    else:
        explainer_reg = None

    fi_paths = [
        BASE_DIR / "results/feature_importance_modern.csv",
        BASE_DIR / "results/feature_importance.csv",
    ]
    for p in fi_paths:
        if p.exists():
            try:
                feature_importance = pd.read_csv(p)
            except Exception:
                feature_importance = None
            break


def get_team_recent_stats(team_name: str, cutoff_date: Optional[pd.Timestamp] = None) -> Optional[pd.Series]:
    assert team_data is not None
    if cutoff_date is None and "gameDate" in team_data.columns:
        cutoff_date = team_data["gameDate"].max()

    df = team_data
    mask = (df["teamName"] == team_name)
    if cutoff_date is not None and "gameDate" in df.columns:
        mask &= df["gameDate"] <= cutoff_date

    team_games = df.loc[mask].sort_values("gameDate", ascending=False)
    if len(team_games) == 0:
        return None
    return team_games.iloc[0]


def build_features_for_prediction(home_team: str, away_team: str) -> Optional[Tuple[Dict[str, Any], pd.Series, pd.Series]]:
    assert team_data is not None
    home_stats = get_team_recent_stats(home_team)
    away_stats = get_team_recent_stats(away_team)
    if home_stats is None or away_stats is None:
        return None

    features: Dict[str, Any] = {}

    features["home"] = 1
    features["days_rest"] = home_stats.get("days_rest", 2)
    features["is_back_to_back"] = int(home_stats.get("is_back_to_back", 0))
    features["month"] = datetime.now().month
    features["streak"] = home_stats.get("streak", 0)

    for stat in [
        "win_pct",
        "teamScore",
        "assists",
        "reboundsTotal",
        "steals",
        "blocks",
        "turnovers",
        "fieldGoalsPercentage",
        "threePointersPercentage",
        "pointsInThePaint",
        "pointsFastBreak",
        "benchPoints",
        "efg_pct",
        "ts_pct",
        "tov_rate",
        "pace",
    ]:
        col_name = f"{stat}{feature_suffix}"
        features[col_name] = home_stats.get(col_name, 0)

    for stat in ["score", "assists", "rebounds", "fg_pct", "win_pct"]:
        if stat == "score":
            away_col = f"teamScore{feature_suffix}"
        elif stat == "rebounds":
            away_col = f"reboundsTotal{feature_suffix}"
        elif stat == "fg_pct":
            away_col = f"fieldGoalsPercentage{feature_suffix}"
        elif stat == "win_pct":
            away_col = f"win_pct{feature_suffix}"
        else:
            away_col = f"{stat}{feature_suffix}"
        features[f"opp_{stat}{feature_suffix}"] = away_stats.get(away_col, 0)

    features["h2h_win_pct"] = home_stats.get("h2h_win_pct", 0.5)
    features["season_late"] = 0
    features["season_mid"] = 1

    if has_player_features:
        for pf in [
            "top5_points_avg",
            "top5_assists_avg",
            "top5_rebounds_avg",
            "top5_plusminus_avg",
            "star_points_max",
        ]:
            features[pf] = home_stats.get(pf, 0)

    return features, home_stats, away_stats


def get_season_stats(team_name: str) -> Optional[Dict[str, Any]]:
    """Get current season stats for a team (2025-26 season, no preseason)."""
    assert team_data is not None
    
    if "gameDate" not in team_data.columns:
        return None
    
    mask = (team_data["teamName"] == team_name) & (team_data["gameDate"] >= "2025-10-21")
    
    if "seasonType" in team_data.columns:
        mask = mask & (team_data["seasonType"] != "Pre Season")
    
    current_season = team_data[mask]
    
    if len(current_season) == 0:
        return None
    
    wins = int(current_season["win"].sum()) if "win" in current_season.columns else 0
    games = len(current_season)
    losses = games - wins
    
    stats = {
        "games_played": games,
        "wins": wins,
        "losses": losses,
        "ppg": float(current_season["teamScore"].mean()) if "teamScore" in current_season.columns else 0.0,
        "fg_pct": float(current_season["fieldGoalsPercentage"].mean()) if "fieldGoalsPercentage" in current_season.columns else 0.0,
        "three_pct": float(current_season["threePointersPercentage"].mean()) if "threePointersPercentage" in current_season.columns else 0.0,
        "rebounds": float(current_season["reboundsTotal"].mean()) if "reboundsTotal" in current_season.columns else 0.0,
        "assists": float(current_season["assists"].mean()) if "assists" in current_season.columns else 0.0,
    }
    
    return stats


_load_models_and_data()

@app.get("/health")
def health() -> Any:
    return jsonify({
        "status": "ok",
        "models": {
            "classification": True,
            "regression": reg_spread is not None,
        },
        "data": {
            "rows": int(team_data.shape[0]) if team_data is not None else 0,
            "teams": len(teams),
            "has_player_features": has_player_features,
            "feature_suffix": feature_suffix,
        },
    })


@app.get("/teams")
def list_teams() -> Any:
    return jsonify({"teams": teams})


@app.post("/predict")
def predict() -> Any:
    payload = request.get_json(silent=True) or {}
    inp_home = (payload.get("home_team") or "").strip()
    inp_away = (payload.get("away_team") or "").strip()

    if not inp_home or not inp_away:
        return jsonify({"error": "home_team and away_team are required"}), 400

    home_team = canonicalize_team(inp_home)
    away_team = canonicalize_team(inp_away)

    if home_team is None:
        suggestion = get_close_matches(inp_home, teams, n=3)
        return jsonify({"error": f"Unknown home_team: {inp_home}", "suggestions": suggestion}), 400
    if away_team is None:
        suggestion = get_close_matches(inp_away, teams, n=3)
        return jsonify({"error": f"Unknown away_team: {inp_away}", "suggestions": suggestion}), 400

    if home_team == away_team:
        return jsonify({"error": "home_team and away_team must be different"}), 400

    built = build_features_for_prediction(home_team, away_team)
    if built is None:
        return jsonify({"error": "Unable to build features for the given teams"}), 422

    features_dict, home_stats, away_stats = built

    X_pred = pd.DataFrame([features_dict])[feature_columns]

    home_win_prob = float(clf.predict_proba(X_pred)[0, 1])
    away_win_prob = float(1.0 - home_win_prob)

    spread_value: Optional[float] = None
    if reg_spread is not None:
        spread_value = float(reg_spread.predict(X_pred)[0])

    if home_win_prob >= 0.5:
        predicted_winner = home_team
        confidence = home_win_prob
        prob_favors_home = True
    else:
        predicted_winner = away_team
        confidence = away_win_prob
        prob_favors_home = False

    consistency: Optional[bool] = None
    if spread_value is not None:
        spread_favors_home = spread_value > 0
        consistency = spread_favors_home == prob_favors_home

    home_season_stats = get_season_stats(home_team)
    away_season_stats = get_season_stats(away_team)

    explanations = []
    if explainer_clf is not None:
        try:
            if hasattr(explainer_clf.masker, 'feature_names'):
                 masker_features = explainer_clf.masker.feature_names
                 X_shap = X_pred[masker_features]
            else:
                 X_shap = X_pred
                 
            shap_explanation = explainer_clf(X_shap)
            
            vals = shap_explanation.values[0]
            
            feature_names = X_pred.columns.tolist()
            feature_values = X_pred.iloc[0].tolist()
            
            for name, val, impact in zip(feature_names, feature_values, vals):
                explanations.append({
                    "feature": name,
                    "value": val,
                    "impact": float(impact),
                    "type": "positive" if impact > 0 else "negative"
                })
            
            explanations.sort(key=lambda x: abs(x["impact"]), reverse=True)
            
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            import traceback; traceback.print_exc()
            explanations = []
    else:
        print("DEBUG: explainer_clf is None")

    return jsonify(
        {
            "matchup": {
                "home_team": home_team,
                "away_team": away_team,
            },
            "probabilities": {
                "home": home_win_prob,
                "away": away_win_prob,
            },
            "prediction": {
                "winner": predicted_winner,
                "confidence": confidence,
                "predicted_spread": spread_value,
                "consistent": consistency,
            },
            "explanations": explanations[:10],
            "season_stats": {
                "home": home_season_stats,
                "away": away_season_stats,
            },
            "meta": {
                "feature_suffix": feature_suffix,
                "has_player_features": has_player_features,
            },
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False)
