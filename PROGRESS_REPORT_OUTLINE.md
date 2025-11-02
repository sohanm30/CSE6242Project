# NBA GamePlan: An Interactive Matchup Predictor & Strategy Analyzer
## Progress Report Outline

**Team 29**: Harold Huang, Shyamanth Kudum, John Fox, Adam Ezzaoudi, Rashmith Repala, Sohan Malladi

---

## 1. INTRODUCTION (FINAL - From Proposal)

**[Copy from proposal, keep concise - ~0.5 pages]**

We aim to build NBA GamePlan, an interactive web application that predicts NBA game outcomes while explaining *why* through statistical insights. Existing prediction systems like ESPN's Basketball Power Index provide numerical forecasts with little interpretability, creating a black box that prevents users from reasoning with the insights. Our novelty lies in combining state-of-the-art predictive modeling with Explainable AI (XAI) and visual analytics.

This project benefits NBA fans, sports bettors, fantasy basketball managers, and analysts seeking data-driven insights that are interpretable and actionable. We measure success through accuracy metrics (F1, precision, recall, AUC) on held-out games and evaluation of the clarity and trustworthiness of our explanations.

---

## 2. PROBLEM DEFINITION

**[~0.25 pages - can expand from proposal]**

**Problem Statement**: NBA game prediction systems lack transparency and interpretability. Users receive win probabilities without understanding the underlying factors driving predictions.

**Objectives**:
1. Predict game outcomes (win/loss) with competitive accuracy (>60%)
2. Predict point spreads with reasonable precision (MAE <15 points)
3. Explain predictions through feature importance and statistical comparisons
4. Visualize predictions and explanations in an interactive dashboard

**Scope**: Focus on modern NBA (2020-2025), active teams only (30 teams), using publicly available statistics from Kaggle.

---

## 3. LITERATURE SURVEY (FINAL - From Proposal)

**[Copy from proposal - ~1 page, already in final form]**

The core of our prediction engine relies on established machine learning techniques, especially gradient boosting. Friedman (2001) introduced the gradient boosting machine, which provides the basis for our approach. Building on this, Chen and Guestrin (2016) created XGBoost, a highly scalable implementation and our main choice for modeling. Czarnowski and Jędrzejowicz (2020) applied ensemble learning to sports prediction, showing that boosting algorithms usually deliver better accuracy. While these papers offer a strong framework for prediction, their main limitation is a lack of interpretability.

To enable explanations along with predictive results, we incorporate techniques from Explainable AI. Štrumbelj and Kononenko (2014) used Shapley values to explain individual model predictions. Lundberg and Lee (2017) later unified this concept with SHAP (SHapley Additive exPlanations), which we initially explored. The main drawback is high computational cost. Our evaluation follows broader surveys from Molnar et al. (2020) and Chen et al. (2018), which support the need for transparency.

The features of our model are guided by existing research in basketball analytics. Miljković et al. (2017) and Sharma et al. (2023) provide strong basketball analytics context but don't have actionable models. Luo et al. (2018) analyzes player and team performance metrics, serving as a basis for our feature selection. Cervone et al. (2014) and Fernandez et al. (2020) explore sophisticated spatial models, though their immediate use is limited for our game-level scope.

Finally, the user-facing component draws inspiration from visual analytics. Keim et al. (2008) describe visual analytics as the combination of automated analysis and interactive visualization, shaping our dashboard philosophy. Goldsberry (2012) created CourtVision, an innovative work in visual shot analytics that influences our interactive dashboard design. We aim to integrate predictive modeling with these visualization approaches.

---

## 4. PROPOSED METHOD (ALMOST FINISHED - 70% of grade)

### 4.1 Overview

Our system consists of three main components:
1. **Data Pipeline**: Collection, cleaning, and feature engineering
2. **Prediction Engine**: Dual XGBoost models (classification + regression)
3. **Visualization Dashboard**: Interactive interface with explainability *(in progress)*

**[INSERT FIGURE 1: System Architecture Diagram]**
*Suggested: Flowchart showing Data → Feature Engineering → Models → Visualization*

---

### 4.2 Data Sources and Processing

**Data Sources**:
- TeamStatistics.csv: 144,016 game records (2000-2025)
- PlayerStatistics.csv: 1,631,540 player-game records
- Games.csv: Game metadata and schedules
- Current NBA rosters and team mappings

**Modern NBA Focus**: We filter to 2020-2025 seasons and active teams only (30 franchises), yielding 16,230 processed games. This eliminates historical noise from defunct teams (e.g., Charlotte Bobcats) and captures contemporary basketball style (3-point heavy, pace-and-space).

**[INSERT TABLE 1: Dataset Statistics]**

| Dataset | Total Records | Filtered Records | Date Range |
|---------|--------------|------------------|------------|
| Team Games | 144,016 | 16,230 | 2020-2025 |
| Player Games | 1,631,540 | ~580,000 | 2020-2025 |
| Active Teams | 30 | 30 | 2024-25 |
| Train/Test Split | 12,000 / 4,000 | - | 2020-23 / 2024-25 |

---

### 4.3 Feature Engineering

We engineer 35 features across five categories, employing Exponentially Weighted Moving Averages (EWMA) to capture momentum and recent form.

#### **Innovation #1: Exponentially Weighted Moving Averages (EWMA)**

Traditional sports analytics use simple rolling averages (e.g., last 10 games). We implement EWMA with decay parameter span=10 for team stats and span=5 for player stats. This weights recent games exponentially higher, capturing hot/cold streaks and momentum shifts.

**Mathematical formulation**:
- EWMA(t) = α × value(t) + (1-α) × EWMA(t-1)
- α = 2/(span+1)

**Advantages**:
- Adapts faster to performance changes (coaching adjustments, injuries)
- Reduces noise from outlier games further in the past
- Reflects current team state more accurately than uniform averaging

#### **Innovation #2: Player-Team Hybrid Feature Integration**

Most team-level models ignore individual player contributions. We bridge this gap by aggregating top-5 players (by recent minutes played) into team-level features:

1. Identify top 5 players per team per game (by EWMA of minutes)
2. Calculate player EWMA stats: PPG, APG, RPG, +/-
3. Aggregate to team level:
   - Average of top 5 (e.g., `top5_points_avg`)
   - Maximum of star player (e.g., `star_points_max`)

This captures roster quality, star player impact, and injuries without requiring game-by-game lineup data.

**[INSERT TABLE 2: Feature Categories and Descriptions]**

| Category | Count | Examples | EWMA Applied? |
|----------|-------|----------|---------------|
| Team Context | 5 | home, days_rest, is_back_to_back | No |
| Team Performance (EWMA) | 16 | win_pct_ewm, teamScore_ewm, efg_pct_ewm | Yes (span=10) |
| Opponent Strength (EWMA) | 5 | opp_win_pct_ewm, opp_score_ewm | Yes (span=10) |
| Head-to-Head | 1 | h2h_win_pct (last 5 games) | No |
| Player Features | 5 | top5_plusminus_avg, star_points_max | Yes (span=5) |
| Season Timing | 3 | season_early, season_mid, season_late | No |
| **Total** | **35** | - | - |

**Advanced Metrics Included**:
- Effective FG% (eFG%): Accounts for 3-pointers being worth more
- True Shooting % (TS%): Includes free throws in efficiency calculation
- Turnover Rate (TOV%): Per-possession turnover metric
- Pace: Possessions per 48 minutes

---

### 4.4 Prediction Models

#### **Innovation #3: Dual-Model Framework with Consistency Checking**

We train two complementary XGBoost models:

**Model 1: Binary Classification (Win/Loss)**
- Predicts probability of home team winning
- Outputs: P(home wins), P(away wins) = 1 - P(home)
- Use case: Fantasy sports, fan predictions

**Model 2: Point Spread Regression**
- Predicts point differential (home score - away score)
- Positive value = home team favored
- Use case: Sports betting, margin analysis

**Configuration** (both models):
```
XGBoost Parameters:
- n_estimators: 300 trees (increased from baseline 200)
- max_depth: 7 (allows complex interactions)
- learning_rate: 0.05 (lower for stability)
- subsample: 0.8 (row sampling)
- colsample_bytree: 0.8 (feature sampling)
- random_state: 42 (reproducibility)
```

**Consistency Verification**: After prediction, we verify that:
- If P(home) > 0.5, then predicted_spread > 0
- If P(home) < 0.5, then predicted_spread < 0

This cross-validation builds user trust and catches model errors.

**[INSERT FIGURE 2: Model Architecture]**
*Suggested: Diagram showing inputs (35 features) → Two parallel XGBoost models → Outputs (probability + spread)*

---

### 4.5 Explainability and Interpretability

We provide three levels of explanation:

1. **Global Importance**: Feature importance from XGBoost (gain-based)
   - Shows which features matter most across all games
   - Home court advantage identified as #1 factor (7.22%)

2. **Per-Prediction Feature Values**:
   - Display top 10 features with their actual values for the specific matchup
   - Direct comparison between home and away teams

3. **Statistical Comparison**:
   - Side-by-side bar charts of key stats (win%, scoring, efficiency)
   - Highlights team strengths/weaknesses

**Note on SHAP**: We initially implemented SHAP explanations (TreeExplainer) but found computational costs prohibitive for real-time predictions (>30 seconds per game). We pivoted to permutation importance and feature value displays, which provide sufficient interpretability with <1 second latency.

---

### 4.6 Visualization Dashboard *(IN PROGRESS)*

**Current Status**: Prediction system generates static 4-panel visualizations (PNG) for individual matchups.

**Visualization Components** (per prediction):
1. Win probability bar chart (home vs away)
2. Top 10 feature importance (bar chart)
3. Key stats comparison (grouped bar chart)
4. Formatted text summary with confidence level

**[INSERT FIGURE 3: Sample Prediction Output]**
*Suggested: Screenshot of actual prediction (e.g., Thunder vs Kings) showing 4 panels*

**Planned Dashboard Features** *(Section 5 - Future Work)*:
- Interactive web interface (Plotly Dash / Streamlit)
- Team dropdown selectors
- Date picker for scheduling
- Historical accuracy tracking by team
- "What-if" analysis: Adjust feature values to see impact on prediction

---

### 4.7 Summary of Innovations

1. **Exponentially Weighted Moving Averages**: Recent games weighted higher to capture momentum
2. **Player-Team Hybrid Integration**: Top-5 player stats aggregated to team level
3. **Dual-Model Framework**: Classification + regression with consistency checking

**Comparison to Existing Systems**:

| Feature | ESPN BPI | FiveThirtyEight | Our System |
|---------|----------|-----------------|------------|
| Explainability | ✗ (Black box) | ✗ (Elo formula only) | ✓ (Feature importance + comparisons) |
| Player Integration | Partial | ✗ (Team Elo) | ✓ (Top-5 aggregation) |
| Recency Weighting | Unknown | ✓ (Elo decay) | ✓ (EWMA, span=10) |
| Point Spread | ✓ | ✓ | ✓ |
| Open Source | ✗ | Partial | ✓ (Educational) |

---

## 5. EXPERIMENTS AND EVALUATION

### 5.1 Experimental Setup

**Train/Test Split**:
- Training: 2020-2023 seasons (~12,000 games)
- Testing: 2024-2025 seasons (~4,000 games)
- Temporal split ensures no data leakage

**Evaluation Metrics**:
- Classification: Accuracy, Precision, Recall, F1, ROC AUC
- Regression: MAE (Mean Absolute Error), RMSE
- Explainability: Feature importance rankings

---

### 5.2 Current Results

**[INSERT TABLE 3: Model Performance Metrics]**

| Model | Metric | Value | Interpretation |
|-------|--------|-------|----------------|
| **Classification** | Accuracy | **61.31%** | Correctly predicted 61% of games |
| | Precision | 60.29% | Of predicted wins, 60% were correct |
| | Recall | 66.41% | Caught 66% of actual wins |
| | F1 Score | 0.632 | Balanced performance |
| | ROC AUC | 0.652 | Good discrimination ability |
| **Regression** | MAE | **11.81 points** | Average error ~12 points |
| | RMSE | 15.03 points | Standard deviation of errors |

**[INSERT FIGURE 4: Confusion Matrix]**
*Use: results/confusion_matrix_modern.png*

**[INSERT FIGURE 5: ROC Curve]**
*Use: results/roc_curve_modern.png*

**[INSERT FIGURE 6: Feature Importance]**
*Use: results/feature_importance_modern.png*

---

### 5.3 Feature Importance Analysis

**[INSERT TABLE 4: Top 10 Features by Importance]**

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | home | 7.22% | Context (home court advantage) |
| 2 | opp_win_pct_ewm | 4.40% | Opponent strength (EWMA) |
| 3 | win_pct_ewm | 4.31% | Team recent form (EWMA) |
| 4 | h2h_win_pct | 3.61% | Head-to-head history |
| 5 | top5_plusminus_avg | 3.42% | **Player feature** |
| 6 | star_points_max | 3.17% | **Player feature** |
| 7 | ts_pct_ewm | 3.15% | True shooting % (EWMA) |
| 8 | tov_rate_ewm | 3.13% | Turnover rate (EWMA) |
| 9 | top5_rebounds_avg | 3.09% | **Player feature** |
| 10 | is_back_to_back | 3.09% | Schedule factor |

**Key Findings**:
1. **Home court advantage** is the strongest predictor (7.22%), quantifying the well-known NBA phenomenon
2. **Player features** appear in top 10 (ranks 5, 6, 9), validating Innovation #2
3. **EWMA features dominate**: 6 of top 10 use exponential weighting, validating Innovation #1
4. **Opponent strength matters**: Understanding who you're playing is nearly as important as your own form

---

### 5.4 Performance Benchmarking

**Comparison to Baselines**:

| System | Accuracy | Notes |
|--------|----------|-------|
| Random Guess | 50% | Baseline |
| Home Team Always Wins | ~58% | Naive baseline (home court) |
| **Our Modern Model** | **61.31%** | With explainability |
| ESPN BPI | ~67%* | Proprietary, no public metrics |
| FiveThirtyEight | ~65-68%* | Elo-based, crowd-sourced |
| Vegas Lines | ~70%* | Market-driven, insider info |

*Industry estimates from published reports; not directly comparable due to different test sets

**Analysis**: Our model achieves competitive accuracy while maintaining full explainability. We sacrifice 5-8% accuracy compared to proprietary systems but gain transparency, educational value, and player-awareness.

**Point Spread Analysis**: MAE of 11.81 points is reasonable given NBA's high variance. For context:
- Average NBA final margin: ~10-12 points
- Standard deviation of margins: ~13 points
- Our RMSE (15.03) is slightly above 1σ, indicating most predictions are within expected variance

---

### 5.5 Upcoming Experiments (PLANNED - NOT YET DONE)

#### **Experiment 1: Ablation Study**
**Objective**: Quantify contribution of each innovation

**Method**:
1. Baseline: Train model WITHOUT EWMA (use simple 10-game averages)
2. +EWMA: Add exponential weighting
3. +EWMA +Players: Add player features
4. Full Model: EWMA + Players + Dual Models

**Metrics**: Compare accuracy, F1, MAE across variants

**Expected Outcome**: Show that each innovation improves performance

**Status**: *Data prepared, experiment design finalized, execution planned for Week 3*

---

#### **Experiment 2: Feature Ablation**
**Objective**: Identify minimal feature set for good performance

**Method**:
1. Start with all 35 features (baseline)
2. Remove bottom 50% by importance → retrain
3. Remove bottom 75% → retrain
4. Test top 5, top 10, top 15 features only

**Metrics**: Accuracy vs. number of features (efficiency tradeoff)

**Expected Outcome**: Identify if top 10-15 features capture most signal

**Status**: *Planned for Week 3, after Experiment 1*

---

#### **Experiment 3: Temporal Validation**
**Objective**: Test model stability across seasons

**Method**:
1. Train on 2020-2022, test on 2023
2. Train on 2021-2023, test on 2024
3. Train on 2020-2023, test on 2025
4. Compare performance across test years

**Metrics**: Accuracy by season, identify if model degrades over time

**Expected Outcome**: Model remains stable (±3% accuracy) across modern era

**Status**: *Data available, planned for Week 4*

---

#### **Experiment 4: Dashboard Usability Evaluation**
**Objective**: Assess if explanations are understandable and useful

**Method**:
1. Develop interactive dashboard (Plotly Dash)
2. Recruit 10-15 classmates/friends as users
3. Tasks:
   - "Predict outcome of Lakers vs Celtics"
   - "Explain why Lakers are favored"
   - "Which team metric is most important?"
4. Post-task survey:
   - Understandability (1-5 Likert scale)
   - Trust in predictions (1-5)
   - Usefulness for decision-making (1-5)
   - Open feedback

**Metrics**: Survey scores, task completion time, qualitative feedback

**Expected Outcome**: >4.0/5.0 average ratings, positive qualitative feedback

**Status**: *Dashboard development in progress (Week 3-4), user study planned for Week 5*

---

#### **Experiment 5: Scalability Testing**
**Objective**: Ensure system can handle real-time predictions

**Method**:
1. Measure prediction latency for single game
2. Measure throughput for full slate (10-15 games/night)
3. Test on different hardware (laptop vs server)

**Metrics**:
- Latency (ms per prediction)
- Throughput (predictions/second)

**Expected Outcome**: <100ms per prediction, >10 predictions/second

**Status**: *Planned for Week 4, after dashboard completion*

---

### 5.6 Evaluation Timeline

**[INSERT TABLE 5: Experiment Timeline]**

| Week | Experiments | Deliverable |
|------|-------------|-------------|
| 1-2 (Done) | Model training, baseline metrics | Current results (Table 3) |
| 3 | Ablation studies (Exp 1-2) | Feature contribution analysis |
| 4 | Temporal validation (Exp 3), Scalability (Exp 5) | Stability report |
| 5 | Dashboard development | Interactive prototype |
| 6 | User study (Exp 4) | Usability evaluation report |
| 7 | Final analysis, report writing | Final report + poster |

---

## 6. CONCLUSIONS AND FUTURE WORK

### 6.1 Current Status

We have successfully implemented a functional NBA game prediction system with three key innovations:
1. EWMA-based feature engineering for momentum capture
2. Player-team hybrid features for roster-aware predictions
3. Dual-model framework (classification + regression) with consistency checking

Our modern model achieves 61.31% accuracy and 11.81 MAE on 2024-2025 test data, demonstrating competitive performance with full explainability. Feature importance analysis validates our innovations: home court is the strongest predictor (7.22%), followed by opponent strength and team form (EWMA), with player features appearing in the top 10.

**Completed Components**:
- ✓ Data collection and cleaning pipeline
- ✓ Feature engineering (35 features, EWMA-based)
- ✓ XGBoost model training (classification + regression)
- ✓ Evaluation metrics and benchmarking
- ✓ Static prediction visualizations (4-panel charts)

**In Progress**:
- ⚙ Interactive dashboard development (Plotly Dash / Streamlit)
- ⚙ Ablation studies to quantify innovation contributions
- ⚙ User study design for usability evaluation

---

### 6.2 Challenges Encountered

1. **SHAP Computational Cost**: Initial full SHAP implementation took >30s per prediction, making real-time use infeasible. We pivoted to permutation importance and feature value displays.

2. **Player Data Sparsity**: Some player stats have gaps due to injuries/trades. We handle this with 10-day lookback windows and median imputation.

3. **Model Selection Trade-off**: Deeper models (max_depth > 8) showed overfitting. We settled on depth=7 after cross-validation.

---

### 6.3 Remaining Work

**High Priority (Weeks 3-4)**:
1. Complete interactive dashboard with team selectors and date picker
2. Run ablation studies (Experiments 1-2)
3. Temporal validation across seasons (Experiment 3)

**Medium Priority (Weeks 5-6)**:
1. User study for usability evaluation (Experiment 4)
2. Optimize prediction latency for real-time use
3. Add "what-if" analysis feature (adjust features, see impact)

**Low Priority / Future Extensions**:
1. Integrate injury reports (requires external API)
2. Add lineup-specific predictions (requires play-by-play data)
3. Deploy as public web app (requires hosting infrastructure)

---

### 6.4 Expected Contributions

By the final report, we expect to deliver:

1. **Technical Contribution**: A fully documented, reproducible NBA prediction system combining modern ML with XAI techniques, demonstrating three concrete innovations

2. **Empirical Insights**: Quantitative evidence that EWMA and player integration improve prediction accuracy through ablation studies

3. **Practical Tool**: An interactive dashboard that non-experts can use to understand NBA matchups, bridging the gap between advanced analytics and public accessibility

4. **Educational Value**: Open-source code and documentation for students/researchers to learn sports analytics and explainable ML

---

## 7. PROJECT TIMELINE

### 7.1 Original Plan (From Proposal)

| Phase | Task | Members | Start | End |
|-------|------|---------|-------|-----|
| 1 | Data collection & cleaning | All | Oct 9 | Oct 14 |
| 2 | Feature engineering & modeling (XGBoost) | Sunny, John | Oct 15 | Oct 25 |
| 3 | XAI (SHAP) integration | Sohan, Harold | Oct 26 | Nov 5 |
| 4 | Dashboard development | Rashmith, Adam | Nov 6 | Nov 13 |
| 5 | Testing & evaluation | All | Nov 14 | Nov 17 |
| 6 | Final report & poster | All | Nov 18 | Nov 21 |

---

### 7.2 Revised Plan (Current)

| Phase | Task | Members | Start | End | Status |
|-------|------|---------|-------|-----|--------|
| 1 | Data collection & cleaning | All | Oct 9 | Oct 14 | ✓ Done |
| 2A | Modern feature engineering (EWMA) | Sunny, John | Oct 15 | Oct 22 | ✓ Done |
| 2B | Player feature integration | Sunny, John | Oct 23 | Oct 28 | ✓ Done |
| 3A | XGBoost model training | Sunny, John | Oct 29 | Nov 1 | ✓ Done |
| 3B | SHAP exploration & pivot | Sohan, Harold | Oct 26 | Nov 3 | ✓ Done (pivoted) |
| 4A | Static prediction visualizations | Sunny | Nov 2 | Nov 4 | ✓ Done |
| 4B | Interactive dashboard development | Rashmith, Adam | Nov 5 | Nov 11 | ⚙ In Progress |
| 5A | Ablation studies (Exp 1-2) | John, Sohan | Nov 8 | Nov 12 | ⏳ Planned |
| 5B | User study (Exp 4) | Rashmith, Harold | Nov 12 | Nov 15 | ⏳ Planned |
| 6 | Final analysis & report | All | Nov 16 | Nov 21 | ⏳ Planned |

**[INSERT FIGURE 7: Revised Gantt Chart]**
*Suggested: Visual Gantt chart with completed tasks (green), in-progress (yellow), planned (blue)*

---

### 7.3 Key Revisions from Original Plan

**Changes**:
1. **SHAP Integration (Phase 3)**: Scaled back from full SHAP to permutation importance due to computational constraints. This reduced Phase 3 workload but maintained explainability goals.

2. **Feature Engineering Expansion (Phase 2)**: Split into two sub-phases:
   - 2A: EWMA implementation (Innovation #1)
   - 2B: Player integration (Innovation #2)
   Total time increased by 3 days to accommodate additional complexity.

3. **Dashboard Development (Phase 4)**: Split into static (4A) and interactive (4B) components. Static visualizations completed early to enable progress report submission.

4. **Evaluation Expansion (Phase 5)**: Added ablation studies (5A) not in original plan to quantify innovation contributions. User study (5B) remains on track.

**Impact on Timeline**: Project remains on schedule. SHAP simplification saved 2 days, which we reallocated to ablation studies. Final report deadline (Nov 21) is achievable.

---

### 7.4 Effort Distribution Statement

**Progress Report Period (Oct 9 - Nov 4)**:
- **Sunny Kudum, John Fox**: Primary work on data pipeline, feature engineering (EWMA, player integration), model training, and evaluation. Approximately 35% each.
- **Sohan Malladi, Harold Huang**: SHAP exploration, literature review refinement, permutation importance implementation. Approximately 15% each.
- **Rashmith Repala, Adam Ezzaoudi**: Dashboard design, static visualization generation, preparing for interactive development. Approximately 10% each (ramping up Week 3-4).

**Assessment**: Effort distribution has been slightly uneven due to phasing. Sunny and John front-loaded computational work (Phases 1-3), while Rashmith and Adam are ramping up for dashboard development (Phase 4). By final report, we expect all members to have contributed approximately equally, with Rashmith and Adam delivering the interactive dashboard, and Sohan and Harold leading the user study evaluation.

---

## REFERENCES

[Copy all references from proposal - alphabetical order]

Baughman, A., et al. (2020). Deep Artificial Intelligence for Fantasy Football...

[... all other references ...]

---

## NOTES FOR FORMATTING

**Page Budget** (4 pages max, excluding references):
- Section 1-3 (Intro, Problem, Literature): ~1.5 pages
- Section 4 (Methods): ~1.5 pages (THIS IS THE MEAT - 70% of grade)
- Section 5 (Experiments): ~0.75 pages
- Section 6 (Conclusions): ~0.25 pages
- Section 7 (Timeline): ~0.5 pages (can be compact with visual Gantt)

**Figures/Tables to Include**:
- MUST INCLUDE:
  - Table 1: Dataset statistics
  - Table 2: Feature categories
  - Table 3: Model performance metrics ⭐
  - Table 4: Top 10 features ⭐
  - Table 5: Experiment timeline
  - Figure 3: Sample prediction output ⭐
  - Figure 4: Confusion matrix ⭐
  - Figure 5: ROC curve
  - Figure 6: Feature importance bar chart ⭐
  - Figure 7: Revised Gantt chart ⭐

- OPTIONAL (if space allows):
  - Figure 1: System architecture
  - Figure 2: Model architecture

**Formatting**:
- 11pt font (Times New Roman or similar)
- 1-inch margins all sides
- Single or 1.15 line spacing (check with TA/professor)
- Tables: Use booktabs style (clean, minimal lines)
- Figures: 300 DPI, clear labels, readable when printed

**Pro Tips**:
- Use two-column layout for Tables 3 and 4 to save space
- Make Gantt chart visual (color-coded) rather than text-heavy
- Keep figure captions concise (1 sentence)
- Reference figures/tables in text: "As shown in Table 3..."
