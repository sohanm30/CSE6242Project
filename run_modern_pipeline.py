"""
NBA GamePlan - Complete Modern Pipeline
Team 29: Run all preprocessing, player features, and training in one command
"""

import subprocess
import sys
import time

print("="*80)
print("NBA GAMEPLAN - COMPLETE MODERN PIPELINE")
print("="*80)
print("\nThis will run:")
print("  1. Modern NBA preprocessing (2020-2025, active teams, EWMA)")
print("  2. Player features integration")
print("  3. Model training with improved features")
print("\nEstimated time: 15-20 minutes")
print("="*80)

input("\nPress ENTER to start...")

scripts = [
    ("Modern NBA Preprocessing", "02c_modern_preprocessing.py"),
    ("Player Features Integration", "02d_add_player_features_modern.py"),
    ("Modern Model Training", "03b_train_modern_model.py")
]

for i, (name, script) in enumerate(scripts, 1):
    print(f"\n{'='*80}")
    print(f"STEP {i}/{len(scripts)}: {name}")
    print(f"{'='*80}")

    start_time = time.time()

    result = subprocess.run([sys.executable, script], capture_output=False)

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n⚠ ERROR: {name} failed!")
        print(f"Please check the error messages above.")
        sys.exit(1)

    print(f"\n✓ {name} completed in {elapsed:.1f} seconds")

print("\n" + "="*80)
print("✅ COMPLETE MODERN PIPELINE FINISHED SUCCESSFULLY!")
print("="*80)
print("\n✓ Your models are now ready with:")
print("  - 2020-2025 data only (modern NBA)")
print("  - Active teams only (no Bobcats, etc.)")
print("  - Exponentially weighted features (recent games matter more)")
print("  - Player-level statistics (top 5 players per team)")
print("\n✓ Next steps:")
print("  1. Check results/feature_importance_modern.csv")
print("  2. Run predictions with: python 04_predict_game_modern.py")
print("  3. Review test_predictions_modern.csv for accuracy on 2024-25 season")
