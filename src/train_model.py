import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/training_data.csv"
MODELS_DIR  = "models"

# ── Model settings ─────────────────────────────────────────────────────────────
N_ESTIMATORS = 200    # number of trees in the forest
MAX_DEPTH    = None   # let trees grow fully
RANDOM_STATE = 42
TEST_SIZE    = 0.2    # 20% held out for evaluation


def load_data(path=DATA_PATH):
    """Load CSV and split into features and targets."""
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

    # Features — what the model sees as input
    X = df[["delay_ms", "packet_loss", "noise_std"]].values

    # Targets — what the model must predict
    y_kp = df["best_kp"].values
    y_ki = df["best_ki"].values
    y_kd = df["best_kd"].values

    return X, y_kp, y_ki, y_kd, df


def train_model(X_train, y_train, label=""):
    """Train one Random Forest regressor for a single target (Kp, Ki, or Kd)."""
    model = RandomForestRegressor(
        n_estimators = N_ESTIMATORS,
        max_depth    = MAX_DEPTH,
        random_state = RANDOM_STATE,
        n_jobs       = -1,    # use all CPU cores
    )
    model.fit(X_train, y_train)
    print(f"  Trained model for {label}  ({N_ESTIMATORS} trees)")
    return model


def evaluate_model(model, X_test, y_test, label=""):
    """Print MAE and R² score for one model."""
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"  {label:8s} →  MAE: {mae:.4f}   R²: {r2:.4f}")
    return y_pred


def save_models(model_kp, model_ki, model_kd, scaler=None):
    """Save trained models to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model_kp, f"{MODELS_DIR}/kp_model.pkl")
    joblib.dump(model_ki, f"{MODELS_DIR}/ki_model.pkl")
    joblib.dump(model_kd, f"{MODELS_DIR}/kd_model.pkl")
    print(f"\n  Models saved to {MODELS_DIR}/")


def plot_predictions(y_tests, y_preds, labels):
    """
    Scatter plot: predicted vs actual gains for Kp, Ki, Kd.
    Perfect prediction = all points on the diagonal line.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#1D9E75", "#7F77DD", "#D85A30"]

    for ax, y_test, y_pred, label, color in zip(axes, y_tests, y_preds, labels, colors):
        ax.scatter(y_test, y_pred, alpha=0.5, s=25, color=color, edgecolors="none")

        # Perfect prediction line
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2, label="Perfect prediction")

        mae = mean_absolute_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)

        ax.set_xlabel(f"Actual {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label}   MAE={mae:.3f}   R²={r2:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Random Forest — Predicted vs Actual PID Gains", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/plots/model_predictions.png", dpi=150)
    plt.show()
    print("Plot saved to results/plots/model_predictions.png")


def plot_feature_importance(model_kp, model_ki, model_kd):
    """Show which wireless feature matters most for each gain prediction."""
    features = ["delay_ms", "packet_loss", "noise_std"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors    = ["#1D9E75", "#7F77DD", "#D85A30"]
    models    = [model_kp, model_ki, model_kd]
    labels    = ["Kp", "Ki", "Kd"]

    for ax, model, label, color in zip(axes, models, labels, colors):
        importance = model.feature_importances_
        bars = ax.barh(features, importance, color=color, edgecolor="none")
        ax.set_title(f"Feature importance for {label}")
        ax.set_xlabel("Importance")
        ax.grid(True, axis="x", alpha=0.3)
        for bar, val in zip(bars, importance):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)

    plt.suptitle("Which wireless condition affects each gain the most?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/plots/feature_importance.png", dpi=150)
    plt.show()
    print("Plot saved to results/plots/feature_importance.png")


def run_training():
    print("=" * 55)
    print("  AI Model Training — Random Forest Regressor")
    print("=" * 55)

    # 1. Load data
    X, y_kp, y_ki, y_kd, df = load_data()

    # 2. Train/test split
    X_train, X_test, \
    kp_train, kp_test, \
    ki_train, ki_test, \
    kd_train, kd_test = train_test_split(
        X, y_kp, y_ki, y_kd,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE
    )
    print(f"\nTrain samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}\n")

    # 3. Train one model per gain
    print("Training models...")
    model_kp = train_model(X_train, kp_train, label="Kp")
    model_ki = train_model(X_train, ki_train, label="Ki")
    model_kd = train_model(X_train, kd_train, label="Kd")

    # 4. Evaluate
    print("\nEvaluation on test set:")
    pred_kp = evaluate_model(model_kp, X_test, kp_test, label="Kp")
    pred_ki = evaluate_model(model_ki, X_test, ki_test, label="Ki")
    pred_kd = evaluate_model(model_kd, X_test, kd_test, label="Kd")

    # 5. Save models
    save_models(model_kp, model_ki, model_kd)

    # 6. Plots
    print("\nGenerating plots...")
    plot_predictions(
        [kp_test, ki_test, kd_test],
        [pred_kp, pred_ki, pred_kd],
        ["Kp", "Ki", "Kd"]
    )
    plot_feature_importance(model_kp, model_ki, model_kd)

    print("\nTraining complete.")
    return model_kp, model_ki, model_kd


if __name__ == "__main__":
    run_training()