
"""
train_model.py
===============
Generates a realistic synthetic diabetes dataset (modelled on the
statistical properties of the well-known Pima Indians Diabetes Dataset),
trains three classical machine-learning classifiers, evaluates them, and
persists the best-performing model plus its scaler and metrics to disk.

Run directly:
    python train_model.py
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    LOG_FORMAT,
    LOG_LEVEL,
    METRICS_PATH,
    MODEL_PATH,
    N_SYNTHETIC_SAMPLES,
    RANDOM_STATE,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def generate_synthetic_dataset(n_samples: int = N_SYNTHETIC_SAMPLES) -> pd.DataFrame:
    """Generate a realistic synthetic diabetes dataset.

    The feature distributions are modelled loosely on the publicly known
    summary statistics of the Pima Indians Diabetes Dataset, extended with
    a few extra lifestyle features (PhysicalActivity, Smoking,
    FamilyHistory) requested by the application. The target label
    (``Outcome``) is generated from a logistic combination of the risk
    factors plus noise, so the resulting dataset has genuine, learnable
    structure rather than being pure random noise.

    Args:
        n_samples: Number of synthetic patient records to generate.

    Returns:
        A pandas DataFrame with feature columns and an ``Outcome`` column
        (1 = diabetic / high-risk, 0 = non-diabetic).
    """
    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(18, 85, n_samples)
    pregnancies = np.clip(rng.poisson(1.5, n_samples), 0, 15)
    glucose = np.clip(rng.normal(120, 32, n_samples), 44, 300)
    blood_pressure = np.clip(rng.normal(72, 12, n_samples), 24, 180)
    skin_thickness = np.clip(rng.normal(23, 10, n_samples), 0, 99)
    insulin = np.clip(rng.normal(85, 90, n_samples), 0, 850)
    bmi = np.clip(rng.normal(31, 7, n_samples), 15, 67)
    pedigree = np.clip(rng.gamma(2, 0.25, n_samples), 0.05, 2.5)
    activity = np.clip(rng.integers(0, 11, n_samples), 0, 10)
    smoking = rng.binomial(1, 0.2, n_samples)
    family_history = rng.binomial(1, 0.3, n_samples)

    # Logistic combination of risk factors -> probability of diabetes.
    # The intercept is calibrated so the overall prevalence sits around a
    # realistic ~30-35%, similar to the Pima Indians Diabetes Dataset.
    linear_score = (
        0.035 * glucose
        + 0.08 * bmi
        + 0.02 * age
        + 0.35 * pedigree
        + 0.015 * blood_pressure
        + 0.4 * family_history
        + 0.25 * smoking
        - 0.12 * activity
        + 0.01 * insulin / 10
    )
    intercept = -(np.mean(linear_score) + 0.8)  # centers prevalence ~30-35%
    z = intercept + linear_score
    prob = 1 / (1 + np.exp(-z))
    outcome = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "Pregnancies": pregnancies,
            "Glucose": glucose.round(1),
            "BloodPressure": blood_pressure.round(1),
            "SkinThickness": skin_thickness.round(1),
            "Insulin": insulin.round(1),
            "BMI": bmi.round(1),
            "DiabetesPedigreeFunction": pedigree.round(3),
            "Age": age,
            "PhysicalActivity": activity,
            "Smoking": smoking,
            "FamilyHistory": family_history,
            TARGET_COLUMN: outcome,
        }
    )
    return df


def train_and_select_best_model() -> dict:
    """Train Random Forest, Logistic Regression, and Decision Tree models,
    evaluate them on a held-out test set, and persist the best model.

    Returns:
        A metrics dictionary summarising the performance of all models,
        which is also written to ``model_metrics.json``.
    """
    logger.info("Generating synthetic dataset ...")
    df = generate_synthetic_dataset()
    df.to_csv(DATASET_PATH, index=False)
    logger.info("Dataset saved to %s (%d rows).", DATASET_PATH, len(df))

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE
        ),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    }

    results = {}
    fitted_models = {}

    for name, clf in candidates.items():
        # Tree-based models don't strictly need scaling but we keep the
        # pipeline consistent for all models to simplify serving.
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        proba = clf.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, proba)
        cm = confusion_matrix(y_test, preds).tolist()
        fpr, tpr, _ = roc_curve(y_test, proba)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm,
            "roc_curve": {
                "fpr": fpr.round(4).tolist(),
                "tpr": tpr.round(4).tolist(),
            },
        }
        fitted_models[name] = clf
        logger.info("%s -> accuracy=%.4f, f1=%.4f, roc_auc=%.4f", name, acc, f1, auc)

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    best_model = fitted_models[best_name]
    logger.info("Best model selected: %s", best_name)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metrics_payload = {
        "best_model": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "results": results,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info("Model saved to %s, scaler to %s, metrics to %s.", MODEL_PATH, SCALER_PATH, METRICS_PATH)
    return metrics_payload


if __name__ == "__main__":
    train_and_select_best_model()
