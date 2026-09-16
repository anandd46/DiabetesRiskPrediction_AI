"""
config.py
=========
Central configuration file for the Explainable AI-Based Diabetes Risk
Prediction and Early Screening System.

Keeping all constants in one place makes the project easier to maintain,
tune, and reason about -- a good practice recruiters look for.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# GENERAL PROJECT INFO
# --------------------------------------------------------------------------
APP_TITLE: str = "Explainable AI Diabetes Risk & Early Screening System"
APP_SUBTITLE: str = (
    "Constraint Satisfaction \u2022 Fuzzy Logic \u2022 Explainable Machine Learning"
)
MEDICAL_DISCLAIMER: str = (
    "\u26A0\uFE0F This tool performs **risk screening only**. It is **NOT a medical "
    "diagnosis** and must never replace advice from a qualified healthcare "
    "professional. If you are experiencing symptoms, please consult a doctor."
)

# --------------------------------------------------------------------------
# FILE PATHS (all files live in the SAME root folder - no subfolders)
# --------------------------------------------------------------------------
DATASET_PATH: str = "dataset.csv"
MODEL_PATH: str = "diabetes_model.pkl"
SCALER_PATH: str = "scaler.pkl"
METRICS_PATH: str = "model_metrics.json"
DATABASE_PATH: str = "temp_database.db"

# --------------------------------------------------------------------------
# ML TRAINING SETTINGS
# --------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
N_SYNTHETIC_SAMPLES: int = 2000

FEATURE_COLUMNS: list[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "PhysicalActivity",
    "Smoking",
    "FamilyHistory",
]

TARGET_COLUMN: str = "Outcome"

# --------------------------------------------------------------------------
# CSP (CONSTRAINT SATISFACTION) REALISTIC LIMITS
# --------------------------------------------------------------------------
CSP_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "Age": (1, 120),
    "Pregnancies": (0, 20),
    "Glucose": (0, 400),
    "BloodPressure": (0, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (5, 80),
    "DiabetesPedigreeFunction": (0.0, 3.0),
    "PhysicalActivity": (0, 10),
    "Weight": (2, 400),
    "Height": (30, 250),
}

# --------------------------------------------------------------------------
# RISK THRESHOLDS (used for final risk-band classification)
# --------------------------------------------------------------------------
RISK_BANDS: dict[str, tuple[float, float]] = {
    "Low Risk": (0.0, 0.33),
    "Moderate Risk": (0.33, 0.66),
    "High Risk": (0.66, 1.01),
}

RISK_COLORS: dict[str, str] = {
    "Low Risk": "#2ecc71",
    "Moderate Risk": "#f39c12",
    "High Risk": "#e74c3c",
}

# --------------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_LEVEL: str = "INFO"
