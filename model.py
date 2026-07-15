"""
model.py
========
Wraps the trained scikit-learn model (and its scaler) behind a clean,
typed interface, and provides SHAP-based explainability on top of raw
predictions.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import shap

from config import (
    FEATURE_COLUMNS,
    LOG_FORMAT,
    LOG_LEVEL,
    METRICS_PATH,
    MODEL_PATH,
    RISK_BANDS,
    SCALER_PATH,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured result returned by :meth:`DiabetesRiskModel.predict`."""

    probability: float
    risk_band: str
    confidence: float


@dataclass
class ExplanationResult:
    """Structured SHAP-based explanation for a single prediction."""

    shap_values: dict[str, float]
    positive_factors: list[tuple[str, float]]
    negative_factors: list[tuple[str, float]]
    base_value: float


def probability_to_band(probability: float) -> str:
    """Map a raw probability in [0, 1] to a human-readable risk band using
    the thresholds defined in ``config.RISK_BANDS``."""
    for band, (low, high) in RISK_BANDS.items():
        if low <= probability < high:
            return band
    return "High Risk"


class DiabetesRiskModel:
    """High-level interface around the trained diabetes risk classifier."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        scaler_path: str = SCALER_PATH,
        metrics_path: str = METRICS_PATH,
    ) -> None:
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metrics_path = metrics_path
        self.model = None
        self.scaler = None
        self.metrics: dict = {}
        self._explainer = None
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not (os.path.exists(self.model_path) and os.path.exists(self.scaler_path)):
            logger.warning(
                "Model artifacts not found. Run `python train_model.py` first."
            )
            return
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)
        logger.info("Model, scaler, and metrics loaded successfully.")

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    # ------------------------------------------------------------------
    def _to_frame(self, patient: dict[str, float]) -> pd.DataFrame:
        row = {col: patient.get(col, 0) for col in FEATURE_COLUMNS}
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def predict(self, patient: dict[str, float]) -> PredictionResult:
        """Predict diabetes risk probability for one patient.

        Args:
            patient: Dictionary of raw feature values keyed by the names
                in ``config.FEATURE_COLUMNS``.

        Returns:
            A :class:`PredictionResult` with probability, risk band, and a
            confidence score (distance of the probability from the
            decision boundary, rescaled to [0, 1]).
        """
        if not self.is_ready:
            raise RuntimeError("Model is not loaded. Run train_model.py first.")

        X = self._to_frame(patient)
        X_scaled = self.scaler.transform(X)
        proba = float(self.model.predict_proba(X_scaled)[0, 1])
        band = probability_to_band(proba)
        confidence = float(abs(proba - 0.5) * 2)  # 0 = uncertain, 1 = very confident

        return PredictionResult(probability=proba, risk_band=band, confidence=confidence)

    # ------------------------------------------------------------------
    def explain(self, patient: dict[str, float]) -> ExplanationResult:
        """Generate a SHAP explanation for a single patient's prediction.

        Uses a TreeExplainer when the underlying model supports it (Random
        Forest / Decision Tree) and falls back to a model-agnostic
        KernelExplainer otherwise (e.g. Logistic Regression).
        """
        if not self.is_ready:
            raise RuntimeError("Model is not loaded. Run train_model.py first.")

        X = self._to_frame(patient)
        X_scaled = self.scaler.transform(X)

        if self._explainer is None:
            model_type = type(self.model).__name__
            if model_type in ("RandomForestClassifier", "DecisionTreeClassifier"):
                self._explainer = shap.TreeExplainer(self.model)
            else:
                background = np.zeros((1, len(FEATURE_COLUMNS)))
                self._explainer = shap.LinearExplainer(
                    self.model, background, feature_names=FEATURE_COLUMNS
                )

        raw_values = self._explainer.shap_values(X_scaled)

        # Normalise output shape across SHAP versions / explainer types.
        if isinstance(raw_values, list):
            # [class0_values, class1_values]
            values = np.array(raw_values[1])[0]
            base_value = self._explainer.expected_value
            base_value = base_value[1] if isinstance(base_value, (list, np.ndarray)) else base_value
        else:
            arr = np.array(raw_values)
            if arr.ndim == 3:  # (n_samples, n_features, n_classes)
                values = arr[0, :, 1]
            else:
                values = arr[0]
            base_value = self._explainer.expected_value
            base_value = base_value[1] if isinstance(base_value, (list, np.ndarray)) and len(np.shape(base_value)) else base_value

        shap_dict = {col: round(float(v), 4) for col, v in zip(FEATURE_COLUMNS, values)}
        sorted_items = sorted(shap_dict.items(), key=lambda kv: kv[1], reverse=True)
        positive = [(k, v) for k, v in sorted_items if v > 0][:5]
        negative = [(k, v) for k, v in sorted_items if v < 0][-5:]

        return ExplanationResult(
            shap_values=shap_dict,
            positive_factors=positive,
            negative_factors=negative,
            base_value=float(base_value) if base_value is not None else 0.0,
        )

    # ------------------------------------------------------------------
    def feature_importance(self) -> dict[str, float]:
        """Return global feature importance from the trained model, if the
        model type supports it (tree-based models)."""
        if not self.is_ready:
            return {}
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            return {}
        return {
            col: round(float(imp), 4)
            for col, imp in sorted(
                zip(FEATURE_COLUMNS, importances), key=lambda kv: kv[1], reverse=True
            )
        }
