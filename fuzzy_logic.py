

"""
fuzzy_logic.py
===============
Fuzzy Logic module implementing a Mamdani-style Fuzzy Inference System (FIS)
for diabetes risk estimation, built with ``scikit-fuzzy``.

Unlike crisp rule-based systems, fuzzy logic allows variables such as
"Glucose" to be simultaneously a little LOW and a little HIGH, weighted by
membership degrees. This models the inherent uncertainty of medical
thresholds far better than hard cut-offs, and is a classic example of
"Reasoning under Uncertainty" from classical AI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from config import LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class FuzzyResult:
    """Container for the results of a fuzzy inference run."""

    risk_score: float  # 0-100 crisp output
    risk_label: str
    fired_rules: list[str]
    memberships: dict[str, dict[str, float]]


class FuzzyDiabetesRiskSystem:
    """Builds and evaluates a fuzzy inference system for diabetes risk.

    Inputs:
        glucose (0-300 mg/dL)
        bmi (10-60 kg/m^2)
        age (1-100 years)
        activity (0-10 activity score, 10 = very active)

    Output:
        risk (0-100): a crisp risk score mapped to 5 linguistic bands:
        Very Low, Low, Moderate, High, Very High.
    """

    def __init__(self) -> None:
        self._build_system()

    # ------------------------------------------------------------------
    def _build_system(self) -> None:
        # Antecedents (inputs)
        self.glucose = ctrl.Antecedent(np.arange(0, 301, 1), "glucose")
        self.bmi = ctrl.Antecedent(np.arange(10, 61, 1), "bmi")
        self.age = ctrl.Antecedent(np.arange(1, 101, 1), "age")
        self.activity = ctrl.Antecedent(np.arange(0, 11, 1), "activity")

        # Consequent (output)
        self.risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")

        # Membership functions -------------------------------------------------
        self.glucose["low"] = fuzz.trimf(self.glucose.universe, [0, 0, 100])
        self.glucose["normal"] = fuzz.trimf(self.glucose.universe, [70, 110, 140])
        self.glucose["high"] = fuzz.trimf(self.glucose.universe, [125, 200, 300])

        self.bmi["low"] = fuzz.trimf(self.bmi.universe, [10, 10, 20])
        self.bmi["normal"] = fuzz.trimf(self.bmi.universe, [18, 23, 27])
        self.bmi["high"] = fuzz.trimf(self.bmi.universe, [25, 32, 40])
        self.bmi["very_high"] = fuzz.trimf(self.bmi.universe, [35, 50, 60])

        self.age["young"] = fuzz.trimf(self.age.universe, [1, 1, 35])
        self.age["middle"] = fuzz.trimf(self.age.universe, [25, 45, 60])
        self.age["senior"] = fuzz.trimf(self.age.universe, [50, 100, 100])

        self.activity["low"] = fuzz.trimf(self.activity.universe, [0, 0, 4])
        self.activity["moderate"] = fuzz.trimf(self.activity.universe, [3, 5, 7])
        self.activity["high"] = fuzz.trimf(self.activity.universe, [6, 10, 10])

        self.risk["very_low"] = fuzz.trimf(self.risk.universe, [0, 0, 20])
        self.risk["low"] = fuzz.trimf(self.risk.universe, [10, 25, 40])
        self.risk["moderate"] = fuzz.trimf(self.risk.universe, [30, 50, 70])
        self.risk["high"] = fuzz.trimf(self.risk.universe, [60, 75, 90])
        self.risk["very_high"] = fuzz.trimf(self.risk.universe, [80, 100, 100])

        # Rule base ---------------------------------------------------------
        self._rule_defs = [
            ("Glucose HIGH AND BMI HIGH -> Risk HIGH",
             self.glucose["high"] & self.bmi["high"], self.risk["high"]),
            ("Glucose HIGH AND BMI VERY_HIGH -> Risk VERY_HIGH",
             self.glucose["high"] & self.bmi["very_high"], self.risk["very_high"]),
            ("Glucose NORMAL AND BMI NORMAL -> Risk LOW",
             self.glucose["normal"] & self.bmi["normal"], self.risk["low"]),
            ("Glucose LOW AND BMI LOW -> Risk VERY_LOW",
             self.glucose["low"] & self.bmi["low"], self.risk["very_low"]),
            ("Age SENIOR AND Activity LOW -> Risk MODERATE",
             self.age["senior"] & self.activity["low"], self.risk["moderate"]),
            ("Glucose HIGH AND Activity LOW -> Risk VERY_HIGH",
             self.glucose["high"] & self.activity["low"], self.risk["very_high"]),
            ("BMI HIGH AND Activity LOW -> Risk HIGH",
             self.bmi["high"] & self.activity["low"], self.risk["high"]),
            ("Activity HIGH AND BMI NORMAL -> Risk VERY_LOW",
             self.activity["high"] & self.bmi["normal"], self.risk["very_low"]),
            ("Glucose NORMAL AND Age YOUNG -> Risk VERY_LOW",
             self.glucose["normal"] & self.age["young"], self.risk["very_low"]),
            ("Glucose HIGH AND Age SENIOR -> Risk VERY_HIGH",
             self.glucose["high"] & self.age["senior"], self.risk["very_high"]),
            ("BMI VERY_HIGH AND Activity LOW -> Risk VERY_HIGH",
             self.bmi["very_high"] & self.activity["low"], self.risk["very_high"]),
            ("Glucose NORMAL AND BMI HIGH -> Risk MODERATE",
             self.glucose["normal"] & self.bmi["high"], self.risk["moderate"]),
        ]

        rules = [ctrl.Rule(cond, concl) for _, cond, concl in self._rule_defs]
        self.system = ctrl.ControlSystem(rules)

    # ------------------------------------------------------------------
    def infer(self, glucose: float, bmi: float, age: float, activity: float) -> FuzzyResult:
        """Run fuzzy inference for a single patient.

        Args:
            glucose: Glucose level (mg/dL).
            bmi: Body Mass Index.
            age: Age in years.
            activity: Physical activity score (0-10).

        Returns:
            A :class:`FuzzyResult` with the crisp risk score, linguistic
            label, list of fired rule descriptions, and membership degrees.
        """
        sim = ctrl.ControlSystemSimulation(self.system)
        glucose_c = float(np.clip(glucose, 0, 300))
        bmi_c = float(np.clip(bmi, 10, 60))
        age_c = float(np.clip(age, 1, 100))
        activity_c = float(np.clip(activity, 0, 10))

        sim.input["glucose"] = glucose_c
        sim.input["bmi"] = bmi_c
        sim.input["age"] = age_c
        sim.input["activity"] = activity_c

        try:
            sim.compute()
            risk_score = float(sim.output["risk"])
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Fuzzy computation failed (%s); falling back to heuristic.", exc)
            risk_score = self._fallback_score(glucose_c, bmi_c, age_c, activity_c)

        label = self._score_to_label(risk_score)
        memberships = self._compute_memberships(glucose_c, bmi_c, age_c, activity_c)
        fired = self._fired_rules(memberships)

        return FuzzyResult(
            risk_score=round(risk_score, 2),
            risk_label=label,
            fired_rules=fired,
            memberships=memberships,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_score(glucose: float, bmi: float, age: float, activity: float) -> float:
        """Simple weighted heuristic used only if the fuzzy solver fails
        (e.g. no rule fired for an edge-case input)."""
        score = (
            0.45 * (glucose / 300) * 100
            + 0.30 * (bmi / 60) * 100
            + 0.15 * (age / 100) * 100
            + 0.10 * ((10 - activity) / 10) * 100
        )
        return float(np.clip(score, 0, 100))

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score < 20:
            return "Very Low Risk"
        if score < 40:
            return "Low Risk"
        if score < 60:
            return "Moderate Risk"
        if score < 80:
            return "High Risk"
        return "Very High Risk"

    def _compute_memberships(
        self, glucose: float, bmi: float, age: float, activity: float
    ) -> dict[str, dict[str, float]]:
        """Compute membership degrees for every linguistic term of every
        input variable, for display in the UI."""

        def term_memberships(var: ctrl.Antecedent, value: float) -> dict[str, float]:
            return {
                term: round(float(fuzz.interp_membership(var.universe, mf.mf, value)), 3)
                for term, mf in var.terms.items()
            }

        return {
            "glucose": term_memberships(self.glucose, glucose),
            "bmi": term_memberships(self.bmi, bmi),
            "age": term_memberships(self.age, age),
            "activity": term_memberships(self.activity, activity),
        }

    def _fired_rules(self, memberships: dict[str, dict[str, float]], threshold: float = 0.1) -> list[str]:
        """Return descriptions of rules whose *both* antecedent terms have
        membership above ``threshold`` (approximate firing strength report
        for explainability -- the true firing strength is computed
        internally by scikit-fuzzy)."""
        fired = []
        term_lookup = {
            "glucose": memberships["glucose"],
            "bmi": memberships["bmi"],
            "age": memberships["age"],
            "activity": memberships["activity"],
        }
        for description, _cond, _concl in self._rule_defs:
            # crude parse: "VAR TERM AND VAR TERM -> ..."
            lhs = description.split("->")[0]
            parts = lhs.replace(" AND ", "|").split("|")
            strengths = []
            ok = True
            for part in parts:
                tokens = part.strip().split()
                var_name = tokens[0].lower()
                term_name = tokens[1].lower()
                mship = term_lookup.get(var_name, {}).get(term_name, 0.0)
                strengths.append(mship)
                if mship < threshold:
                    ok = False
            if ok and strengths:
                fired.append(f"{description} (strength={min(strengths):.2f})")
        return fired
