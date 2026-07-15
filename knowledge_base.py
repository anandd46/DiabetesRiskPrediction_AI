"""
knowledge_base.py
==================
Knowledge Representation module.

This module encodes simple medical domain knowledge as an explicit set of
IF-THEN rules, stored as Python objects. These rules are consumed by the
forward-chaining and backward-chaining reasoning engines in ``reasoning.py``.

Representing knowledge explicitly (rather than burying it inside the ML
model) is what makes this system "explainable" -- every conclusion can be
traced back to a human-readable rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rule:
    """A single IF-THEN production rule used for forward/backward reasoning.

    Attributes:
        rule_id: Unique identifier, e.g. "R1".
        description: Human readable description of the rule.
        conditions: Facts that must ALL be present (AND semantics) for the
            rule to fire.
        conclusion: The single fact produced when the rule fires.
    """

    rule_id: str
    description: str
    conditions: tuple[str, ...]
    conclusion: str


# --------------------------------------------------------------------------
# RULE BASE
# --------------------------------------------------------------------------
# The rules below intentionally mirror common, publicly known clinical
# heuristics (e.g. WHO/ADA thresholds for glucose and BMI). They are meant
# for educational / screening demonstration purposes only.
RULES: list[Rule] = [
    Rule("R1", "High glucose implies high blood sugar",
         ("HighGlucose",), "HighBloodSugar"),
    Rule("R2", "BMI above 30 implies obesity",
         ("HighBMI",), "Obese"),
    Rule("R3", "High blood sugar and obesity imply high diabetes risk",
         ("HighBloodSugar", "Obese"), "HighDiabetesRisk"),
    Rule("R4", "Family history combined with high blood sugar raises risk",
         ("FamilyHistoryPositive", "HighBloodSugar"), "HighDiabetesRisk"),
    Rule("R5", "Age above 45 is an aging risk factor",
         ("SeniorAge",), "AgeRiskFactor"),
    Rule("R6", "Aging risk factor plus obesity implies moderate diabetes risk",
         ("AgeRiskFactor", "Obese"), "ModerateDiabetesRisk"),
    Rule("R7", "Low physical activity increases metabolic risk",
         ("LowActivity",), "MetabolicRiskFactor"),
    Rule("R8", "Metabolic risk factor with high blood sugar implies high risk",
         ("MetabolicRiskFactor", "HighBloodSugar"), "HighDiabetesRisk"),
    Rule("R9", "Smoking combined with obesity implies moderate risk",
         ("SmokerPositive", "Obese"), "ModerateDiabetesRisk"),
    Rule("R10", "High blood pressure implies cardiovascular strain",
         ("HighBloodPressure",), "CardiovascularStrain"),
    Rule("R11", "Cardiovascular strain with obesity implies high risk",
         ("CardiovascularStrain", "Obese"), "HighDiabetesRisk"),
    Rule("R12", "High diabetes risk with high pedigree score confirms high risk",
         ("HighDiabetesRisk", "HighPedigree"), "ConfirmedHighRisk"),
    Rule("R13", "No strong risk factors imply low diabetes risk",
         ("NormalGlucose", "NormalBMI"), "LowDiabetesRisk"),
    Rule("R14", "Elevated skin thickness and insulin suggest insulin resistance",
         ("HighSkinThickness", "HighInsulin"), "InsulinResistance"),
    Rule("R15", "Insulin resistance implies high diabetes risk",
         ("InsulinResistance",), "HighDiabetesRisk"),
]


def rules_by_conclusion(conclusion: str) -> list[Rule]:
    """Return every rule whose conclusion matches ``conclusion``.

    Used by the backward-chaining engine to find which rules could have
    produced a given goal fact.
    """
    return [r for r in RULES if r.conclusion == conclusion]


# --------------------------------------------------------------------------
# HEALTH RECOMMENDATION KNOWLEDGE
# --------------------------------------------------------------------------
RECOMMENDATIONS: dict[str, list[str]] = {
    "Low Risk": [
        "Maintain your current healthy lifestyle and balanced diet.",
        "Continue regular physical activity (at least 150 minutes/week).",
        "Schedule a routine health screening once a year.",
    ],
    "Moderate Risk": [
        "Increase physical activity to at least 30 minutes a day.",
        "Reduce intake of refined sugar and processed carbohydrates.",
        "Monitor your BMI and aim for a gradual, healthy weight reduction if overweight.",
        "Schedule a medical screening (HbA1c / fasting glucose test) within the next few months.",
    ],
    "High Risk": [
        "Consult a healthcare professional as soon as possible for a formal diagnostic test.",
        "Adopt a low-glycemic, high-fiber diet and reduce sugar/processed food intake.",
        "Increase physical activity gradually under medical guidance.",
        "Monitor blood pressure and glucose levels regularly.",
        "Discuss family history and personal risk factors with your doctor.",
    ],
}


@dataclass
class KnowledgeBase:
    """Convenience wrapper bundling rules and recommendations together."""

    rules: list[Rule] = field(default_factory=lambda: list(RULES))
    recommendations: dict[str, list[str]] = field(
        default_factory=lambda: dict(RECOMMENDATIONS)
    )

    def get_recommendations(self, risk_band: str) -> list[str]:
        """Return lifestyle recommendations for a given risk band."""
        return self.recommendations.get(risk_band, [])
