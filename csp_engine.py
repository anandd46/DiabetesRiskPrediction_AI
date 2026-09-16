"""
csp_engine.py
=============
Constraint Satisfaction Problem (CSP) engine.

Classical CSP theory defines a problem as a set of *variables*, each with a
*domain* of legal values, and a set of *constraints* restricting the values
variables can simultaneously take. Here we apply that idea to patient input
validation: every clinical field is a variable with a realistic numeric
domain (e.g. Age in [1, 120]). If any assignment violates a constraint, the
CSP is declared *unsatisfiable* and detailed, human-readable explanations
are returned to the user instead of silently accepting bad data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import CSP_CONSTRAINTS, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class ConstraintViolation:
    """Represents a single violated constraint."""

    variable: str
    value: float
    lower: float
    upper: float

    def explain(self) -> str:
        return (
            f"'{self.variable}' = {self.value} is outside the realistic "
            f"domain [{self.lower}, {self.upper}]."
        )


class CSPEngine:
    """A lightweight CSP engine that validates patient input variables.

    Each variable is checked against its domain (a closed interval). This
    is a simplified, unary-constraint CSP (each constraint only involves a
    single variable), plus a couple of binary/relational constraints
    (e.g. Weight/Height consistency) to demonstrate multi-variable
    constraint checking.
    """

    def __init__(self, constraints: dict[str, tuple[float, float]] | None = None):
        self.constraints = constraints or CSP_CONSTRAINTS

    def validate(self, data: dict[str, float]) -> list[ConstraintViolation]:
        """Validate ``data`` against every known domain constraint.

        Args:
            data: Mapping of variable name -> value provided by the user.

        Returns:
            A list of :class:`ConstraintViolation`. An empty list means the
            CSP is satisfied (all constraints hold).
        """
        violations: list[ConstraintViolation] = []

        # --- Unary domain constraints -------------------------------------
        for variable, value in data.items():
            if variable not in self.constraints:
                continue
            lower, upper = self.constraints[variable]
            if value is None:
                continue
            if not (lower <= value <= upper):
                violations.append(
                    ConstraintViolation(variable, value, lower, upper)
                )

        # --- Relational constraint: BMI vs Weight/Height consistency ------
        weight = data.get("Weight")
        height = data.get("Height")
        bmi_reported = data.get("BMI")
        if weight and height and bmi_reported:
            height_m = height / 100.0
            computed_bmi = weight / (height_m ** 2) if height_m > 0 else 0
            if abs(computed_bmi - bmi_reported) > 5:
                violations.append(
                    ConstraintViolation(
                        "BMI(consistency)",
                        bmi_reported,
                        round(computed_bmi - 5, 2),
                        round(computed_bmi + 5, 2),
                    )
                )

        if violations:
            logger.warning("CSP validation failed with %d violation(s).", len(violations))
        else:
            logger.info("CSP validation passed: all constraints satisfied.")

        return violations

    def is_satisfied(self, data: dict[str, float]) -> bool:
        """Return True if all constraints are satisfied."""
        return len(self.validate(data)) == 0
