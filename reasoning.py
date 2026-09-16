"""
reasoning.py
============
Forward-chaining and backward-chaining inference engines that operate over
the rule base defined in ``knowledge_base.py``.

Forward chaining: start from known facts (derived from patient input) and
repeatedly apply rules until no new facts can be derived ("data-driven"
reasoning).

Backward chaining: start from a goal fact (e.g. "HighDiabetesRisk") and work
backwards to discover which chain of rules could prove it ("goal-driven"
reasoning). This is used to build the human-readable explanation trace
shown to the user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import LOG_FORMAT, LOG_LEVEL
from knowledge_base import RULES, Rule, rules_by_conclusion

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# FACT EXTRACTION
# --------------------------------------------------------------------------
def derive_initial_facts(patient: dict[str, float]) -> set[str]:
    """Translate raw numeric patient inputs into symbolic boolean facts.

    Args:
        patient: Dictionary of raw clinical values (Glucose, BMI, Age, ...).

    Returns:
        A set of fact strings usable by the forward/backward reasoners.
    """
    facts: set[str] = set()

    glucose = patient.get("Glucose", 0)
    bmi = patient.get("BMI", 0)
    age = patient.get("Age", 0)
    activity = patient.get("PhysicalActivity", 5)
    bp = patient.get("BloodPressure", 0)
    pedigree = patient.get("DiabetesPedigreeFunction", 0)
    smoking = patient.get("Smoking", 0)
    family_history = patient.get("FamilyHistory", 0)
    skin = patient.get("SkinThickness", 0)
    insulin = patient.get("Insulin", 0)

    facts.add("HighGlucose" if glucose >= 140 else "NormalGlucose")
    facts.add("HighBMI" if bmi >= 30 else "NormalBMI")
    facts.add("SeniorAge" if age >= 45 else "YoungAge")
    facts.add("LowActivity" if activity <= 3 else "AdequateActivity")
    facts.add("HighBloodPressure" if bp >= 130 else "NormalBloodPressure")
    facts.add("HighPedigree" if pedigree >= 0.8 else "NormalPedigree")
    facts.add("SmokerPositive" if smoking == 1 else "NonSmoker")
    facts.add("FamilyHistoryPositive" if family_history == 1 else "NoFamilyHistory")
    facts.add("HighSkinThickness" if skin >= 35 else "NormalSkinThickness")
    facts.add("HighInsulin" if insulin >= 150 else "NormalInsulin")

    return facts


# --------------------------------------------------------------------------
# FORWARD CHAINING
# --------------------------------------------------------------------------
@dataclass
class ForwardChainResult:
    """Result of a forward-chaining reasoning run."""

    initial_facts: set[str]
    derived_facts: set[str]
    firing_trace: list[str] = field(default_factory=list)

    @property
    def all_facts(self) -> set[str]:
        return self.initial_facts | self.derived_facts


def forward_chain(initial_facts: set[str], rules: list[Rule] | None = None) -> ForwardChainResult:
    """Apply forward (data-driven) chaining until a fixed point is reached.

    Repeatedly scans the rule base; whenever every condition of a rule is
    already known, the rule "fires" and its conclusion is added to the
    known facts. This continues until no new facts can be derived.

    Args:
        initial_facts: Facts already known from patient input.
        rules: Rule base to reason over (defaults to the global RULES list).

    Returns:
        A :class:`ForwardChainResult` describing everything that was
        derived, plus a human-readable firing trace for explainability.
    """
    rules = rules if rules is not None else RULES
    known: set[str] = set(initial_facts)
    derived: set[str] = set()
    trace: list[str] = []

    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.conclusion in known:
                continue
            if all(cond in known for cond in rule.conditions):
                known.add(rule.conclusion)
                derived.add(rule.conclusion)
                trace.append(
                    f"{rule.rule_id}: IF {' AND '.join(rule.conditions)} "
                    f"THEN {rule.conclusion}  \u2192  ({rule.description})"
                )
                changed = True

    logger.info("Forward chaining derived %d new fact(s).", len(derived))
    return ForwardChainResult(initial_facts=initial_facts, derived_facts=derived, firing_trace=trace)


# --------------------------------------------------------------------------
# BACKWARD CHAINING
# --------------------------------------------------------------------------
@dataclass
class BackwardChainNode:
    """A single node in the backward-reasoning proof tree."""

    fact: str
    proven: bool
    via_rule: str | None = None
    children: list["BackwardChainNode"] = field(default_factory=list)

    def to_lines(self, depth: int = 0) -> list[str]:
        indent = "    " * depth
        status = "\u2713" if self.proven else "\u2717"
        rule_note = f" (via {self.via_rule})" if self.via_rule else ""
        lines = [f"{indent}{status} {self.fact}{rule_note}"]
        for child in self.children:
            lines.extend(child.to_lines(depth + 1))
        return lines


def backward_chain(
    goal: str, known_facts: set[str], rules: list[Rule] | None = None, _depth: int = 0
) -> BackwardChainNode:
    """Attempt to prove ``goal`` using goal-driven (backward) reasoning.

    Args:
        goal: The fact we want to prove (e.g. "HighDiabetesRisk").
        known_facts: Facts already known to be true (from patient input or
            prior forward chaining).
        rules: Rule base to search (defaults to the global RULES list).

    Returns:
        A :class:`BackwardChainNode` proof tree. ``node.proven`` is True if
        the goal could be established from the known facts and rule base.
    """
    rules = rules if rules is not None else RULES

    if goal in known_facts:
        return BackwardChainNode(fact=goal, proven=True, via_rule="known fact")

    if _depth > 10:  # safety guard against pathological cycles
        return BackwardChainNode(fact=goal, proven=False)

    for rule in rules_by_conclusion(goal):
        child_nodes = [
            backward_chain(cond, known_facts, rules, _depth + 1) for cond in rule.conditions
        ]
        if all(child.proven for child in child_nodes):
            node = BackwardChainNode(fact=goal, proven=True, via_rule=rule.rule_id)
            node.children = child_nodes
            return node

    return BackwardChainNode(fact=goal, proven=False)
