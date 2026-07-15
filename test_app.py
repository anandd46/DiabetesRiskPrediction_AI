

"""
test_app.py
============
Simple test suite for the Diabetes Risk Prediction system.

Run with:
    pytest test_app.py -v
or:
    python -m pytest test_app.py -v
"""

from __future__ import annotations

import os

import pytest

from csp_engine import CSPEngine
from database import Database
from fuzzy_logic import FuzzyDiabetesRiskSystem
from model import DiabetesRiskModel
from reasoning import backward_chain, derive_initial_facts, forward_chain

SAMPLE_PATIENT = {
    "Age": 50,
    "Pregnancies": 2,
    "Glucose": 175,
    "BloodPressure": 88,
    "SkinThickness": 32,
    "Insulin": 160,
    "BMI": 34.5,
    "DiabetesPedigreeFunction": 0.95,
    "PhysicalActivity": 2,
    "Smoking": 1,
    "FamilyHistory": 1,
}

LOW_RISK_PATIENT = {
    "Age": 25,
    "Pregnancies": 0,
    "Glucose": 90,
    "BloodPressure": 70,
    "SkinThickness": 18,
    "Insulin": 60,
    "BMI": 21.0,
    "DiabetesPedigreeFunction": 0.15,
    "PhysicalActivity": 8,
    "Smoking": 0,
    "FamilyHistory": 0,
}


# --------------------------------------------------------------------------
# MODEL / PREDICTION TESTS
# --------------------------------------------------------------------------
class TestPrediction:
    def test_model_loads(self) -> None:
        model = DiabetesRiskModel()
        assert model.is_ready, "Model artifacts should be loadable. Run train_model.py first."

    def test_predict_returns_valid_probability(self) -> None:
        model = DiabetesRiskModel()
        result = model.predict(SAMPLE_PATIENT)
        assert 0.0 <= result.probability <= 1.0
        assert result.risk_band in {"Low Risk", "Moderate Risk", "High Risk"}
        assert 0.0 <= result.confidence <= 1.0

    def test_high_risk_scores_higher_than_low_risk(self) -> None:
        model = DiabetesRiskModel()
        high = model.predict(SAMPLE_PATIENT)
        low = model.predict(LOW_RISK_PATIENT)
        assert high.probability > low.probability

    def test_explain_returns_shap_values(self) -> None:
        model = DiabetesRiskModel()
        explanation = model.explain(SAMPLE_PATIENT)
        assert "Glucose" in explanation.shap_values
        assert isinstance(explanation.positive_factors, list)
        assert isinstance(explanation.negative_factors, list)

    def test_feature_importance_not_empty(self) -> None:
        model = DiabetesRiskModel()
        importance = model.feature_importance()
        assert len(importance) > 0


# --------------------------------------------------------------------------
# CSP VALIDATION TESTS
# --------------------------------------------------------------------------
class TestCSPValidation:
    def setup_method(self) -> None:
        self.csp = CSPEngine()

    def test_valid_input_has_no_violations(self) -> None:
        violations = self.csp.validate({"Age": 40, "BMI": 25, "Glucose": 110})
        assert violations == []

    def test_invalid_age_detected(self) -> None:
        violations = self.csp.validate({"Age": 999})
        assert len(violations) == 1
        assert violations[0].variable == "Age"

    def test_negative_bmi_detected(self) -> None:
        violations = self.csp.validate({"BMI": -10})
        assert any(v.variable == "BMI" for v in violations)

    def test_is_satisfied_true_for_realistic_values(self) -> None:
        assert self.csp.is_satisfied({"Age": 30, "Glucose": 100, "BMI": 22})

    def test_is_satisfied_false_for_unrealistic_values(self) -> None:
        assert not self.csp.is_satisfied({"Age": -5})


# --------------------------------------------------------------------------
# FUZZY LOGIC TESTS
# --------------------------------------------------------------------------
class TestFuzzyLogic:
    def setup_method(self) -> None:
        self.fuzzy = FuzzyDiabetesRiskSystem()

    def test_infer_returns_valid_score_range(self) -> None:
        result = self.fuzzy.infer(glucose=150, bmi=30, age=45, activity=4)
        assert 0.0 <= result.risk_score <= 100.0
        assert result.risk_label in {
            "Very Low Risk", "Low Risk", "Moderate Risk", "High Risk", "Very High Risk",
        }

    def test_high_inputs_yield_higher_risk_than_low_inputs(self) -> None:
        high = self.fuzzy.infer(glucose=250, bmi=40, age=60, activity=1)
        low = self.fuzzy.infer(glucose=85, bmi=20, age=22, activity=9)
        assert high.risk_score > low.risk_score

    def test_memberships_are_bounded(self) -> None:
        result = self.fuzzy.infer(glucose=150, bmi=30, age=45, activity=4)
        for terms in result.memberships.values():
            for value in terms.values():
                assert 0.0 <= value <= 1.0


# --------------------------------------------------------------------------
# REASONING (FORWARD / BACKWARD CHAINING) TESTS
# --------------------------------------------------------------------------
class TestReasoning:
    def test_forward_chain_derives_high_diabetes_risk(self) -> None:
        facts = derive_initial_facts(SAMPLE_PATIENT)
        result = forward_chain(facts)
        assert "HighDiabetesRisk" in result.derived_facts

    def test_backward_chain_proves_known_fact(self) -> None:
        facts = derive_initial_facts(SAMPLE_PATIENT)
        fwd = forward_chain(facts)
        node = backward_chain("HighDiabetesRisk", fwd.all_facts)
        assert node.proven

    def test_backward_chain_fails_for_unreachable_goal(self) -> None:
        facts = derive_initial_facts(LOW_RISK_PATIENT)
        node = backward_chain("NonExistentGoalFact", facts)
        assert not node.proven


# --------------------------------------------------------------------------
# DATABASE TESTS
# --------------------------------------------------------------------------
class TestDatabase:
    TEST_DB_PATH = "test_temp_database.db"

    def setup_method(self) -> None:
        if os.path.exists(self.TEST_DB_PATH):
            os.remove(self.TEST_DB_PATH)
        self.db = Database(self.TEST_DB_PATH)

    def teardown_method(self) -> None:
        if os.path.exists(self.TEST_DB_PATH):
            os.remove(self.TEST_DB_PATH)

    def test_insert_and_retrieve(self) -> None:
        record_id = self.db.insert_prediction("High Risk", 82.5, 65.0, SAMPLE_PATIENT)
        history = self.db.get_history()
        assert len(history) == 1
        assert history[0].id == record_id
        assert history[0].risk_band == "High Risk"

    def test_delete_record(self) -> None:
        record_id = self.db.insert_prediction("Low Risk", 12.0, 76.0, LOW_RISK_PATIENT)
        self.db.delete_record(record_id)
        assert self.db.count() == 0

    def test_clear_all(self) -> None:
        self.db.insert_prediction("High Risk", 82.5, 65.0, SAMPLE_PATIENT)
        self.db.insert_prediction("Low Risk", 12.0, 76.0, LOW_RISK_PATIENT)
        self.db.clear_all()
        assert self.db.count() == 0

    def test_count_increments(self) -> None:
        assert self.db.count() == 0
        self.db.insert_prediction("Moderate Risk", 50.0, 30.0, SAMPLE_PATIENT)
        assert self.db.count() == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
