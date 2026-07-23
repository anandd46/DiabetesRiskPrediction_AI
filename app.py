

"""
app.py
======
Streamlit front-end for the Explainable AI-Based Diabetes Risk Prediction
and Early Screening System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    APP_SUBTITLE,
    APP_TITLE,
    DATASET_PATH,
    FEATURE_COLUMNS,
    LOG_FORMAT,
    LOG_LEVEL,
    MEDICAL_DISCLAIMER,
    RISK_COLORS,
)
from csp_engine import CSPEngine
from database import Database
from fuzzy_logic import FuzzyDiabetesRiskSystem
from knowledge_base import KnowledgeBase
from model import DiabetesRiskModel
from reasoning import backward_chain, derive_initial_facts, forward_chain
from utils import age_group_bucket, calculate_bmi, fairness_by_group, generate_pdf_report

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

st.set_page_config(page_title=APP_TITLE, page_icon="\U0001FA7A", layout="wide")


# --------------------------------------------------------------------------
# CACHED RESOURCES
# --------------------------------------------------------------------------
@st.cache_resource
def get_model() -> DiabetesRiskModel:
    return DiabetesRiskModel()


@st.cache_resource
def get_fuzzy_system() -> FuzzyDiabetesRiskSystem:
    return FuzzyDiabetesRiskSystem()


@st.cache_resource
def get_database() -> Database:
    return Database()


@st.cache_resource
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()


@st.cache_data
def load_dataset() -> pd.DataFrame:
    try:
        return pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        return pd.DataFrame()


model = get_model()
fuzzy_system = get_fuzzy_system()
db = get_database()
kb = get_knowledge_base()
csp = CSPEngine()

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None


# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
st.sidebar.title("\U0001FA7A " + APP_TITLE)
st.sidebar.caption(APP_SUBTITLE)
section = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Risk Prediction",
        "Explainability",
        "CSP Validation",
        "Fuzzy Logic",
        "Prediction History",
        "Model Performance",
        "About Project",
    ],
)
st.sidebar.info(MEDICAL_DISCLAIMER)


# --------------------------------------------------------------------------
# HOME
# --------------------------------------------------------------------------
if section == "Home":
    st.title(APP_TITLE)
    st.subheader(APP_SUBTITLE)
    st.warning(MEDICAL_DISCLAIMER)

    st.markdown(
        """
        This project demonstrates how **classical Artificial Intelligence**
        techniques -- Constraint Satisfaction, Fuzzy Logic, and Rule-Based
        Reasoning -- can be combined with **modern Machine Learning** and
        **Explainable AI (SHAP)** to build a transparent, trustworthy health
        screening tool.

        ### What makes this project different?
        - **Not a black box**: every prediction comes with a full reasoning
          trace (forward + backward chaining) and SHAP feature attributions.
        - **Validated inputs**: a Constraint Satisfaction engine rejects
          unrealistic patient data before it ever reaches the model.
        - **Uncertainty-aware**: a fuzzy inference system models risk as a
          spectrum rather than a hard yes/no cut-off.
        - **Fairness-checked**: prediction rates are compared across gender
          and age groups to surface potential bias.

        Use the sidebar to explore each capability.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    df = load_dataset()
    col1.metric("Training Records", f"{len(df):,}" if not df.empty else "N/A")
    col2.metric("Best Model", model.metrics.get("best_model", "N/A"))
    if model.metrics:
        best_acc = model.metrics["results"][model.metrics["best_model"]]["accuracy"]
        col3.metric("Test Accuracy", f"{best_acc * 100:.1f}%")
    else:
        col3.metric("Test Accuracy", "N/A")
    col4.metric("Predictions Logged", db.count())


# --------------------------------------------------------------------------
# RISK PREDICTION
# --------------------------------------------------------------------------
elif section == "Risk Prediction":
    st.title("\U0001F52C Diabetes Risk Prediction")
    st.warning(MEDICAL_DISCLAIMER)

    with st.form("patient_form"):
        st.subheader("Patient Information")
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=35)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
            weight = st.number_input("Weight (kg)", min_value=2.0, max_value=400.0, value=70.0)
            height = st.number_input("Height (cm)", min_value=30.0, max_value=250.0, value=165.0)

        with c2:
            glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=400.0, value=110.0)
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=75.0)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
            insulin = st.number_input("Insulin (mu U/mL)", min_value=0.0, max_value=900.0, value=80.0)

        with c3:
            pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.4, step=0.01)
            activity = st.slider("Physical Activity Level (0=none, 10=very active)", 0, 10, 5)
            smoking = st.selectbox("Smoking Status", ["Non-smoker", "Smoker"])
            family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])

        bmi_auto = calculate_bmi(weight, height)
        st.caption(f"Auto-calculated BMI from weight/height: **{bmi_auto}**")
        bmi_override = st.number_input(
            "BMI (edit if you want to override the auto-calculated value)",
            min_value=5.0, max_value=80.0, value=float(bmi_auto) if bmi_auto else 24.0,
        )

        submitted = st.form_submit_button("Run Risk Assessment", use_container_width=True)

    if submitted:
        patient_raw = {
            "Age": age,
            "Gender": gender,
            "Pregnancies": pregnancies,
            "Weight": weight,
            "Height": height,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi_override,
            "DiabetesPedigreeFunction": pedigree,
            "PhysicalActivity": activity,
            "Smoking": 1 if smoking == "Smoker" else 0,
            "FamilyHistory": 1 if family_history == "Yes" else 0,
        }

        # --- Step 1: CSP validation -----------------------------------
        violations = csp.validate(patient_raw)
        if violations:
            st.error("\u274C Input validation failed (Constraint Satisfaction violated):")
            for v in violations:
                st.write(f"- {v.explain()}")
            st.stop()
        else:
            st.success("\u2705 All constraints satisfied. Input data is realistic and consistent.")

        # --- Step 2: ML Prediction --------------------------------------
        if not model.is_ready:
            st.error("Model artifacts not found. Please run `python train_model.py` first.")
            st.stop()

        result = model.predict(patient_raw)
        risk_pct = result.probability * 100
        confidence_pct = result.confidence * 100

        # --- Step 3: Fuzzy Logic ----------------------------------------
        fuzzy_result = fuzzy_system.infer(glucose, bmi_override, age, activity)

        # --- Step 4: Symbolic reasoning (forward + backward) -------------
        initial_facts = derive_initial_facts(patient_raw)
        fwd_result = forward_chain(initial_facts)
        goal = "HighDiabetesRisk" if result.risk_band != "Low Risk" else "LowDiabetesRisk"
        bwd_node = backward_chain(goal, fwd_result.all_facts)

        # --- Persist to DB -------------------------------------------------
        db.insert_prediction(result.risk_band, risk_pct, confidence_pct, patient_raw)

        st.session_state["last_prediction"] = {
            "patient": patient_raw,
            "result": result,
            "fuzzy_result": fuzzy_result,
            "fwd_result": fwd_result,
            "bwd_node": bwd_node,
        }

        st.divider()
        st.subheader("\U0001F4CA Prediction Result")

        color = RISK_COLORS.get(result.risk_band, "#95a5a6")
        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(
            f"<h2 style='color:{color};'>{result.risk_band}</h2>", unsafe_allow_html=True
        )
        rc2.metric("Risk Percentage (ML)", f"{risk_pct:.1f}%")
        rc3.metric("Confidence Score", f"{confidence_pct:.1f}%")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                title={"text": "ML-Based Diabetes Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 33], "color": "#d4f7dc"},
                        {"range": [33, 66], "color": "#fdeecb"},
                        {"range": [66, 100], "color": "#fadbd8"},
                    ],
                },
            )
        )
        st.plotly_chart(gauge, use_container_width=True)

        st.info(
            f"\U0001F300 **Fuzzy Logic Cross-Check:** {fuzzy_result.risk_label} "
            f"(fuzzy risk score = {fuzzy_result.risk_score}/100)"
        )

        st.subheader("\U0001F4A1 Health Recommendations")
        for rec in kb.get_recommendations(result.risk_band):
            st.write(f"- {rec}")

        # --- PDF export -----------------------------------------------------
        pdf_bytes = generate_pdf_report(
            patient=patient_raw,
            risk_band=result.risk_band,
            risk_score=risk_pct,
            confidence=confidence_pct,
            reasoning_trace=fwd_result.firing_trace,
            recommendations=kb.get_recommendations(result.risk_band),
        )
        st.download_button(
            "\U0001F4C4 Download PDF Report",
            data=pdf_bytes,
            file_name="diabetes_risk_report.pdf",
            mime="application/pdf",
        )

        st.success(
            "Prediction saved to history. Visit **Explainability**, "
            "**Fuzzy Logic**, or **Prediction History** for more detail."
        )


# --------------------------------------------------------------------------
# EXPLAINABILITY
# --------------------------------------------------------------------------
elif section == "Explainability":
    st.title("\U0001F9E0 Explainable AI (XAI)")
    st.warning(MEDICAL_DISCLAIMER)

    last = st.session_state.get("last_prediction")
    if not last:
        st.info("Run a prediction on the **Risk Prediction** page first.")
    else:
        patient = last["patient"]
        result = last["result"]
        fwd_result = last["fwd_result"]
        bwd_node = last["bwd_node"]

        st.subheader("Decision Summary")
        st.write(
            f"The model predicts **{result.risk_band}** with a probability of "
            f"**{result.probability * 100:.1f}%** and confidence of "
            f"**{result.confidence * 100:.1f}%**."
        )

        explanation = model.explain(patient)

        st.subheader("SHAP Feature Contributions")
        shap_df = pd.DataFrame(
            {"Feature": list(explanation.shap_values.keys()), "SHAP Value": list(explanation.shap_values.values())}
        ).sort_values("SHAP Value")
        fig = px.bar(
            shap_df,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            color="SHAP Value",
            color_continuous_scale=["#2ecc71", "#e74c3c"],
            title="SHAP Value per Feature (positive = increases risk)",
        )
        st.plotly_chart(fig, use_container_width=True)

        colp, coln = st.columns(2)
        with colp:
            st.markdown("**\u2795 Positive Contributing Factors (increase risk)**")
            for feat, val in explanation.positive_factors:
                st.write(f"- {feat}: +{val}")
        with coln:
            st.markdown("**\u2796 Negative Contributing Factors (decrease risk)**")
            for feat, val in explanation.negative_factors:
                st.write(f"- {feat}: {val}")

        st.subheader("Global Feature Importance")
        importance = model.feature_importance()
        if importance:
            imp_df = pd.DataFrame({"Feature": list(importance.keys()), "Importance": list(importance.values())})
            st.plotly_chart(px.bar(imp_df, x="Importance", y="Feature", orientation="h"), use_container_width=True)

        st.subheader("\u27A1\uFE0F Forward Reasoning Chain (data-driven)")
        st.caption("Facts derived automatically from patient input, applying rules until no new facts emerge.")
        if fwd_result.firing_trace:
            for step in fwd_result.firing_trace:
                st.write(f"- {step}")
        else:
            st.write("No symbolic rules fired for this patient profile.")

        st.subheader("\u2B05\uFE0F Backward Reasoning Chain (goal-driven)")
        st.caption(f"Tracing which rules justify the goal fact: **{bwd_node.fact}**")
        st.code("\n".join(bwd_node.to_lines()) or "No proof tree available.")


# --------------------------------------------------------------------------
# CSP VALIDATION
# --------------------------------------------------------------------------
elif section == "CSP Validation":
    st.title("\U0001F512 Constraint Satisfaction (CSP) Validation")
    st.markdown(
        """
        This page lets you experiment directly with the CSP engine that
        validates every patient record before it reaches the ML model.
        Each variable has a realistic **domain** (range of legal values);
        the CSP is *satisfied* only if every value lies within its domain
        (and a couple of cross-variable consistency checks also hold).
        """
    )

    test_col1, test_col2 = st.columns(2)
    with test_col1:
        t_age = st.number_input("Age", value=200)
        t_bmi = st.number_input("BMI", value=-5.0)
        t_glucose = st.number_input("Glucose", value=110.0)
    with test_col2:
        t_bp = st.number_input("Blood Pressure", value=80.0)
        t_weight = st.number_input("Weight (kg)", value=70.0)
        t_height = st.number_input("Height (cm)", value=165.0)

    if st.button("Validate Constraints"):
        test_data = {
            "Age": t_age, "BMI": t_bmi, "Glucose": t_glucose,
            "BloodPressure": t_bp, "Weight": t_weight, "Height": t_height,
        }
        violations = csp.validate(test_data)
        if violations:
            st.error(f"CSP unsatisfiable: {len(violations)} violation(s) found.")
            for v in violations:
                st.write(f"- {v.explain()}")
        else:
            st.success("CSP satisfied: all values fall within realistic medical domains.")

    st.subheader("Configured Domains")
    from config import CSP_CONSTRAINTS
    domain_df = pd.DataFrame(
        [{"Variable": k, "Min": v[0], "Max": v[1]} for k, v in CSP_CONSTRAINTS.items()]
    )
    st.dataframe(domain_df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# FUZZY LOGIC
# --------------------------------------------------------------------------
elif section == "Fuzzy Logic":
    st.title("\U0001F300 Fuzzy Inference System")
    st.markdown(
        "Adjust the sliders below to see how the fuzzy inference system "
        "reasons under uncertainty, blending multiple linguistic terms "
        "(e.g. Glucose can be simultaneously a little *normal* and a "
        "little *high*)."
    )

    f1, f2, f3, f4 = st.columns(4)
    fg = f1.slider("Glucose", 0, 300, 130)
    fb = f2.slider("BMI", 10, 60, 29)
    fa = f3.slider("Age", 1, 100, 40)
    fact = f4.slider("Activity", 0, 10, 4)

    fuzzy_out = fuzzy_system.infer(fg, fb, fa, fact)

    st.metric("Fuzzy Risk Score", f"{fuzzy_out.risk_score}/100", fuzzy_out.risk_label)

    st.subheader("Membership Degrees")
    for var_name, terms in fuzzy_out.memberships.items():
        mdf = pd.DataFrame({"Term": list(terms.keys()), "Membership": list(terms.values())})
        st.write(f"**{var_name.capitalize()}**")
        st.bar_chart(mdf.set_index("Term"))

    st.subheader("Fired Rules")
    if fuzzy_out.fired_rules:
        for rule in fuzzy_out.fired_rules:
            st.write(f"- {rule}")
    else:
        st.write("No rules fired strongly for this combination of inputs.")


# --------------------------------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------------------------------
elif section == "Prediction History":
    st.title("\U0001F5C3\uFE0F Prediction History")

    records = db.get_history()
    if not records:
        st.info("No predictions stored yet. Run a prediction first.")
    else:
        hist_df = pd.DataFrame(
            [
                {
                    "ID": r.id,
                    "Timestamp": r.timestamp,
                    "Risk Band": r.risk_band,
                    "Risk Score (%)": round(r.risk_score, 1),
                    "Confidence (%)": round(r.confidence, 1),
                }
                for r in records
            ]
        )
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        col_del, col_clear = st.columns(2)
        with col_del:
            del_id = st.number_input("Record ID to delete", min_value=0, step=1, value=0)
            if st.button("Delete Record"):
                db.delete_record(int(del_id))
                st.rerun()
        with col_clear:
            if st.button("\U0001F5D1\uFE0F Clear Entire History", type="secondary"):
                db.clear_all()
                st.rerun()


# --------------------------------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------------------------------
elif section == "Model Performance":
    st.title("\U0001F4C8 Model Performance & Fairness")

    if not model.metrics:
        st.warning("No metrics found. Run `python train_model.py` first.")
    else:
        results = model.metrics["results"]
        best = model.metrics["best_model"]
        st.success(f"Best-performing model (auto-selected): **{best}**")

        comp_df = pd.DataFrame(
            [
                {
                    "Model": name,
                    "Accuracy": r["accuracy"],
                    "Precision": r["precision"],
                    "Recall": r["recall"],
                    "F1 Score": r["f1_score"],
                    "ROC AUC": r["roc_auc"],
                }
                for name, r in results.items()
            ]
        )
        st.subheader("Model Comparison")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.bar(comp_df, x="Model", y=["Accuracy", "F1 Score", "ROC AUC"], barmode="group"),
            use_container_width=True,
        )

        st.subheader(f"Confusion Matrix ({best})")
        cm = np.array(results[best]["confusion_matrix"])
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["No Diabetes", "Diabetes"], y=["No Diabetes", "Diabetes"],
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader(f"ROC Curve ({best})")
        roc = results[best]["roc_curve"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=best))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    df = load_dataset()
    if not df.empty:
        st.subheader("Dataset Exploration")
        d1, d2 = st.columns(2)
        d1.plotly_chart(px.histogram(df, x="BMI", nbins=30, title="BMI Distribution"), use_container_width=True)
        d2.plotly_chart(px.histogram(df, x="Glucose", nbins=30, title="Glucose Distribution"), use_container_width=True)

        st.subheader("Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)

        st.subheader("\u2696\uFE0F Fairness Analysis")
        st.caption(
            "Comparing prediction (Outcome) rates across simulated demographic "
            "groups. This is a simplified fairness check for demonstration "
            "purposes, not a substitute for a rigorous bias audit."
        )
        sim_gender = np.random.default_rng(0).choice(["Female", "Male"], size=len(df))
        df_fair = df.copy()
        df_fair["Gender"] = sim_gender
        df_fair["AgeGroup"] = df_fair["Age"].apply(age_group_bucket)

        fg1, fg2 = st.columns(2)
        with fg1:
            st.write("**By Gender**")
            st.dataframe(fairness_by_group(df_fair, "Gender"), hide_index=True)
        with fg2:
            st.write("**By Age Group**")
            st.dataframe(fairness_by_group(df_fair, "AgeGroup"), hide_index=True)

        st.info(
            "**Limitation:** Gender was randomly simulated for this "
            "demonstration since the training data does not include a real "
            "gender field. In a production system, fairness analysis must "
            "use genuine demographic data collected and stored responsibly, "
            "and should be paired with formal fairness metrics (e.g. "
            "demographic parity, equalized odds)."
        )


# --------------------------------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------------------------------
elif section == "About Project":
    st.title("\u2139\uFE0F About This Project")
    st.warning(MEDICAL_DISCLAIMER)
    st.markdown(
        """
        ### Explainable AI-Based Diabetes Risk Prediction and Early Screening System

        This project was built to demonstrate how **classical AI reasoning
        techniques** and **modern explainable machine learning** can work
        together in a single, transparent application.

        **Core AI concepts demonstrated:**
        - Constraint Satisfaction Problem (CSP) for input validation
        - Fuzzy Logic for reasoning under uncertainty
        - Knowledge Representation using explicit IF-THEN rules
        - Forward Chaining (data-driven reasoning)
        - Backward Chaining (goal-driven reasoning / proof trees)
        - Machine Learning (Random Forest, Logistic Regression, Decision Tree)
        - Explainable AI via SHAP
        - Fairness analysis across demographic groups

        See **README.md** in the project root for the full technical
        write-up, architecture diagram, installation guide, and interview
        Q&A prepared alongside this project.

        This tool is a **portfolio / educational project** and is **not
        approved for clinical use**.
        """
    )
