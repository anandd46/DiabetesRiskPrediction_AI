# 🩺 Explainable AI-Based Diabetes Risk Prediction and Early Screening System

**Constraint Satisfaction • Fuzzy Logic • Rule-Based Reasoning • Explainable Machine Learning**

> ⚠️ **Medical Disclaimer:** This project is a **risk screening and educational tool only**. It is **NOT a medical diagnosis system** and must never replace the advice of a qualified healthcare professional. If you are experiencing symptoms, please consult a doctor.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation](#motivation)
3. [Problem Statement](#problem-statement)
4. [Objectives](#objectives)
5. [AI Concepts Used](#ai-concepts-used)
6. [Machine Learning Workflow](#machine-learning-workflow)
7. [Constraint Satisfaction Problem (CSP)](#constraint-satisfaction-problem-csp)
8. [Fuzzy Logic](#fuzzy-logic)
9. [Explainable AI](#explainable-ai)
10. [Knowledge Representation](#knowledge-representation)
11. [Forward Reasoning](#forward-reasoning)
12. [Backward Reasoning](#backward-reasoning)
13. [Fairness Analysis](#fairness-analysis)
14. [Transparency](#transparency)
15. [Dataset Description](#dataset-description)
16. [Project Architecture](#project-architecture)
17. [Technologies Used](#technologies-used)
18. [Installation Guide](#installation-guide)
19. [Requirements](#requirements)
20. [How to Run](#how-to-run)
21. [Screenshots](#screenshots)
22. [Project Structure](#project-structure)
23. [Feature Descriptions](#feature-descriptions)
24. [Algorithms Used](#algorithms-used)
25. [Model Evaluation](#model-evaluation)
26. [Sample Predictions](#sample-predictions)
27. [Future Enhancements](#future-enhancements)
28. [Limitations](#limitations)
29. [Resume Highlights](#resume-highlights)
30. [Interview Questions and Answers](#interview-questions-and-answers)
31. [Learning Outcomes](#learning-outcomes)
32. [References](#references)
33. [License](#license)

---

## Project Overview

The **Explainable AI-Based Diabetes Risk Prediction and Early Screening System** is a self-contained Python + Streamlit application that estimates a person's risk of diabetes from routine clinical and lifestyle inputs. What sets this project apart from a typical "train a classifier and deploy it" tutorial is that it deliberately combines **classical, symbolic Artificial Intelligence** — Constraint Satisfaction, Fuzzy Logic, and Rule-Based Reasoning — with **modern statistical Machine Learning** and **Explainable AI (SHAP)**.

Instead of treating the ML model as an opaque black box, the system wraps it in three complementary layers of transparency:

1. A **Constraint Satisfaction Problem (CSP) engine** that guarantees every input is medically realistic *before* it ever reaches the model.
2. A **Fuzzy Inference System** that cross-checks the ML prediction using human-like linguistic reasoning ("glucose is somewhat high", "activity is low").
3. A **symbolic knowledge base with forward and backward chaining** that produces a readable, rule-by-rule justification for the outcome, on top of **SHAP** feature attributions for the ML model itself.

The result is a screening tool that not only predicts *what* the risk is, but clearly explains *why*, in language a non-technical user (or a recruiter reviewing your GitHub) can follow.

---

## Motivation

Machine learning models used in healthcare are frequently criticized for being "black boxes" — they might be accurate, but nobody, not even the developer, can fully explain a specific prediction. In high-stakes domains like health screening, this lack of transparency is a serious problem: patients and clinicians need to *trust* and *understand* a recommendation before acting on it.

At the same time, most **classical AI coursework** (CSP, fuzzy logic, forward/backward chaining, knowledge representation) is taught in isolation from the **modern ML pipeline** most practitioners actually build day to day. This project was created to bridge that gap — to show that "old school" symbolic AI concepts are not obsolete, but instead make an excellent **explainability and validation layer** around a modern ML model.

---

## Problem Statement

Given a set of routine patient measurements (glucose level, BMI, blood pressure, age, lifestyle factors, family history, etc.), can we:

1. **Validate** that the input data is medically plausible (CSP)?
2. **Predict** the likelihood that the patient is at risk of diabetes (ML)?
3. **Explain** exactly why that prediction was made, using both statistical (SHAP) and symbolic (rule-based) reasoning?
4. **Reason under uncertainty** about risk as a continuous spectrum rather than a hard binary label (Fuzzy Logic)?
5. **Audit** the system for potential unfairness across demographic groups?

All of this must happen in a simple, single-folder, local-first Python project that a beginner can run and understand, yet impressive enough to be discussed confidently in a technical interview.

---

## Objectives

- Build a working end-to-end diabetes risk screening pipeline.
- Demonstrate mastery of classical AI: CSP, Fuzzy Logic, Knowledge Representation, Forward/Backward Chaining.
- Demonstrate mastery of applied ML: data generation, training, model comparison, model selection.
- Demonstrate Explainable AI using SHAP (both local and global explanations).
- Demonstrate Responsible AI practices: fairness analysis, transparency, and clear medical disclaimers.
- Package everything into a clean, well-documented, testable, recruiter-ready Streamlit application.

---

## AI Concepts Used

| Concept | Where it appears |
|---|---|
| Constraint Satisfaction Problem (CSP) | `csp_engine.py` — validates every patient field against a realistic domain |
| Fuzzy Logic / Reasoning under Uncertainty | `fuzzy_logic.py` — Mamdani fuzzy inference system built with scikit-fuzzy |
| Knowledge Representation | `knowledge_base.py` — explicit IF-THEN rule base as Python dataclasses |
| Forward Reasoning (data-driven) | `reasoning.py::forward_chain` — derives new facts until a fixed point |
| Backward Reasoning (goal-driven) | `reasoning.py::backward_chain` — builds a proof tree for a target conclusion |
| Machine Learning | `train_model.py`, `model.py` — Random Forest, Logistic Regression, Decision Tree |
| Explainable AI (XAI) | `model.py::explain` — SHAP values, local + global feature importance |
| Responsible AI / Fairness | `utils.py::fairness_by_group` — prediction-rate comparison across groups |

---

## Machine Learning Workflow

1. **Data Generation** — `train_model.py` synthesizes a realistic dataset (`dataset.csv`) whose feature distributions are modelled on the well-known Pima Indians Diabetes Dataset, extended with lifestyle features (physical activity, smoking, family history). The target label is generated from a calibrated logistic combination of risk factors plus stochastic noise, so the data has genuine learnable structure and a realistic ~30-35% diabetes prevalence.
2. **Preprocessing** — Features are scaled using `StandardScaler` and split into train/test sets (80/20, stratified).
3. **Model Training** — Three classical classifiers are trained: **Random Forest**, **Logistic Regression**, and **Decision Tree**.
4. **Model Comparison** — Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, and ROC curve are computed for each model.
5. **Automatic Model Selection** — The model with the highest F1-score is automatically selected as the "best model" and persisted to disk (`diabetes_model.pkl`), along with the fitted scaler (`scaler.pkl`) and a metrics report (`model_metrics.json`).
6. **Inference** — `model.py::DiabetesRiskModel` loads the saved artifacts and exposes a clean `predict()` / `explain()` API used by the Streamlit app.

---

## Constraint Satisfaction Problem (CSP)

Classical CSP theory defines a problem using three components:

- **Variables** — e.g. Age, Glucose, BMI, Blood Pressure.
- **Domains** — the realistic range of legal values for each variable (e.g. Age ∈ [1, 120]).
- **Constraints** — rules restricting which combinations of values are acceptable.

`csp_engine.py` implements exactly this: each clinical field is checked against a configured domain in `config.CSP_CONSTRAINTS`, and a secondary **relational constraint** cross-checks that the reported BMI is consistent with the reported weight and height. If any constraint fails, the CSP is *unsatisfiable*, and the engine returns detailed, human-readable `ConstraintViolation` explanations — the input is rejected *before* it ever reaches the ML model, preventing garbage-in/garbage-out predictions.

---

## Fuzzy Logic

Real medical thresholds are not hard cut-offs. A glucose reading of 139 mg/dL is not meaningfully different from 141 mg/dL, yet a crisp rule ("high if ≥ 140") would classify them completely differently. **Fuzzy Logic** solves this by allowing a value to partially belong to multiple linguistic categories simultaneously (e.g. glucose can be 70% "high" and 30% "normal").

`fuzzy_logic.py` implements a **Mamdani-style Fuzzy Inference System** using `scikit-fuzzy`, with:

- **Inputs:** Glucose, BMI, Age, Physical Activity — each with triangular membership functions defining linguistic terms (low/normal/high, young/middle/senior, etc.).
- **Output:** a crisp Risk score (0–100), mapped to five linguistic risk bands: *Very Low, Low, Moderate, High, Very High*.
- **Rule base:** 12 human-readable fuzzy rules such as *"IF Glucose is HIGH AND BMI is HIGH THEN Risk is HIGH"*.
- **Explainability:** the system reports which rules fired and their approximate firing strength, plus the raw membership degree of every input in every linguistic term — all visualized in the **Fuzzy Logic** page of the Streamlit app.

This fuzzy score acts as an independent, human-interpretable cross-check against the statistical ML prediction.

---

## Explainable AI

Explainability is implemented at two levels:

1. **Local explanations (SHAP):** For every individual prediction, `model.py::DiabetesRiskModel.explain()` computes SHAP (SHapley Additive exPlanations) values for each feature, showing exactly how much each input pushed the prediction toward or away from "diabetic". These are surfaced as **positive contributing factors** (increase risk) and **negative contributing factors** (decrease risk), plus a bar chart of all SHAP values.
2. **Global explanations:** `model.py::DiabetesRiskModel.feature_importance()` extracts the model's built-in global feature importances (or absolute coefficients for linear models), shown on the **Explainability** page.

On top of the statistical explanation, the app also shows the **symbolic reasoning chain** (see Forward/Backward Reasoning below) — giving two independent, complementary views into "why" a prediction was made.

---

## Knowledge Representation

`knowledge_base.py` encodes medical domain knowledge as an explicit list of `Rule` objects (IF-THEN production rules), for example:

```
IF HighBMI THEN Obese
IF HighGlucose THEN HighBloodSugar
IF HighBloodSugar AND Obese THEN HighDiabetesRisk
```

Representing knowledge this way — rather than hiding it inside model weights — is what allows the forward and backward reasoning engines to produce fully human-readable explanations.

---

## Forward Reasoning

Forward chaining is **data-driven**: starting from facts known about the patient (e.g. `HighGlucose`, `HighBMI`), the engine (`reasoning.py::forward_chain`) repeatedly scans the rule base and "fires" any rule whose conditions are already satisfied, adding its conclusion to the known facts. This repeats until no new facts can be derived (a fixed point), building a full firing trace:

```
HighGlucose → HighBloodSugar → (combined with Obese) → HighDiabetesRisk
```

---

## Backward Reasoning

Backward chaining is **goal-driven**: starting from a target conclusion (e.g. `HighDiabetesRisk`), the engine (`reasoning.py::backward_chain`) works backwards, recursively asking "which rule could prove this, and are *its* conditions provable?" — building a proof tree that is displayed to the user as a chain of ✓/✗ marks, for example:

```
✓ HighDiabetesRisk (via R3)
    ✓ HighBloodSugar (via R1)
        ✓ HighGlucose (known fact)
    ✓ Obese (via R2)
        ✓ HighBMI (known fact)
```

---

## Fairness Analysis

Responsible AI requires checking whether a model behaves consistently across demographic groups. The **Model Performance** page includes a simplified fairness audit (`utils.py::fairness_by_group`) comparing positive-prediction rates across **Gender** and **Age Group** buckets. The app is transparent about the fact that gender is *simulated* for this demonstration (the synthetic training data does not include a genuine gender field), and explicitly states this as a limitation — a real deployment would need genuine demographic data and formal fairness metrics such as demographic parity or equalized odds.

---

## Transparency

Every prediction the app makes is accompanied by:
- The exact risk probability and confidence score.
- The SHAP-based statistical explanation.
- The full symbolic forward/backward reasoning trace.
- The fuzzy-logic cross-check score.
- A downloadable PDF report summarizing all of the above.

Nothing is hidden — this is the core design philosophy of the project.

---

## Dataset Description

Since the classic Pima Indians Diabetes Dataset does not include several lifestyle fields requested by this project (physical activity, smoking, family history), `train_model.py` **generates a realistic synthetic dataset** whose feature distributions are calibrated to match the statistical shape of the Pima dataset (e.g. glucose ~N(120, 32), BMI ~N(31, 7)), and whose target label is derived from a logistic combination of the real risk factors, giving the dataset genuine, learnable structure rather than pure noise. This keeps the project fully **offline and reproducible** — no external downloads or API keys are required.

Columns: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, PhysicalActivity, Smoking, FamilyHistory, Outcome`.

---

## Project Architecture

```
                 ┌─────────────────────┐
                 │   Streamlit UI       │
                 │      (app.py)        │
                 └──────────┬───────────┘
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
┌───────▼───────┐  ┌────────▼────────┐  ┌────────▼─────────┐
│  CSP Engine    │  │  ML Model        │  │  Fuzzy Logic      │
│ (csp_engine.py)│  │ (model.py)       │  │ (fuzzy_logic.py)  │
└───────┬────────┘  └────────┬────────┘  └────────┬──────────┘
        │                    │                    │
        │           ┌────────▼────────┐           │
        └──────────►│ Reasoning Engine │◄──────────┘
                     │ (reasoning.py +  │
                     │ knowledge_base.py)│
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │ SQLite Database   │
                     │ (database.py)     │
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │ PDF / Utils        │
                     │ (utils.py)         │
                     └────────────────────┘
```

---

## Technologies Used

- **Python 3.10+**
- **Streamlit** — interactive web UI
- **Pandas / NumPy** — data manipulation
- **Scikit-learn** — Random Forest, Logistic Regression, Decision Tree
- **Scikit-Fuzzy** — fuzzy inference system
- **SHAP** — explainable AI
- **Plotly / Matplotlib** — visualizations
- **SQLite** — local temporary database
- **FPDF2** — PDF report generation

No cloud services, Docker, or authentication are used — everything runs locally.

---

## Installation Guide

```bash
# 1. Clone or download the project folder
cd DiabetesRiskPrediction

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Requirements

See `requirements.txt`. Core packages: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `scikit-fuzzy`, `shap`, `plotly`, `matplotlib`, `fpdf2`, `joblib`, `pytest`.

---

## How to Run

```bash
# Step 1: Train the model (generates dataset.csv, diabetes_model.pkl, scaler.pkl, model_metrics.json)
python train_model.py

# Step 2: Launch the Streamlit app
streamlit run app.py

# Step 3 (optional): Run the test suite
pytest test_app.py -v
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Screenshots

> _Add screenshots of your running app here for your portfolio, e.g.:_
> - `screenshot_home.png` — Home dashboard
> - `screenshot_prediction.png` — Risk Prediction form and gauge
> - `screenshot_explainability.png` — SHAP charts and reasoning trace
> - `screenshot_fuzzy.png` — Fuzzy membership sliders
> - `screenshot_performance.png` — Model comparison and fairness analysis

---

## Project Structure

All files live in a **single root folder** (no subfolders), as required:

```
DiabetesRiskPrediction/
├── app.py                 # Streamlit UI (all pages/sections)
├── model.py                # ML model wrapper + SHAP explainability
├── train_model.py          # Dataset generation + model training/selection
├── csp_engine.py            # Constraint Satisfaction validation engine
├── fuzzy_logic.py            # Fuzzy Inference System
├── reasoning.py               # Forward + backward chaining engines
├── knowledge_base.py            # Rule base + health recommendations
├── database.py                    # SQLite persistence layer
├── utils.py                        # BMI calc, PDF export, fairness analysis
├── config.py                        # Central configuration constants
├── test_app.py                       # Pytest test suite
├── dataset.csv                        # Generated synthetic dataset
├── diabetes_model.pkl                  # Saved best ML model
├── scaler.pkl                           # Saved StandardScaler
├── model_metrics.json                    # Saved evaluation metrics
├── temp_database.db                       # SQLite database (created at runtime)
├── requirements.txt                        # Python dependencies
├── LICENSE                                  # MIT License
└── README.md                                 # This file
```

---

## Feature Descriptions

| Field | Description |
|---|---|
| Age | Patient age in years |
| Gender | Female / Male / Other |
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mmHg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/mL) |
| BMI | Body Mass Index (kg/m²), auto-calculated from weight/height |
| DiabetesPedigreeFunction | Genetic predisposition score |
| PhysicalActivity | Self-reported activity level (0-10) |
| Smoking | Smoking status (yes/no) |
| FamilyHistory | Family history of diabetes (yes/no) |
| Weight / Height | Used to auto-calculate BMI |

---

## Algorithms Used

- **Random Forest Classifier** — ensemble of decision trees, robust to overfitting.
- **Logistic Regression** — interpretable linear baseline.
- **Decision Tree Classifier** — simple, interpretable tree model.
- **Mamdani Fuzzy Inference** — triangular membership functions + centroid defuzzification.
- **Forward/Backward Chaining** — classical rule-based inference algorithms.
- **SHAP (TreeExplainer / LinearExplainer)** — game-theoretic feature attribution.

---

## Model Evaluation

The **Model Performance** page displays, for all three trained models:
- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion Matrix (heatmap)
- ROC Curve
- Side-by-side bar chart comparison

The model with the highest F1-score on the held-out test set is automatically selected for deployment.

---

## Sample Predictions

| Profile | Glucose | BMI | Age | Activity | Predicted Risk |
|---|---|---|---|---|---|
| Young, healthy | 90 | 21 | 25 | 8 | Low Risk |
| Middle-aged, moderate BMI | 130 | 27 | 42 | 5 | Moderate Risk |
| Older, obese, sedentary, family history | 175 | 34.5 | 50 | 2 | High Risk |

---

## Future Enhancements

- Integrate a real, ethically-sourced clinical dataset with IRB approval.
- Add multi-language support for global accessibility.
- Deploy behind proper authentication for multi-user clinical pilots.
- Add time-series tracking of a patient's risk trend over multiple visits.
- Expand fairness analysis with formal metrics (demographic parity, equalized odds, calibration by group).
- Add model monitoring for data/concept drift over time.

---

## Limitations

- Trained on a **synthetic dataset**, not real patient records — not clinically validated.
- Gender-based fairness analysis uses **simulated** gender labels since the dataset lacks a real gender field.
- The fuzzy rule base and CSP domains are illustrative, based on commonly cited clinical thresholds, not a substitute for professional medical guidelines.
- No authentication or encryption — not suitable for storing real patient data.
- SHAP explanations describe correlation captured by the model, not proven causal relationships.

---

## Resume Highlights

- Designed and implemented a full-stack, explainable AI health-screening system combining classical AI (CSP, Fuzzy Logic, Rule-Based Reasoning) with modern ML and SHAP-based Explainable AI.
- Built a Constraint Satisfaction engine for real-time input validation and a Mamdani Fuzzy Inference System for uncertainty-aware risk scoring.
- Implemented forward- and backward-chaining inference engines over an explicit rule base for transparent, human-readable decision explanations.
- Trained, evaluated, and automatically selected among three ML classifiers (Random Forest, Logistic Regression, Decision Tree) using Accuracy/Precision/Recall/F1/ROC-AUC.
- Delivered a polished, interactive Streamlit application with PDF report export, SQLite persistence, and a responsible-AI fairness dashboard.

---

## Interview Questions and Answers

**1. What is a Constraint Satisfaction Problem (CSP), and how did you use it?**
A CSP consists of variables, their domains, and constraints restricting valid value combinations. I used it to validate every patient input field against a realistic numeric domain (e.g. Age ∈ [1,120]) before the data reaches the ML model, preventing invalid predictions from unrealistic input.

**2. Why use Fuzzy Logic instead of simple threshold-based rules?**
Fuzzy logic allows values to partially belong to multiple categories (e.g. glucose can be simultaneously somewhat "normal" and somewhat "high"), which better models the gradual, uncertain nature of medical risk factors compared to brittle hard cut-offs.

**3. What is the difference between forward chaining and backward chaining?**
Forward chaining is data-driven: it starts from known facts and applies rules until no new facts can be derived. Backward chaining is goal-driven: it starts from a target conclusion and works backward to see if it can be proven from known facts.

**4. How does your knowledge representation work?**
Medical knowledge is stored as explicit `Rule` objects (IF-THEN pairs) in Python dataclasses, keeping the logic transparent and easy to extend, rather than hard-coded inside the ML model.

**5. What is SHAP, and why did you use it?**
SHAP (SHapley Additive exPlanations) is a game-theoretic method that attributes a prediction's outcome to individual input features. I used it to generate both local (per-patient) and global feature-importance explanations for the ML model.

**6. How did you choose the best ML model?**
I trained Random Forest, Logistic Regression, and Decision Tree classifiers, evaluated them on a held-out test set using Accuracy, Precision, Recall, F1-score, and ROC-AUC, and automatically selected the model with the highest F1-score.

**7. Why did you generate a synthetic dataset instead of using a real one directly?**
To keep the project fully offline, reproducible, and free of any real patient privacy concerns, while still needing lifestyle fields (physical activity, smoking, family history) not present in the classic Pima dataset. The synthetic data's statistical properties are calibrated to match real-world clinical data distributions.

**8. What does "Reasoning under Uncertainty" mean in this project?**
It refers to the fuzzy inference system, which explicitly models the imprecision of medical thresholds using membership functions and produces a continuous risk score rather than a brittle binary decision.

**9. How do you ensure the app doesn't just feel like a black box?**
By combining SHAP-based statistical explanations, symbolic forward/backward reasoning traces, an independent fuzzy-logic risk cross-check, and full input validation feedback via CSP — giving the user multiple, complementary views into "why".

**10. What fairness considerations did you build in?**
A dedicated fairness analysis compares model prediction rates across gender and age groups, along with an explicit disclosure of the analysis's limitations (e.g. simulated gender labels).

**11. How is the SQLite database used?**
It stores prediction history (timestamp, risk band, score, confidence, and inputs) locally, supporting view, delete, and clear operations, with no login required, consistent with the "simple local tool" design goal.

**12. What would you improve if this were a production system?**
Use real, IRB-approved clinical data; add authentication and encryption; implement formal fairness metrics; add model monitoring for drift; and obtain regulatory clearance before any real clinical use.

**13. What is the role of the `config.py` file?**
It centralizes constants (file paths, CSP domains, risk thresholds, ML settings) so the rest of the codebase avoids "magic numbers" and remains easy to tune and maintain.

**14. How does the CSP engine handle relational constraints, not just per-variable ones?**
Beyond per-variable domain checks, it cross-validates that the reported BMI is consistent with the reported weight and height (a simple binary/relational constraint), demonstrating multi-variable constraint checking.

**15. What's the difference between "risk score" (ML) and "risk score" (fuzzy)?**
The ML risk score is the classifier's predicted probability of the positive class. The fuzzy risk score is an independent estimate derived purely from symbolic linguistic rules over glucose, BMI, age, and activity — the two are shown side by side to cross-validate each other.

**16. How would you extend the rule base?**
Add new `Rule` objects to `knowledge_base.py` with their `conditions` and `conclusion`; the forward/backward chaining engines will automatically incorporate them without any other code changes.

**17. Why does the app show a medical disclaimer everywhere?**
Because health-related AI tools carry real risk of misuse if users mistake risk screening for diagnosis. Making the disclaimer highly visible on every relevant page is a Responsible AI practice.

**18. What testing did you implement?**
Unit tests (`test_app.py`) covering the ML prediction pipeline, CSP validation logic, fuzzy inference output ranges, and database CRUD operations, all runnable via `pytest`.

**19. How does the PDF report get generated?**
`utils.py::generate_pdf_report` uses the `fpdf2` library to assemble a formatted PDF containing the prediction, risk score, confidence, reasoning trace, recommendations, and date, returned as bytes for direct download in Streamlit.

**20. What's the biggest technical challenge you solved in this project?**
Reconciling different explanation styles: statistical (SHAP additive attributions), symbolic (rule chains), and fuzzy (linguistic membership) into one coherent, non-contradictory user experience, while keeping the codebase simple enough for a single root folder with no subdirectories.

**21. How does backward chaining handle rules with multiple conditions?**
It recursively attempts to prove each condition of a candidate rule; only if *all* conditions are provably true does the rule "fire", and the resulting proof tree records each supporting sub-proof for later display.

**22. Why use Streamlit instead of Flask/Django?**
Streamlit dramatically reduces UI boilerplate for data-science-style applications, letting the project focus on the AI logic rather than front-end engineering, while still producing an interactive, professional-looking dashboard.

---

## Learning Outcomes

By building this project, you will gain hands-on experience with:
- Designing and implementing a CSP engine from first principles.
- Building a fuzzy inference system with real membership functions and rule bases.
- Implementing forward and backward chaining algorithms over a symbolic knowledge base.
- Training, evaluating, and comparing multiple ML classifiers.
- Applying SHAP for both local and global explainability.
- Structuring a Responsible AI application with fairness checks and disclaimers.
- Building a complete, testable, multi-page Streamlit application.

---

## References

- Smith, J.W., et al. "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus." (Pima Indians Diabetes Dataset origin study)
- Lundberg, S.M. & Lee, S.I. "A Unified Approach to Interpreting Model Predictions" (SHAP), NeurIPS 2017.
- Zadeh, L.A. "Fuzzy Sets." Information and Control, 1965.
- Russell, S. & Norvig, P. "Artificial Intelligence: A Modern Approach." (CSP, forward/backward chaining foundations)
- American Diabetes Association (ADA) — general public clinical threshold references used as illustrative guidance only.
- scikit-learn, scikit-fuzzy, SHAP, and Streamlit official documentation.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. It also includes an explicit medical-use disclaimer: this software is for educational and portfolio purposes only and is not a certified medical device.
