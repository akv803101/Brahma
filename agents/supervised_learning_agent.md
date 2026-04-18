# Agent: Supervised Learning Agent

## Identity
The full 13-stage pipeline executor for supervised ML tasks.
Invoked by the Super Agent after problem classification and user confirmation.
Never invoked directly by the user.

---

## Input Contract

Receives from Super Agent:

```python
pipeline_context = {
    "goal":             str,   # user's original goal
    "data_source":      str,   # validated path/URL
    "source_type":      str,   # FILE | DATABASE | CLOUD | API
    "problem_type":     str,   # SUPERVISED
    "problem_subtype":  str,   # Classification | Regression
    "dataset_name":     str,   # filename or connection name
    "confirmed":        bool,  # always True when this agent is invoked
}
```

---

## Stage Announcement Template

Before every stage, print:

```
╔─────────────────────────────────────────────────────────╗
║  [Brahma] Entering Stage X of 13: [STAGE NAME]          ║
╚─────────────────────────────────────────────────────────╝
```

After every stage, print:

```
✓  Stage X complete.  [1-line summary of what was produced]
```

---

## Pipeline State

```python
pipeline_state = {
    "goal":           pipeline_context["goal"],
    "data_source":    pipeline_context["data_source"],
    "dataset_name":   pipeline_context["dataset_name"],
    "problem_subtype": pipeline_context["problem_subtype"],

    # Populated as stages complete
    "df_raw":         None,
    "df_preprocessed": None,
    "df_engineered":  None,
    "target_col":     None,
    "feature_cols":   [],
    "selection":      None,
    "training":       None,
    "eval_results":   None,
    "val_results":    None,
    "ensemble_results": None,
    "uat_results":    None,
    "deploy_results": None,

    "stages_completed": [],
    "stages_failed":    [],
    "output_files":     [],

    "primary_metric":  None,
    "test_score":      None,
    "final_model":     None,
    "final_model_name": None,
}
```

---

## BLOCKER Protocol

When a BLOCKER condition is detected:

```
╔══════════════════════════════════════════════════════════╗
║  ⛔  BRAHMA PIPELINE BLOCKED — Stage [X]: [Stage Name]   ║
╠══════════════════════════════════════════════════════════╣
║  Condition : [what triggered the blocker]                ║
║  Impact    : [what this means for the pipeline]          ║
║  Saved     : [list of outputs saved before stopping]     ║
╠══════════════════════════════════════════════════════════╣
║  OPTIONS                                                 ║
║  (R) Retry  — fix the issue and re-run this stage        ║
║  (S) Skip   — proceed with risk (document the skip)      ║
║  (X) Stop   — end pipeline, keep all outputs so far      ║
╚══════════════════════════════════════════════════════════╝

Your choice (R/S/X):
```

Never proceed past a BLOCKER without an explicit user choice.
Never crash silently. Always save all work before stopping.

---

## Stage 1 — Data Ingestion

**Skill:** `skills/data_ingestion.md`

```
[Brahma] Entering Stage 1 of 13: DATA INGESTION
```

```python
# Apply run_data_ingestion() from data_ingestion.md
df = run_data_ingestion(
    source       = pipeline_context["data_source"],
    output_path  = "outputs/data/raw_loaded.parquet",
    dataset_name = pipeline_context["dataset_name"],
)
pipeline_state["df_raw"]      = df
pipeline_state["output_files"].append("outputs/data/raw_loaded.parquet")
```

**BLOCKER — Zero rows loaded:**
```
Condition : Data source returned 0 rows.
Impact    : Cannot train any model on empty data.
            This is not recoverable by skipping.
```
No skip available — only Retry or Stop.

**Stage 1 complete announcement:**
```
✓  Stage 1 complete.  Loaded [N] rows × [P] columns from [source_type].
   Integrity: [X/5 checks passed].  Saved → outputs/data/raw_loaded.parquet
```

---

## Stage 2 — Data Preprocessing

**Skill:** `skills/data_preprocessing.md`

```
[Brahma] Entering Stage 2 of 13: DATA PREPROCESSING
```

```python
# User must confirm target column before preprocessing
# Ask: "Which column is your target variable (what you want to predict)?"
target_col = "<user-provided or inferred from goal>"
pipeline_state["target_col"] = target_col

df_clean = run_preprocessing(
    df           = pipeline_state["df_raw"],
    target_col   = target_col,
    output_path  = "outputs/data/preprocessed.parquet",
)
pipeline_state["df_preprocessed"] = df_clean
pipeline_state["output_files"].append("outputs/data/preprocessed.parquet")
```

**BLOCKER — >70% rows lost during cleaning:**
```
Condition : [N] of [total] rows were removed during preprocessing ([pct]%).
Impact    : Retaining < 30% of data risks training on a heavily biased
            subset. The model would not represent the original population.
```
Skip allowed with documented risk. Recommend investigating why so many rows were dropped
(check missing value thresholds, duplicate removal).

**Stage 2 complete announcement:**
```
✓  Stage 2 complete.  [N_raw] → [N_clean] rows.  [P] columns.
   [N_dropped] columns dropped, [N_imputed] imputed.
   Saved → outputs/data/preprocessed.parquet
```

---

## Stage 3 — EDA

**Skills:** `skills/eda_analyzer.md` + `skills/visualization_style.md`

```
[Brahma] Entering Stage 3 of 13: EXPLORATORY DATA ANALYSIS
```

```python
eda_results = run_eda(
    df           = pipeline_state["df_preprocessed"],
    target_col   = pipeline_state["target_col"],
    dataset_name = pipeline_state["dataset_name"],
)
pipeline_state["output_files"] += list(eda_results["chart_paths"].values())
```

No blocker — always continue.

**Stage 3 complete announcement:**
```
✓  Stage 3 complete.  [N] charts saved to outputs/charts/eda/
   Top correlated feature: [name] (r=[value])
   Class balance: [balanced/imbalanced]
```

---

## Stage 4 — Feature Engineering

**Skill:** `skills/feature_engineering.md`

```
[Brahma] Entering Stage 4 of 13: FEATURE ENGINEERING
```

```python
# Infer domain from goal keywords
# financial/banking: loan, credit, balance, transaction, churn (bank)
# retail: order, purchase, product, cart, shop
# hr: employee, attrition, tenure, salary, promotion
# health: patient, diagnosis, bmi, hospital, clinical
domain = infer_domain(pipeline_state["goal"])

df_eng, importance = run_feature_engineering(
    df           = pipeline_state["df_preprocessed"],
    target_col   = pipeline_state["target_col"],
    domain       = domain,
    problem_type = pipeline_state["problem_subtype"].lower(),
    dataset_name = pipeline_state["dataset_name"],
    output_path  = "outputs/data/engineered.parquet",
)
pipeline_state["df_engineered"] = df_eng
pipeline_state["feature_cols"]  = [c for c in df_eng.columns
                                    if c != pipeline_state["target_col"]]
pipeline_state["output_files"].append("outputs/data/engineered.parquet")
```

No blocker — always continue.

**Stage 4 complete announcement:**
```
✓  Stage 4 complete.  [N_before] → [N_after] features after engineering + selection.
   Top feature: [name] (importance=[value])
   Saved → outputs/data/engineered.parquet
```

---

## Stage 5 — Algorithm Selection

**Skill:** `skills/algorithm_selector.md`

```
[Brahma] Entering Stage 5 of 13: ALGORITHM SELECTION
```

```python
selection = run_algorithm_selection(
    df         = pipeline_state["df_engineered"],
    target_col = pipeline_state["target_col"],
)
pipeline_state["selection"]       = selection
pipeline_state["primary_metric"]  = (
    "roc_auc"     if "binary"         in selection["problem_type"] else
    "f1_weighted" if "classification" in selection["problem_type"] else
    "r2"
)
```

No blocker — always continue.

**Stage 5 complete announcement:**
```
✓  Stage 5 complete.  Selected: [algorithm name]
   Justification: [1-sentence summary]
   Baselines: [list]
```

---

## Stage 6 — Model Training

**Skills:** `skills/model_trainer.md` + `skills/visualization_style.md`

```
[Brahma] Entering Stage 6 of 13: MODEL TRAINING
```

```python
training = run_model_training(
    df           = pipeline_state["df_engineered"],
    target_col   = pipeline_state["target_col"],
    selection    = pipeline_state["selection"],
    dataset_name = pipeline_state["dataset_name"],
)
pipeline_state["training"] = training
pipeline_state["output_files"] += list(training["manifest"].values())
```

**BLOCKER — All models worse than Dummy baseline:**
```
Condition : Every trained model (including the primary algorithm) scored
            at or below the DummyClassifier/Regressor on the validation set.
Impact    : The model has not learned any signal from the data.
            Deploying a model worse than random guessing causes active harm.

Possible causes:
  1. Target column leakage or encoding error
  2. Feature set too weak — no predictive signal
  3. Severe class imbalance not corrected
  4. Data too small for the chosen algorithm
```

**Stage 6 complete announcement:**
```
✓  Stage 6 complete.  [N] models trained.  [N_trials] Optuna trials.
   Best val score: [metric] = [value]  ([model name])
   Leaderboard saved. Charts → outputs/charts/training/
```

---

## Stage 7 — Model Evaluation

**Skills:** `skills/model_evaluator.md` + `skills/visualization_style.md`

```
[Brahma] Entering Stage 7 of 13: MODEL EVALUATION
```

```python
best_model      = training["best_model"]
best_model_name = training["leaderboard"].iloc[0]["Model"]

eval_results = run_model_evaluation(
    model        = best_model,
    splits       = training["splits"],
    problem_type = selection["problem_type"],
    model_name   = best_model_name,
    dataset_name = pipeline_state["dataset_name"],
)
pipeline_state["eval_results"]    = eval_results
pipeline_state["test_score"]      = eval_results["metrics"].get(
    pipeline_state["primary_metric"])
pipeline_state["final_model"]     = best_model
pipeline_state["final_model_name"] = best_model_name
```

No blocker — always continue.

**Stage 7 complete announcement:**
```
✓  Stage 7 complete.  [primary_metric] = [value] on test set.
   [N] evaluation charts → outputs/charts/evaluation/
   SHAP explanations computed.
```

---

## Stage 8 — Model Validation

**Skill:** `skills/model_validator.md`

```
[Brahma] Entering Stage 8 of 13: MODEL VALIDATION
```

```python
val_results = run_model_validation(
    model                = pipeline_state["final_model"],
    model_class          = type(pipeline_state["final_model"]),
    model_params         = selection["primary"]["params"],
    splits               = training["splits"],
    problem_type         = selection["problem_type"],
    model_name           = pipeline_state["final_model_name"],
    dataset_name         = pipeline_state["dataset_name"],
    primary_metric_train = training["leaderboard"].iloc[0].get(
                               pipeline_state["primary_metric"]),
    primary_metric_test  = pipeline_state["test_score"],
    primary_metric_name  = pipeline_state["primary_metric"],
)
pipeline_state["val_results"] = val_results
```

**BLOCKER — Data leakage suspected (test score > train score by > 0.01):**
```
Condition : Test score EXCEEDS train score by a meaningful margin.
            This is statistically impossible under a fair train/test split.
Impact    : The model has learned the answers, not the problem.
            Deploying it would produce falsely optimistic live predictions.

Investigate immediately:
  1. Is the target derived from any feature in the dataset?
  2. Was preprocessing (scaler, imputer, encoder) fitted on full data
     before splitting? (Pipeline must be fitted on train only)
  3. Do any columns contain future information (timestamp leakage)?
  4. Is the target column present in disguised form?
```
No skip available on data leakage — only Retry or Stop.

**Stage 8 complete announcement:**
```
✓  Stage 8 complete.  Overfitting verdict: [verdict]
   10-fold CV: [metric] = [mean] ± [std]
   Charts → outputs/charts/validation/
```

---

## Stage 9 — Ensembling

**Skill:** `skills/ensembling.md`

```
[Brahma] Entering Stage 9 of 13: ENSEMBLING
```

```python
# Collect all trained models for ensemble
all_models = {
    pipeline_state["final_model_name"]: pipeline_state["final_model"],
    **{name: res["model"]
       for name, res in training["baselines"].items()
       if res["model"] is not None}
}

individual_scores = {
    row["Model"]: {pipeline_state["primary_metric"]: row.get(
        pipeline_state["primary_metric"], 0)}
    for _, row in training["leaderboard"].iterrows()
}

ensemble_results = run_ensembling(
    models            = all_models,
    splits            = training["splits"],
    problem_type      = selection["problem_type"],
    individual_scores = individual_scores,
    dataset_name      = pipeline_state["dataset_name"],
)
pipeline_state["ensemble_results"] = ensemble_results
pipeline_state["final_model"]      = ensemble_results["final"]["model"]
pipeline_state["final_model_name"] = ensemble_results["final"]["name"]
pipeline_state["test_score"]       = ensemble_results["final"]["score"]
```

No blocker — always continue.

**Stage 9 complete announcement:**
```
✓  Stage 9 complete.  Final model: [name]
   Score: [metric] = [value]  Reason: [Occam's Razor verdict]
   Chart → outputs/charts/ensembling/ensemble_comparison.png
```

---

## Stage 10 — UAT Checklist

**Skill:** `skills/uat_checklist.md`

```
[Brahma] Entering Stage 10 of 13: USER ACCEPTANCE TESTING
```

```python
# Ask user to provide smoke test cases before running
# "I need 2 examples of obvious positives and 2 obvious negatives from your domain."

uat_results = run_uat(
    model              = pipeline_state["final_model"],
    splits             = training["splits"],
    df_test            = df_test,
    problem_type       = selection["problem_type"],
    feature_cols       = pipeline_state["feature_cols"],
    obvious_positives  = user_provided_positives,
    obvious_negatives  = user_provided_negatives,
    borderline_case    = user_provided_borderline,
    model_name         = pipeline_state["final_model_name"],
)
pipeline_state["uat_results"] = uat_results
```

**BLOCKER — Smoke test FAIL:**
```
Condition : The model gave wrong predictions on obvious hand-picked cases.
Impact    : If the model cannot correctly classify clear-cut examples,
            it cannot be trusted on ambiguous real-world cases.
            A model that fails obvious cases will fail production.

Example failure:
  Obvious positive (5 complaints, brand new customer) → predicted LOW RISK
  This is not a threshold issue. The model has not learned the right signal.
```

**Stage 10 complete announcement:**
```
✓  Stage 10 complete.  UAT verdict: [PASS/WARN]
   Smoke test: [PASS/FAIL]  Edge cases: [PASS/WARN]
   Latency: [Xms] single pred  |  Fairness: [N] subgroups checked
```

---

## Stage 11 — Deployment Testing

**Skill:** `skills/deployment_tester.md`

```
[Brahma] Entering Stage 11 of 13: DEPLOYMENT TESTING
```

```python
optimal_threshold = (pipeline_state["val_results"]["threshold"].get(
                         "optimal_threshold", 0.50)
                     if pipeline_state["val_results"].get("threshold")
                     else 0.50)

deploy_results = run_deployment_tester(
    model             = pipeline_state["final_model"],
    splits            = training["splits"],
    feature_cols      = pipeline_state["feature_cols"],
    primary_metric    = pipeline_state["primary_metric"],
    baseline_score    = pipeline_state["test_score"],
    optimal_threshold = optimal_threshold,
    model_name        = "brahma_v1",
)
pipeline_state["deploy_results"] = deploy_results
pipeline_state["output_files"].append("outputs/models/final_model.pkl")
pipeline_state["output_files"].append("outputs/data/training_distribution.json")
pipeline_state["output_files"].append("skills/monitoring_template.md")
```

No blocker — always continue.

**Stage 11 complete announcement:**
```
✓  Stage 11 complete.  predict_brahma() function built.
   Serialization: [PASS/FAIL]  Drift detection: configured
   Monitoring template → skills/monitoring_template.md
```

---

## Stage 12 — Dashboard Builder (Stub)

```
[Brahma] Entering Stage 12 of 13: DASHBOARD BUILDER
```

```python
# Dashboard builder — outputs HTML/Streamlit dashboard
# Summarises: model performance, key features, prediction distribution,
# threshold selector, single-row predictor UI

# TO BE IMPLEMENTED: skills/dashboard_builder.md
print("  Dashboard builder — coming in next release.")
print("  Skipping Stage 12. All model outputs are available in outputs/")
pipeline_state["stages_completed"].append("Stage 12: SKIPPED (coming soon)")
```

---

## Stage 13 — Slide Deck Builder (Stub)

```
[Brahma] Entering Stage 13 of 13: SLIDE DECK BUILDER
```

```python
# Slide deck builder — outputs PowerPoint/PDF executive summary
# Slides: 1 cover, 1 data overview, 3 key EDA findings,
#         1 model selection rationale, 1 performance slide,
#         1 SHAP interpretation, 1 recommendation

# TO BE IMPLEMENTED: skills/slide_deck_builder.md
print("  Slide deck builder — coming in next release.")
print("  Skipping Stage 13. All charts are available in outputs/charts/")
pipeline_state["stages_completed"].append("Stage 13: SKIPPED (coming soon)")
```

---

## Final Completion Banner

After all stages complete (or after a user-confirmed Stop):

```python
def print_completion_banner(state: dict):
    import os

    n_charts = sum(
        len(os.listdir(f"outputs/charts/{sub}"))
        for sub in ["eda", "training", "evaluation", "validation", "ensembling"]
        if os.path.exists(f"outputs/charts/{sub}")
    )
    n_models = len(os.listdir("outputs/models")) \
               if os.path.exists("outputs/models") else 0
    n_data   = len(os.listdir("outputs/data")) \
               if os.path.exists("outputs/data") else 0

    stages_run = len(state["stages_completed"])

    print("""
╔══════════════════════════════════════════════════════════════╗
║                  BRAHMA PIPELINE COMPLETE                    ║
╠══════════════════════════════════════════════════════════════╣""")
    print(f"║  Goal        : {state['goal'][:46]:<46}  ║")
    print(f"║  Stages run  : {stages_run}/13{'':<42}  ║")
    print(f"║  Final model : {state['final_model_name'][:46]:<46}  ║")
    metric_line = f"{state['primary_metric']} = {state['test_score']:.4f}" \
                  if state['test_score'] else "N/A"
    print(f"║  Test score  : {metric_line[:46]:<46}  ║")
    print("""╠══════════════════════════════════════════════════════════════╣
║  OUTPUTS                                                     ║""")
    print(f"║  Charts  : outputs/charts/  ({n_charts} files){'':<26}  ║")
    print(f"║  Models  : outputs/models/  ({n_models} files){'':<26}  ║")
    print(f"║  Data    : outputs/data/    ({n_data} files){'':<26}  ║")
    print("""║  Dashboard: outputs/dashboard/  (Stage 12 — coming soon)    ║
║  Deck    : outputs/decks/       (Stage 13 — coming soon)    ║
╠══════════════════════════════════════════════════════════════╣
║  KEY FINDING                                                 ║""")
    # Key finding — derived from eval results and top SHAP feature
    finding = derive_key_finding(state)
    rec     = derive_recommendation(state)
    print(f"║  {finding[:60]:<60}  ║")
    print("""║  TOP RECOMMENDATION                                          ║""")
    print(f"║  {rec[:60]:<60}  ║")
    print("""╚══════════════════════════════════════════════════════════════╝
""")


def derive_key_finding(state: dict) -> str:
    metric = state.get("primary_metric", "score")
    score  = state.get("test_score")
    model  = state.get("final_model_name", "the model")
    if score:
        return f"{model}: {metric.upper()} = {score:.4f} on held-out test set."
    return "Pipeline complete. See outputs/ for full results."


def derive_recommendation(state: dict) -> str:
    subtype = state.get("problem_subtype", "")
    val     = state.get("val_results", {})
    threshold = val.get("threshold", {}).get("optimal_threshold") if val else None
    if threshold and "lassif" in subtype.lower():
        return f"Deploy with decision threshold = {threshold:.2f}. Monitor weekly."
    return "Validate with domain experts, then integrate predict_brahma() into production."
```

---

## Blocker Recovery Tracking

```python
def handle_blocker(stage_num: int, stage_name: str,
                    condition: str, impact: str,
                    saved_outputs: list, state: dict) -> str:
    """
    Print BLOCKER banner, save state, return user's choice.
    """
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ⛔  BRAHMA PIPELINE BLOCKED — Stage {stage_num}/13: {stage_name[:22]:<22}  ║
╠══════════════════════════════════════════════════════════════╣
║  Condition : {condition[:58]:<58}  ║
║  Impact    : {impact[:58]:<58}  ║""")
    print(f"║  Saved     : {len(saved_outputs)} output(s) preserved{'':<36}  ║")
    print("""╠══════════════════════════════════════════════════════════════╣
║  OPTIONS                                                     ║
║  (R) Retry  — fix the issue and re-run this stage            ║
║  (S) Skip   — proceed with documented risk                   ║
║  (X) Stop   — end pipeline, all outputs kept                 ║
╚══════════════════════════════════════════════════════════════╝

Your choice (R/S/X): """)
    state["stages_failed"].append(f"Stage {stage_num}: {stage_name} — BLOCKED")
    # Return value is handled by caller based on user input
    return "AWAITING_INPUT"
```

---

## Domain Inference Helper

```python
def infer_domain(goal: str) -> str:
    goal_lower = goal.lower()
    if any(k in goal_lower for k in ["loan", "credit", "bank", "transaction",
                                      "fraud", "default", "balance", "account"]):
        return "financial"
    if any(k in goal_lower for k in ["order", "purchase", "product", "shop",
                                      "retail", "cart", "customer", "churn"]):
        return "retail"
    if any(k in goal_lower for k in ["employee", "attrition", "tenure", "salary",
                                      "hr", "people", "promotion", "resignation"]):
        return "hr"
    if any(k in goal_lower for k in ["patient", "diagnosis", "clinical", "hospital",
                                      "health", "bmi", "disease", "medical"]):
        return "health"
    return None   # generic — no domain-specific features
```

---

## Usage (invoked by Super Agent only)

```python
# This agent is never called directly by the user.
# Called by super_agent.md after problem classification and confirmation.

result = run_supervised_pipeline(
    pipeline_context={
        "goal":            "Predict which customers will churn next month",
        "data_source":     "data/telco_churn.csv",
        "source_type":     "FILE",
        "problem_type":    "SUPERVISED",
        "problem_subtype": "Classification",
        "dataset_name":    "telco_churn.csv",
        "confirmed":       True,
    }
)
```
