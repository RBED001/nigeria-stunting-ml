"""
Explainable Machine Learning Analysis of Child Stunting Risk and Regional
Disparities in Nigeria — full analysis pipeline.

Reproduces the data preparation, exploratory ML/SHAP analysis, confirmatory
interaction modeling, and sensitivity analyses reported in the manuscript.

Requires the Nigeria DHS 2023-24 Children's, Household, and Individual
Recode files (.dta) in a local `data/` folder. See README.md for access.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import chi2
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc,
    brier_score_loss, classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from statsmodels.stats.multitest import multipletests

DATA_PATH = "data/"
FIG_PATH = "figures/"
os.makedirs(FIG_PATH, exist_ok=True)

ZONE_MAP = {1: "North Central", 2: "North East", 3: "North West",
            4: "South East", 5: "South South", 6: "South West"}
ZONE_ORDER = ["North West", "North East", "North Central",
              "South West", "South South", "South East"]
ZONE_LABELS = list(ZONE_MAP.values())

IMPROVED_WATER = [11, 12, 13, 14, 21, 31, 41, 43]
IMPROVED_SANITATION = [11, 12, 13, 14, 15, 21, 22, 41]


# ---------------------------------------------------------------------------
# 1. Data loading and merging (Methods 2.1)
# ---------------------------------------------------------------------------

def load_and_merge():
    kr = pd.read_stata(DATA_PATH + "NGKR8BFL.dta", convert_categoricals=False)
    hr = pd.read_stata(DATA_PATH + "NGHR8BFL.dta", convert_categoricals=False)
    ir = pd.read_stata(DATA_PATH + "NGIR8BFL.dta", convert_categoricals=False)

    kr_small = kr[["caseid", "v001", "v002", "v003", "b4", "hw1", "hw70",
                    "bord", "v024", "v025"]].copy()
    ir_small = ir[[c for c in ["caseid", "v106", "v012", "m14_1", "szone", "v005"]
                   if c in ir.columns]].copy()
    hr_small = hr[[c for c in ["hv001", "hv002", "hv201", "hv205", "hv270",
                                "hv009", "shzone"] if c in hr.columns]].copy()
    del kr, hr, ir

    kr_ir = kr_small.merge(ir_small, on="caseid", how="left")
    merged = kr_ir.merge(hr_small, left_on=["v001", "v002"],
                          right_on=["hv001", "hv002"], how="left")
    return merged


# ---------------------------------------------------------------------------
# 2. Cleaning and outcome/predictor construction (Methods 2.1-2.2)
# ---------------------------------------------------------------------------

def clean_and_build(merged):
    df = merged.copy()
    df["haz"] = df["hw70"] / 100
    clean = df[df["hw70"] < 9990].copy()
    clean["stunted"] = (clean["haz"] < -2).astype(int)
    clean["severe_stunted"] = (clean["haz"] < -3).astype(int)
    clean["zone_name"] = clean["shzone"].map(ZONE_MAP)
    clean["water_improved"] = clean["hv201"].isin(IMPROVED_WATER).astype(int)
    clean["sanitation_improved"] = clean["hv205"].isin(IMPROVED_SANITATION).astype(int)
    clean["dhs_weight"] = clean["v005"] / 1_000_000

    def age_group(m):
        if m < 6: return "0-5"
        if m < 12: return "6-11"
        if m < 24: return "12-23"
        if m < 36: return "24-35"
        if m < 48: return "36-47"
        return "48-59"

    model_df = clean[[
        "caseid", "stunted", "severe_stunted", "v106", "v012", "hv270", "v025",
        "water_improved", "sanitation_improved", "bord", "zone_name",
        "hw1", "b4", "v001", "dhs_weight"
    ]].copy().rename(columns={
        "v106": "maternal_edu_level", "v012": "maternal_age",
        "hv270": "wealth_quintile", "v025": "urban_rural", "bord": "birth_order",
        "hw1": "child_age_months", "b4": "child_sex", "v001": "cluster_id"
    }).dropna()
    model_df["child_age_group"] = model_df["child_age_months"].apply(age_group)

    print(f"Analytic sample: n={len(model_df)}, "
          f"stunting prevalence={model_df['stunted'].mean()*100:.1f}%")
    return clean, model_df


# ---------------------------------------------------------------------------
# 3. Predictive modeling: default, then tuned (Methods 2.3, Results 3.2)
# ---------------------------------------------------------------------------

def build_features(model_df):
    X = pd.get_dummies(
        model_df[["maternal_edu_level", "maternal_age", "wealth_quintile",
                   "urban_rural", "water_improved", "sanitation_improved",
                   "birth_order", "child_sex", "child_age_group"]],
        columns=["child_age_group"], drop_first=True
    )
    X["child_sex"] = X["child_sex"].map({1: 0, 2: 1})
    y = model_df["stunted"]
    return X, y


def train_default_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42,
                                  scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()),
    }
    fitted, roc_data, pr_data = {}, {}, {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        probs = m.predict_proba(X_test)[:, 1]
        fitted[name] = m
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, probs))
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_data[name] = (prec, rec, auc(rec, prec))
        print(f"{name}: AUC={roc_data[name][2]:.3f}, PR-AUC={pr_data[name][2]:.3f}")
    return fitted, roc_data, pr_data


def bootstrap_auc_ci(fitted, X_test, y_test, n_boot=1000):
    rng = np.random.RandomState(42)
    results = {}
    for name, model in fitted.items():
        probs = model.predict_proba(X_test)[:, 1]
        scores = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(probs), len(probs))
            if len(np.unique(y_test.values[idx])) < 2:
                continue
            scores.append(roc_auc_score(y_test.values[idx], probs[idx]))
        results[name] = (np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5))
        print(f"{name}: AUC={results[name][0]:.3f}, "
              f"95% CI=[{results[name][1]:.3f}, {results[name][2]:.3f}]")
    return results


def tune_ensembles(X_train, y_train, X_test, y_test):
    rf_grid = {"max_depth": [5, 10, 15], "n_estimators": [100, 300, 500]}
    rf_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        rf_grid, scoring="roc_auc", cv=5, n_jobs=-1
    ).fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf_search.best_estimator_.predict_proba(X_test)[:, 1])

    xgb_grid = {"max_depth": [3, 6, 9], "learning_rate": [0.01, 0.1, 0.3],
                "n_estimators": [100, 300, 500]}
    xgb_search = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42,
                       scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()),
        xgb_grid, scoring="roc_auc", cv=5, n_jobs=-1
    ).fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb_search.best_estimator_.predict_proba(X_test)[:, 1])

    print(f"Tuned RF AUC: {rf_auc:.3f} ({rf_search.best_params_})")
    print(f"Tuned XGB AUC: {xgb_auc:.3f} ({xgb_search.best_params_})")
    return rf_search, xgb_search


# ---------------------------------------------------------------------------
# 4. Calibration and Brier score (Results 3.2)
# ---------------------------------------------------------------------------

def calibration_and_brier(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, probs)
    naive_brier = brier_score_loss(y_test, np.full(len(y_test), y_test.mean()))
    print(f"Brier score: {brier:.3f} (naive baseline: {naive_brier:.3f})")

    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.title("Calibration Curve — Logistic Regression")
    plt.tight_layout()
    plt.savefig(FIG_PATH + "calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    return brier


# ---------------------------------------------------------------------------
# 5. SHAP explainability (Methods 2.4, Results 3.3)
# ---------------------------------------------------------------------------

def run_shap(model, X_train, X_test, model_df, test_idx):
    X_train_f = X_train.astype(float)
    X_test_f = X_test.astype(float)
    masker = shap.maskers.Independent(X_train_f, max_samples=X_train_f.shape[0])
    explainer = shap.LinearExplainer(model, masker)
    shap_values = explainer(X_test_f)

    shap.summary_plot(shap_values, X_test_f, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIG_PATH + "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_test_f, show=False)
    plt.tight_layout()
    plt.savefig(FIG_PATH + "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Zone-stratified SHAP (exploratory)
    shap_df = pd.DataFrame(shap_values.values, columns=X_test_f.columns, index=X_test_f.index)
    shap_df["zone_name"] = model_df.loc[test_idx, "zone_name"].values
    predictor_cols = [c for c in X_test_f.columns if not c.startswith("child_age_group")]
    zone_shap = shap_df.groupby("zone_name")[predictor_cols].apply(lambda g: g.abs().mean())
    zone_shap = zone_shap.reindex(ZONE_ORDER)
    return shap_values, zone_shap


# ---------------------------------------------------------------------------
# 6. Confirmatory interaction model (Methods 2.4, Results 3.5)
# ---------------------------------------------------------------------------

def confirmatory_interaction_model(model_df):
    df = model_df.copy()
    for col in ["maternal_edu_level", "maternal_age", "wealth_quintile", "birth_order"]:
        df[col] = StandardScaler().fit_transform(df[[col]])
    df["child_sex"] = df["child_sex"].map({1: 0, 2: 1})
    df["zone_name"] = pd.Categorical(
        df["zone_name"], categories=["North Central"] + [z for z in ZONE_LABELS if z != "North Central"]
    )
    df["child_age_group"] = pd.Categorical(
        df["child_age_group"], categories=["0-5", "6-11", "12-23", "24-35", "36-47", "48-59"]
    )

    formula = ("stunted ~ wealth_quintile * zone_name + maternal_edu_level * zone_name + "
               "maternal_age * zone_name + urban_rural + water_improved + sanitation_improved + "
               'birth_order + C(child_age_group, Treatment(reference="0-5")) + child_sex')

    model = smf.logit(formula=formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": model_df["cluster_id"]}
    )
    print(model.summary())

    # Benjamini-Hochberg correction across the 15 interaction terms
    interaction_terms = [t for t in model.pvalues.index if ":" in t]
    reject, p_corrected, _, _ = multipletests(model.pvalues[interaction_terms], alpha=0.05, method="fdr_bh")
    correction_df = pd.DataFrame({
        "term": interaction_terms, "raw_p": model.pvalues[interaction_terms].values,
        "corrected_p": p_corrected, "significant": reject
    }).sort_values("raw_p")
    print(correction_df.to_string(index=False))

    return model, df, correction_df


def age_zone_omnibus_test(model_df, interaction_df, base_model):
    formula_full = ("stunted ~ wealth_quintile * zone_name + maternal_edu_level * zone_name + "
                     "maternal_age * zone_name + urban_rural + water_improved + sanitation_improved + "
                     'birth_order + C(child_age_group, Treatment(reference="0-5")) * zone_name + child_sex')
    full_model = smf.logit(formula=formula_full, data=interaction_df).fit(maxiter=300)
    lr_stat = 2 * (full_model.llf - base_model.llf)
    df_diff = full_model.df_model - base_model.df_model
    p_value = chi2.sf(lr_stat, df_diff)
    print(f"Age x Zone omnibus LR test: stat={lr_stat:.3f}, df={df_diff}, p={p_value:.4f}")
    return full_model, lr_stat, df_diff, p_value


# ---------------------------------------------------------------------------
# 7. Sensitivity analyses (Methods 2.5, Table IV)
# ---------------------------------------------------------------------------

def selection_bias_ipw(merged, model_df, interaction_df):
    sel = merged[["caseid", "v106", "hv270", "v025", "v024", "hw70"]].copy()
    sel["was_measured"] = (sel["hw70"] < 9990).astype(int)
    sel = sel.dropna(subset=["v106", "hv270", "v025", "v024"])

    sel_model = smf.logit("was_measured ~ v106 + hv270 + C(v025) + C(v024)", data=sel).fit()
    sel["prob_measured"] = sel_model.predict(sel)
    sel["ipw"] = 1 / sel["prob_measured"]

    ipw_lookup = sel[sel["was_measured"] == 1][["caseid", "ipw"]]
    interaction_df = interaction_df.merge(
        model_df[["caseid"]].assign(idx=model_df.index).set_index("idx"),
        left_index=True, right_index=True, how="left"
    ) if "caseid" not in interaction_df.columns else interaction_df
    interaction_df = interaction_df.merge(ipw_lookup, on="caseid", how="left")

    formula = ("stunted ~ wealth_quintile * zone_name + maternal_edu_level * zone_name + "
               "maternal_age * zone_name + urban_rural + water_improved + sanitation_improved + "
               'birth_order + C(child_age_group, Treatment(reference="0-5")) + child_sex')
    ipw_model = smf.glm(formula=formula, data=interaction_df, family=sm.families.Binomial(),
                         var_weights=interaction_df["ipw"] / interaction_df["ipw"].mean()).fit()
    print("IPW-corrected wealth x South East:",
          ipw_model.params.get("wealth_quintile:zone_name[T.South East]"),
          ipw_model.pvalues.get("wealth_quintile:zone_name[T.South East]"))
    return ipw_model


def exclusion_by_zone_test(merged):
    merged = merged.copy()
    merged["was_measured"] = (merged["hw70"] < 9990).astype(int)
    merged["zone_temp"] = merged["shzone"].map(ZONE_MAP)
    result = merged.groupby("zone_temp")["was_measured"].agg(["mean", "count"])
    result["excluded_pct"] = (1 - result["mean"]) * 100
    print(result[["excluded_pct", "count"]])
    chi2_stat, p, _, _ = stats.chi2_contingency(pd.crosstab(merged["zone_temp"], merged["was_measured"]))
    print(f"Chi-square (exclusion x zone): p={p:.2e}")
    return result, p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    merged = load_and_merge()
    clean, model_df = clean_and_build(merged)
    X, y = build_features(model_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    fitted, roc_data, pr_data = train_default_models(X_train, X_test, y_train, y_test)
    bootstrap_auc_ci(fitted, X_test, y_test)
    tune_ensembles(X_train, y_train, X_test, y_test)

    best_model = fitted["Logistic Regression"]
    calibration_and_brier(best_model, X_test, y_test)
    run_shap(best_model, X_train, X_test, model_df, X_test.index)

    interaction_model, interaction_df, correction_df = confirmatory_interaction_model(model_df)
    age_zone_omnibus_test(model_df, interaction_df, interaction_model)
    exclusion_by_zone_test(merged)

    print("\nPipeline complete. See figures/ for generated plots.")
