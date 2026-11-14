
# ================================================================
# CHILD STUNTING ANALYSIS IN NIGERIA
# Reproducible analysis pipeline
#
# Study: Explainable Machine Learning Analysis of Child Stunting
# Risk and Regional Disparities in Nigeria
#
# Data: 2023-24 Nigeria Demographic and Health Survey
#
# IMPORTANT:
# - DHS microdata must be obtained through authorised DHS access.
# - This script does not redistribute DHS data.
# - The primary analytical unit is the child.
# - The original notebook's reported results are historical
#   reference values, not values to force into the reconstructed
#   analysis.
# ================================================================


# ================================================================
# 0. IMPORTS AND CONFIGURATION
# ================================================================

from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)

from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

import shap


warnings.filterwarnings("ignore")

SEED = 42
BOOTSTRAP_ITERATIONS = 1000

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "outputs"

FIG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)

# DHS regional codes
ZONE_MAP = {
    1: "North Central",
    2: "North East",
    3: "North West",
    4: "South East",
    5: "South South",
    6: "South West"
}

ZONE_ORDER = list(ZONE_MAP.values())

# DHS improved water and sanitation classifications
IMPROVED_WATER = [11, 12, 13, 14, 21, 31, 41, 43]
IMPROVED_SANITATION = [11, 12, 13, 14, 15, 21, 22, 41]

AGE_LABELS = [
    "0-5", "6-11", "12-23",
    "24-35", "36-47", "48-59"
]


# ================================================================
# 1. DATA LOADING AND MERGING
# ================================================================

def load_data():

    # Read only the variables required by the study.
    # KR = Children's Recode
    # IR = Individual Women's Recode
    # HR = Household Recode

    kr = pd.read_stata(
        DATA_DIR / "NGKR8BFL.dta",
        convert_categoricals=False
    )

    ir = pd.read_stata(
        DATA_DIR / "NGIR8BFL.dta",
        convert_categoricals=False
    )

    hr = pd.read_stata(
        DATA_DIR / "NGHR8BFL.dta",
        convert_categoricals=False
    )

    kr_cols = [
        "caseid", "v001", "v002", "v003",
        "b4", "hw1", "hw70", "bord",
        "v024", "v025"
    ]

    ir_cols = [
        "caseid", "v106", "v012", "m14_1",
        "szone", "v005"
    ]

    hr_cols = [
        "hv001", "hv002", "hv201", "hv205",
        "hv270", "hv009", "shzone"
    ]

    kr = kr[kr_cols].copy()
    ir = ir[ir_cols].copy()
    hr = hr[hr_cols].copy()

    # The KR file contains child records. Preserve a unique
    # row identifier so every child can be tracked independently.
    kr["child_record_id"] = np.arange(len(kr))

    # A respondent may have several children in KR.
    # Therefore, KR-to-IR is many-to-one.
    data = kr.merge(
        ir,
        on="caseid",
        how="left",
        validate="many_to_one"
    )

    # Several children may belong to the same household.
    # Therefore, the household merge is also many-to-one.
    data = data.merge(
        hr,
        left_on=["v001", "v002"],
        right_on=["hv001", "hv002"],
        how="left",
        validate="many_to_one"
    )

    # Ensure that the merge did not multiply child records.
    assert len(data) == len(kr)
    assert data["child_record_id"].is_unique

    print("Merged dataset:", data.shape)

    return data


# ================================================================
# 2. DATA CLEANING AND VARIABLE CONSTRUCTION
# ================================================================

def prepare_data(data):

    df = data.copy()

    # Height-for-age Z-score is stored in DHS as a value
    # multiplied by 100.
    df["haz"] = df["hw70"] / 100

    # DHS special values >= 9990 represent missing or flagged
    # anthropometric measurements.
    df.loc[df["hw70"] >= 9990, "haz"] = np.nan

    # Primary and secondary outcomes.
    df["stunted"] = np.where(
        df["haz"].notna(),
        (df["haz"] < -2).astype(float),
        np.nan
    )

    df["severe_stunted"] = np.where(
        df["haz"].notna(),
        (df["haz"] < -3).astype(float),
        np.nan
    )

    # Socioeconomic predictors.
    df["maternal_edu_level"] = df["v106"]
    df["maternal_age"] = df["v012"]
    df["wealth_quintile"] = df["hv270"]

    # Household characteristics.
    df["urban_rural"] = df["v025"]

    df["water_improved"] = (
        df["hv201"].isin(IMPROVED_WATER).astype(int)
    )

    df["sanitation_improved"] = (
        df["hv205"].isin(IMPROVED_SANITATION).astype(int)
    )

    # Child characteristics.
    df["birth_order"] = df["bord"]

    df["child_sex"] = df["b4"].map({
        1: 0,
        2: 1
    })

    df["child_age_group"] = pd.cut(
        df["hw1"],
        bins=[0, 6, 12, 24, 36, 48, 60],
        labels=AGE_LABELS,
        right=False
    )

    # Geographic region.
    df["zone_name"] = df["shzone"].map(ZONE_MAP)

    # DHS sampling weight.
    df["dhs_weight"] = df["v005"] / 1_000_000

    # Primary sampling unit / cluster.
    df["cluster_id"] = df["v001"]

    # Analytical variables.
    variables = [
        "child_record_id",
        "caseid",
        "stunted",
        "severe_stunted",
        "maternal_edu_level",
        "maternal_age",
        "wealth_quintile",
        "urban_rural",
        "water_improved",
        "sanitation_improved",
        "birth_order",
        "child_sex",
        "child_age_group",
        "zone_name",
        "dhs_weight",
        "cluster_id"
    ]

    model_df = df[variables].copy()

    # Complete-case analysis, following the original notebook.
    model_df = model_df.dropna(
        subset=[
            "stunted",
            "maternal_edu_level",
            "maternal_age",
            "wealth_quintile",
            "urban_rural",
            "water_improved",
            "sanitation_improved",
            "birth_order",
            "child_sex",
            "child_age_group",
            "zone_name",
            "dhs_weight",
            "cluster_id"
        ]
    ).copy()

    model_df["stunted"] = model_df["stunted"].astype(int)
    model_df["severe_stunted"] = (
        model_df["severe_stunted"].astype(int)
    )

    print("\nInitial records:", len(df))
    print("Analytical sample:", len(model_df))
    print("Stunted:", model_df["stunted"].sum())
    print("Non-stunted:", (model_df["stunted"] == 0).sum())
    print("Severe:", model_df["severe_stunted"].sum())

    model_df.to_csv(
        OUTPUT_DIR / "analytical_dataset.csv",
        index=False
    )

    return df, model_df


# ================================================================
# 3. DESCRIPTIVE STATISTICS
# ================================================================

def descriptive_analysis(df, model_df):

    # Overall unweighted prevalence.
    summary = pd.DataFrame({
        "Measure": [
            "Analytical sample",
            "Stunted children",
            "Non-stunted children",
            "Severe stunting"
        ],
        "Value": [
            len(model_df),
            model_df["stunted"].sum(),
            (model_df["stunted"] == 0).sum(),
            model_df["severe_stunted"].sum()
        ]
    })

    summary.to_csv(
        OUTPUT_DIR / "sample_summary.csv",
        index=False
    )

    # Group-specific prevalence.
    categorical = [
        "zone_name",
        "maternal_edu_level",
        "wealth_quintile",
        "urban_rural",
        "child_sex",
        "child_age_group",
        "water_improved",
        "sanitation_improved"
    ]

    for variable in categorical:

        result = (
            model_df.groupby(variable, observed=False)
            ["stunted"]
            .agg(["count", "sum", "mean"])
            .reset_index()
        )

        result["prevalence_percent"] = result["mean"] * 100

        result.to_csv(
            OUTPUT_DIR / f"descriptive_{variable}.csv",
            index=False
        )

    # Continuous variable summaries.
    continuous = [
        "maternal_age",
        "birth_order",
        "wealth_quintile"
    ]

    continuous_summary = (
        model_df.groupby("stunted")[continuous]
        .agg(["mean", "std", "median"])
    )

    continuous_summary.to_csv(
        OUTPUT_DIR / "continuous_descriptives.csv"
    )

    # Weighted prevalence is reported separately from the
    # unweighted prevalence used in the original descriptive table.
    weighted_prevalence = np.average(
        model_df["stunted"],
        weights=model_df["dhs_weight"]
    )

    with open(
        OUTPUT_DIR / "weighted_prevalence.txt",
        "w"
    ) as f:
        f.write(
            f"Weighted prevalence: "
            f"{weighted_prevalence * 100:.4f}%\n"
        )

    print(
        "Weighted prevalence:",
        round(weighted_prevalence * 100, 2),
        "%"
    )


# ================================================================
# 4. EXPLORATORY VISUALISATION
# ================================================================

def exploratory_figures(model_df):

    sns.set_theme(style="whitegrid")

    # Stunting prevalence by geopolitical zone.
    zone_data = (
        model_df.groupby("zone_name", observed=False)
        ["stunted"]
        .mean()
        .reindex(ZONE_ORDER)
        * 100
    )

    plt.figure(figsize=(10, 6))
    zone_data.plot(kind="bar")
    plt.ylabel("Stunting prevalence (%)")
    plt.xlabel("Geopolitical zone")
    plt.title("Child Stunting Prevalence by Zone")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_zone_stunting.png", dpi=300)
    plt.close()

    # Wealth distribution by geopolitical zone.
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=model_df,
        x="zone_name",
        y="wealth_quintile",
        order=ZONE_ORDER
    )

    plt.title("Household Wealth Distribution by Zone")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig12_wealth_by_zone.png", dpi=300)
    plt.close()

    # Wealth-stunting interaction descriptive visual.
    wealth_zone = (
        model_df.groupby(
            ["zone_name", "wealth_quintile"],
            observed=False
        )["stunted"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=wealth_zone,
        x="wealth_quintile",
        y="stunted",
        hue="zone_name",
        hue_order=ZONE_ORDER,
        marker="o"
    )

    plt.ylabel("Stunting probability")
    plt.title("Wealth and Stunting Across Geopolitical Zones")
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "fig11_wealth_zone_interaction.png",
        dpi=300
    )
    plt.close()


# ================================================================
# 5. MACHINE LEARNING FEATURE ENGINEERING
# ================================================================

def build_features(model_df):

    features = [
        "maternal_edu_level",
        "maternal_age",
        "wealth_quintile",
        "urban_rural",
        "water_improved",
        "sanitation_improved",
        "birth_order",
        "child_sex",
        "child_age_group"
    ]

    X = pd.get_dummies(
        model_df[features],
        columns=["child_age_group"],
        drop_first=True,
        dtype=int
    )

    y = model_df["stunted"]

    return X, y


# ================================================================
# 6. TRAIN/TEST SPLIT AND BASELINE MODELS
# ================================================================

def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED
    )

    # Logistic Regression.
    # Balanced class weights compensate for class imbalance.
    lr = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
        class_weight="balanced"
    )

    # Random Forest.
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1
    )

    # XGBoost.
    scale_pos = (
        (y_train == 0).sum() /
        (y_train == 1).sum()
    )

    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=SEED,
        scale_pos_weight=scale_pos,
        n_jobs=-1
    )

    models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb
    }

    results = []

    predictions = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        probability = model.predict_proba(X_test)[:, 1]
        prediction = model.predict(X_test)

        predictions[name] = {
            "probability": probability,
            "prediction": prediction
        }

        results.append({
            "Model": name,
            "ROC_AUC": roc_auc_score(y_test, probability),
            "PR_AUC": average_precision_score(y_test, probability)
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "baseline_model_results.csv",
        index=False
    )

    print("\nBaseline model results:")
    print(results_df)

    return (
        models,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions
    )


# ================================================================
# 7. CONFUSION MATRIX AND CLASSIFICATION PERFORMANCE
# ================================================================

def classification_evaluation(y_test, predictions):

    rows = []

    for name, result in predictions.items():

        cm = confusion_matrix(
            y_test,
            result["prediction"]
        )

        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "Model": name,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Precision": tp / (tp + fp) if tp + fp else np.nan,
            "Recall": tp / (tp + fn) if tp + fn else np.nan,
            "Specificity": tn / (tn + fp) if tn + fp else np.nan,
            "Accuracy": (tp + tn) / cm.sum()
        })

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        filename = name.lower().replace(" ", "_")
        plt.savefig(
            FIG_DIR / f"confusion_matrix_{filename}.png",
            dpi=300
        )
        plt.close()

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "classification_metrics.csv",
        index=False
    )


# ================================================================
# 8. ROC AND PRECISION-RECALL CURVES
# ================================================================

def plot_roc_pr(y_test, predictions):

    plt.figure(figsize=(8, 6))

    for name, result in predictions.items():

        fpr, tpr, _ = roc_curve(
            y_test,
            result["probability"]
        )

        auc = roc_auc_score(
            y_test,
            result["probability"]
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))

    for name, result in predictions.items():

        precision, recall, _ = precision_recall_curve(
            y_test,
            result["probability"]
        )

        ap = average_precision_score(
            y_test,
            result["probability"]
        )

        plt.plot(
            recall,
            precision,
            label=f"{name} (PR-AUC={ap:.3f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pr_curves.png", dpi=300)
    plt.close()


# ================================================================
# 9. BOOTSTRAP CONFIDENCE INTERVALS FOR ROC-AUC
# ================================================================

def bootstrap_auc(y_test, predictions):

    rng = np.random.default_rng(SEED)

    y_array = np.asarray(y_test)

    results = []

    for name, result in predictions.items():

        probabilities = result["probability"]
        scores = []

        # Bootstrap test observations with replacement.
        # Replicates with only one outcome class are skipped.
        for _ in range(BOOTSTRAP_ITERATIONS):

            indices = rng.choice(
                len(y_array),
                size=len(y_array),
                replace=True
            )

            if len(np.unique(y_array[indices])) < 2:
                continue

            scores.append(
                roc_auc_score(
                    y_array[indices],
                    probabilities[indices]
                )
            )

        results.append({
            "Model": name,
            "Mean_AUC": np.mean(scores),
            "CI_Lower": np.percentile(scores, 2.5),
            "CI_Upper": np.percentile(scores, 97.5)
        })

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "bootstrap_auc.csv",
        index=False
    )


# ================================================================
# 10. STRATIFIED FIVE-FOLD CROSS-VALIDATION
# ================================================================

def cross_validation(X, y):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    pd.DataFrame({
        "Fold": np.arange(1, 6),
        "ROC_AUC": scores
    }).to_csv(
        OUTPUT_DIR / "cross_validation.csv",
        index=False
    )

    print(
        "CV mean:",
        scores.mean(),
        "CV SD:",
        scores.std()
    )


# ================================================================
# 11. HYPERPARAMETER TUNING
# ================================================================

def tune_models(X_train, y_train, X_test, y_test):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    # Random Forest grid.
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )

    rf_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 10, 15]
    }

    rf_search = GridSearchCV(
        rf,
        rf_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    rf_search.fit(X_train, y_train)

    # XGBoost grid.
    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1
    )

    xgb_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.3]
    }

    xgb_search = GridSearchCV(
        xgb,
        xgb_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    xgb_search.fit(X_train, y_train)

    results = []

    for name, search in [
        ("Random Forest", rf_search),
        ("XGBoost", xgb_search)
    ]:

        probability = search.best_estimator_.predict_proba(
            X_test
        )[:, 1]

        results.append({
            "Model": name,
            "Best_Params": str(search.best_params_),
            "CV_AUC": search.best_score_,
            "Test_AUC": roc_auc_score(y_test, probability)
        })

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "tuned_model_results.csv",
        index=False
    )

    return rf_search, xgb_search


# ================================================================
# 12. LOGISTIC REGRESSION CALIBRATION
# ================================================================

def calibration_analysis(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    brier = brier_score_loss(
        y_test,
        probabilities
    )

    baseline = np.full(
        len(y_test),
        y_test.mean()
    )

    baseline_brier = brier_score_loss(
        y_test,
        baseline
    )

    fraction_positive, mean_predicted = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="uniform"
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="Logistic Regression"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction positive")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "calibration_curve.png",
        dpi=300
    )
    plt.close()

    pd.DataFrame([{
        "Model_Brier": brier,
        "Prevalence_Baseline_Brier": baseline_brier
    }]).to_csv(
        OUTPUT_DIR / "calibration_metrics.csv",
        index=False
    )


# ================================================================
# 13. PERMUTATION FEATURE IMPORTANCE
# ================================================================

def permutation_analysis(model, X_test, y_test):

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=SEED,
        scoring="roc_auc",
        n_jobs=-1
    )

    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_SD": result.importances_std
    }).sort_values(
        "Importance_Mean",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "permutation_importance.csv",
        index=False
    )


# ================================================================
# 14. SHAP EXPLAINABILITY
# ================================================================

def shap_analysis(model, X_train, X_test):

    # Logistic Regression is a linear model, so LinearExplainer
    # is appropriate for its feature attribution.
    #
    # The training data provide the background distribution.
    # SHAP values are calculated for the held-out test observations.

    masker = shap.maskers.Independent(
        X_train.astype(float),
        max_samples=X_train.shape[0]
    )

    explainer = shap.LinearExplainer(
        model,
        masker
    )

    shap_values = explainer(
        X_test.astype(float)
    )

    # Global feature importance.
    shap.plots.bar(
        shap_values,
        show=False
    )

    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "fig9a_shap_bar.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Distribution and direction of feature effects.
    shap.plots.beeswarm(
        shap_values,
        show=False
    )

    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "fig9b_shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Save mean absolute SHAP values.
    shap_importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean_Absolute_SHAP": np.abs(
            shap_values.values
        ).mean(axis=0)
    }).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    shap_importance.to_csv(
        OUTPUT_DIR / "shap_importance.csv",
        index=False
    )

    return shap_values


# ================================================================
# 15. REGRESSION DATA PREPARATION
# ================================================================

def prepare_regression_data(model_df):

    reg = model_df.copy()

    # The original confirmatory model standardised these
    # continuous/ordinal predictors using the analytical sample.
    #
    # Standardisation improves numerical stability and means
    # interaction coefficients represent effects per one SD.

    scaler = StandardScaler()

    scaled_columns = [
        "wealth_quintile",
        "maternal_edu_level",
        "maternal_age",
        "birth_order"
    ]

    reg[[
        "wealth_z",
        "education_z",
        "maternal_age_z",
        "birth_order_z"
    ]] = scaler.fit_transform(
        reg[scaled_columns]
    )

    reg["zone_name"] = pd.Categorical(
        reg["zone_name"],
        categories=ZONE_ORDER
    )

    reg["child_age_group"] = pd.Categorical(
        reg["child_age_group"],
        categories=AGE_LABELS
    )

    return reg


# ================================================================
# 16. CONFIRMATORY INTERACTION REGRESSION
# ================================================================

def confirmatory_regression(reg):

    # North Central is the reference geopolitical zone.
    # Children aged 0-5 months are the reference age group.
    #
    # The interaction terms assess whether the associations
    # of wealth, education and maternal age with stunting
    # differ across geopolitical zones.
    #
    # Standard errors are clustered by DHS sampling cluster.
    # This is cluster-robust regression, not a full survey design.

    formula = """
    stunted ~
        wealth_z * C(zone_name, Treatment(reference='North Central'))
        + education_z * C(zone_name, Treatment(reference='North Central'))
        + maternal_age_z * C(zone_name, Treatment(reference='North Central'))
        + urban_rural
        + water_improved
        + sanitation_improved
        + birth_order_z
        + C(child_age_group, Treatment(reference='0-5'))
        + child_sex
    """

    model = smf.logit(
        formula,
        data=reg
    )

    result = model.fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={
            "groups": reg["cluster_id"]
        }
    )

    with open(
        OUTPUT_DIR / "confirmatory_regression.txt",
        "w"
    ) as f:
        f.write(result.summary().as_text())

    # Export coefficient table.
    coefficients = pd.DataFrame({
        "Coefficient": result.params,
        "Std_Error": result.bse,
        "P_Value": result.pvalues,
        "CI_Lower": result.conf_int()[0],
        "CI_Upper": result.conf_int()[1]
    })

    coefficients["Odds_Ratio"] = np.exp(
        coefficients["Coefficient"]
    )

    coefficients["OR_CI_Lower"] = np.exp(
        coefficients["CI_Lower"]
    )

    coefficients["OR_CI_Upper"] = np.exp(
        coefficients["CI_Upper"]
    )

    # Benjamini-Hochberg correction across interaction terms.
    interaction_mask = coefficients.index.str.contains(
        ":",
        regex=False
    )

    coefficients["BH_Adjusted_P"] = np.nan

    coefficients.loc[
        interaction_mask,
        "BH_Adjusted_P"
    ] = multipletests(
        coefficients.loc[
            interaction_mask,
            "P_Value"
        ],
        method="fdr_bh"
    )[1]

    coefficients.to_csv(
        OUTPUT_DIR / "confirmatory_coefficients.csv"
    )

    return result, coefficients


# ================================================================
# 17. AGE-GROUP BY ZONE OMNIBUS TEST
# ================================================================

def age_zone_omnibus(reg):

    # Compare the primary interaction model with an expanded
    # model containing child-age-group-by-zone interactions.
    #
    # This tests whether age-group associations differ jointly
    # across geopolitical zones.
    #
    # The likelihood-ratio test uses ordinary model likelihoods;
    # it is not a cluster-robust survey-design LR test.

    base_formula = """
    stunted ~
        wealth_z * C(zone_name, Treatment(reference='North Central'))
        + education_z * C(zone_name, Treatment(reference='North Central'))
        + maternal_age_z * C(zone_name, Treatment(reference='North Central'))
        + urban_rural
        + water_improved
        + sanitation_improved
        + birth_order_z
        + C(child_age_group, Treatment(reference='0-5'))
        + child_sex
    """

    expanded_formula = base_formula + """
        + C(child_age_group, Treatment(reference='0-5')):
          C(zone_name, Treatment(reference='North Central'))
    """

    base = smf.logit(
        base_formula,
        data=reg
    ).fit(disp=False)

    expanded = smf.logit(
        expanded_formula,
        data=reg
    ).fit(disp=False)

    lr_stat = 2 * (
        expanded.llf - base.llf
    )

    df_diff = (
        expanded.df_model - base.df_model
    )

    p_value = chi2.sf(
        lr_stat,
        df_diff
    )

    pd.DataFrame([{
        "LR_Statistic": lr_stat,
        "Degrees_of_Freedom": df_diff,
        "P_Value": p_value
    }]).to_csv(
        OUTPUT_DIR / "age_zone_omnibus.csv",
        index=False
    )


# ================================================================
# 18. SEVERE STUNTING REGRESSION
# ================================================================

def severe_stunting_regression(reg):

    # The severe-stunting analysis is a secondary outcome analysis.
    # It uses the same predictors and interaction structure as
    # the primary confirmatory model.

    formula = """
    severe_stunted ~
        wealth_z * C(zone_name, Treatment(reference='North Central'))
        + education_z * C(zone_name, Treatment(reference='North Central'))
        + maternal_age_z * C(zone_name, Treatment(reference='North Central'))
        + urban_rural
        + water_improved
        + sanitation_improved
        + birth_order_z
        + C(child_age_group, Treatment(reference='0-5'))
        + child_sex
    """

    result = smf.logit(
        formula,
        data=reg
    ).fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": reg["cluster_id"]}
    )

    with open(
        OUTPUT_DIR / "severe_stunting_regression.txt",
        "w"
    ) as f:
        f.write(result.summary().as_text())


# ================================================================
# 19. SURVEY-WEIGHTED REGRESSION SENSITIVITY ANALYSIS
# ================================================================

def weighted_regression(reg):

    # The original notebook used DHS sampling weights and
    # cluster-robust covariance.
    #
    # This is reproduced as a weighted GLM sensitivity analysis.
    # It does not incorporate the complete DHS complex survey
    # design, particularly explicit strata and full design-based
    # variance estimation.

    formula = """
    stunted ~
        wealth_z * C(zone_name, Treatment(reference='North Central'))
        + education_z * C(zone_name, Treatment(reference='North Central'))
        + maternal_age_z * C(zone_name, Treatment(reference='North Central'))
        + urban_rural
        + water_improved
        + sanitation_improved
        + birth_order_z
        + C(child_age_group, Treatment(reference='0-5'))
        + child_sex
    """

    weights = (
        reg["dhs_weight"] /
        reg["dhs_weight"].mean()
    )

    result = smf.glm(
        formula,
        data=reg,
        family=sm.families.Binomial(),
        var_weights=weights
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": reg["cluster_id"]}
    )

    with open(
        OUTPUT_DIR / "weighted_regression.txt",
        "w"
    ) as f:
        f.write(result.summary().as_text())


# ================================================================
# 20. VIF MULTICOLLINEARITY DIAGNOSTICS
# ================================================================

def vif_analysis(reg):

    # VIF assesses linear dependence among selected predictors.
    # It is a diagnostic, not a test of model validity.

    variables = [
        "wealth_z",
        "education_z",
        "maternal_age_z",
        "urban_rural",
        "water_improved",
        "sanitation_improved",
        "birth_order_z"
    ]

    X = reg[variables].astype(float)
    X = sm.add_constant(X)

    results = []

    for i, variable in enumerate(X.columns):

        if variable == "const":
            continue

        results.append({
            "Variable": variable,
            "VIF": variance_inflation_factor(
                X.values,
                i
            )
        })

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "vif_results.csv",
        index=False
    )


# ================================================================
# 21. SELECTION-BIAS DIAGNOSTICS
# ================================================================

def selection_bias_diagnostics(df, model_df):

    # Compare children with valid anthropometric measurements
    # against those without valid measurements.
    #
    # This evaluates whether complete-case inclusion differs
    # across observed characteristics and regions.

    selection = df.copy()

    selection["was_measured"] = (
        selection["hw70"] < 9990
    ).astype(int)

    selection["zone_name"] = selection["shzone"].map(
        ZONE_MAP
    )

    selection["maternal_edu_level"] = selection["v106"]
    selection["wealth_quintile"] = selection["hv270"]

    selection = selection.dropna(
        subset=[
            "was_measured",
            "maternal_edu_level",
            "wealth_quintile",
            "v025",
            "zone_name"
        ]
    )

    summary = (
        selection.groupby("zone_name")
        ["was_measured"]
        .agg(["count", "mean"])
    )

    summary["exclusion_percent"] = (
        1 - summary["mean"]
    ) * 100

    summary.to_csv(
        OUTPUT_DIR / "selection_by_zone.csv"
    )

    # Chi-square test of measurement status by zone.
    contingency = pd.crosstab(
        selection["zone_name"],
        selection["was_measured"]
    )

    chi_stat, p_value, dof, _ = stats.chi2_contingency(
        contingency
    )

    pd.DataFrame([{
        "Chi_Square": chi_stat,
        "Degrees_of_Freedom": dof,
        "P_Value": p_value
    }]).to_csv(
        OUTPUT_DIR / "selection_zone_chisquare.csv",
        index=False
    )


# ================================================================
# 22. IPW SENSITIVITY ANALYSIS
# ================================================================

def ipw_analysis(df, reg):

    # IMPORTANT:
    # The original notebook merged IPW values to the analytical
    # dataset using maternal caseid alone.
    #
    # Because one mother may have multiple child records, this
    # produced a many-to-many expansion from 9,410 to 15,612 rows.
    # That historical IPW result is not a valid child-level result.
    #
    # Here, we retain the unique child_record_id through the
    # selection model and merge weights using that identifier.
    #
    # The selection model estimates the probability that a child
    # has a valid anthropometric measurement, conditional on
    # observed maternal, household and geographic characteristics.
    #
    # These are unstabilised inverse probability weights.
    # This remains a sensitivity analysis and depends on the
    # selection model and positivity assumptions.

    selection = df[[
        "child_record_id",
        "hw70",
        "v106",
        "hv270",
        "v025",
        "v024"
    ]].copy()

    selection["was_measured"] = (
        selection["hw70"] < 9990
    ).astype(int)

    selection = selection.dropna(
        subset=[
            "v106",
            "hv270",
            "v025",
            "v024"
        ]
    )

    selection_model = smf.logit(
        """
        was_measured ~
        v106 + hv270 + C(v025) + C(v024)
        """,
        data=selection
    ).fit(disp=False)

    selection["prob_measured"] = (
        selection_model.predict(selection)
    )

    # Protect against numerical zero probabilities.
    selection["prob_measured"] = (
        selection["prob_measured"].clip(
            lower=0.001,
            upper=1
        )
    )

    selection["ipw"] = (
        1 / selection["prob_measured"]
    )

    measured_weights = selection.loc[
        selection["was_measured"] == 1,
        ["child_record_id", "ipw"]
    ]

    # Validate the child-level merge.
    assert measured_weights["child_record_id"].is_unique

    ipw_df = reg.merge(
        measured_weights,
        on="child_record_id",
        how="inner",
        validate="one_to_one"
    )

    # Normalise weights to mean one.
    ipw_df["ipw_normalised"] = (
        ipw_df["ipw"] / ipw_df["ipw"].mean()
    )

    # Report weight distribution and effective sample size.
    weight_stats = ipw_df["ipw"].describe()

    effective_n = (
        ipw_df["ipw"].sum() ** 2 /
        (ipw_df["ipw"] ** 2).sum()
    )

    weight_stats.to_csv(
        OUTPUT_DIR / "ipw_weight_summary.csv"
    )

    pd.DataFrame([{
        "Analytical_N": len(ipw_df),
        "Effective_Sample_Size": effective_n
    }]).to_csv(
        OUTPUT_DIR / "ipw_sample_summary.csv",
        index=False
    )

    # Weighted logistic regression.
    formula = """
    stunted ~
        wealth_z * C(zone_name, Treatment(reference='North Central'))
        + education_z * C(zone_name, Treatment(reference='North Central'))
        + maternal_age_z * C(zone_name, Treatment(reference='North Central'))
        + urban_rural
        + water_improved
        + sanitation_improved
        + birth_order_z
        + C(child_age_group, Treatment(reference='0-5'))
        + child_sex
    """

    result = smf.glm(
        formula,
        data=ipw_df,
        family=sm.families.Binomial(),
        freq_weights=ipw_df["ipw_normalised"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": ipw_df["cluster_id"]}
    )

    with open(
        OUTPUT_DIR / "ipw_regression.txt",
        "w"
    ) as f:
        f.write(result.summary().as_text())


# ================================================================
# 23. MISCLASSIFICATION BY GEOGRAPHIC ZONE
# ================================================================

def regional_error_analysis(
    model_df,
    y_test,
    predictions,
    test_indices
):

    # Assess false negatives, false positives and correct
    # classifications separately across geopolitical zones.
    #
    # test_indices preserve the correspondence between the
    # held-out test observations and the original analytical data.

    result = predictions["Logistic Regression"]

    evaluation = model_df.loc[
        test_indices,
        ["zone_name"]
    ].copy()

    evaluation["actual"] = np.asarray(y_test)
    evaluation["predicted"] = result["prediction"]

    rows = []

    for zone, group in evaluation.groupby("zone_name"):

        tn, fp, fn, tp = confusion_matrix(
            group["actual"],
            group["predicted"],
            labels=[0, 1]
        ).ravel()

        total = len(group)

        rows.append({
            "Zone": zone,
            "Correct_Percent": (
                (tn + tp) / total * 100
            ),
            "False_Negative_Percent": (
                fn / total * 100
            ),
            "False_Positive_Percent": (
                fp / total * 100
            )
        })

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "regional_error_analysis.csv",
        index=False
    )


# ================================================================
# 24. MAIN EXECUTION
# ================================================================

def main():

    print("\nSTARTING CHILD STUNTING ANALYSIS")

    # Stage 1: Load and merge DHS files.
    data = load_data()

    # Stage 2: Clean data and construct outcomes/predictors.
    df, model_df = prepare_data(data)

    # Stage 3: Descriptive statistics.
    descriptive_analysis(df, model_df)

    # Stage 4: Exploratory figures.
    exploratory_figures(model_df)

    # Stage 5: Build ML feature matrix.
    X, y = build_features(model_df)

    # Stage 6: Split and train baseline models.
    (
        models,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions
    ) = train_models(X, y)

    # Preserve original test row identifiers for regional analysis.
    test_indices = X_test.index

    # Stage 7: Classification evaluation.
    classification_evaluation(y_test, predictions)

    # Stage 8: ROC and PR curves.
    plot_roc_pr(y_test, predictions)

    # Stage 9: Bootstrap confidence intervals.
    bootstrap_auc(y_test, predictions)

    # Stage 10: Cross-validation.
    cross_validation(X, y)

    # Stage 11: Hyperparameter tuning.
    rf_search, xgb_search = tune_models(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # Stage 12: Calibration of Logistic Regression.
    calibration_analysis(
        models["Logistic Regression"],
        X_test,
        y_test
    )

    # Stage 13: Permutation importance.
    permutation_analysis(
        models["Logistic Regression"],
        X_test,
        y_test
    )

    # Stage 14: SHAP explainability.
    shap_analysis(
        models["Logistic Regression"],
        X_train,
        X_test
    )

    # Stage 15: Regression dataset preparation.
    reg = prepare_regression_data(model_df)

    # Stage 16: Main confirmatory interaction model.
    confirmatory_regression(reg)

    # Stage 17: Age-zone omnibus test.
    age_zone_omnibus(reg)

    # Stage 18: Severe-stunting regression.
    severe_stunting_regression(reg)

    # Stage 19: Weighted regression sensitivity analysis.
    weighted_regression(reg)

    # Stage 20: Multicollinearity diagnostics.
    vif_analysis(reg)

    # Stage 21: Selection-bias diagnostics.
    selection_bias_diagnostics(df, model_df)

    # Stage 22: Corrected child-level IPW analysis.
    ipw_analysis(df, reg)

    # Stage 23: Regional prediction error analysis.
    regional_error_analysis(
        model_df,
        y_test,
        predictions,
        test_indices
    )

    print("\n" + "=" * 65)
    print("ANALYSIS PIPELINE COMPLETED")
    print("=" * 65)

    print("Figures saved in:", FIG_DIR)
    print("Tables and results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
