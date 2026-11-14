# Explainable Machine Learning Analysis of Child Stunting Risk and Regional Disparities in Nigeria

Analysis code accompanying the manuscript submitted to *Applied Computing and Informatics* (Emerald).

**Author:** Umar Yusuf Rimingado  
**ORCID:** [0009-0003-5999-3553](https://orcid.org/0009-0003-5999-3553)  
**Affiliation:** Department of Software Engineering, Bayero University Kano, Nigeria

## Overview

This repository contains the Python analysis pipeline for investigating child stunting risk, its socioeconomic determinants, and regional disparities in Nigeria using data from the 2023–24 Nigeria Demographic and Health Survey (NDHS).

The analysis combines statistical modelling and explainable machine learning to examine factors associated with childhood stunting and assess how selected socioeconomic associations vary across Nigeria's six geopolitical zones.

The pipeline includes:

- **Data preparation:** Loading, cleaning, and merging the DHS Children's Recode (KR), Household Recode (HR), and Individual Women's Recode (IR) datasets.
- **Descriptive analysis:** Examining stunting prevalence and its distribution across socioeconomic, demographic, and geographic characteristics.
- **Predictive modelling:** Training Logistic Regression, Random Forest, and XGBoost classifiers.
- **Model evaluation:** Assessing ROC-AUC, PR-AUC, classification performance, bootstrap confidence intervals, cross-validation, and calibration.
- **Model explainability:** Applying SHAP and permutation feature importance to examine the contribution of predictors to model performance.
- **Confirmatory analysis:** Fitting logistic regression models with interactions between geopolitical zone and selected socioeconomic predictors, using cluster-robust standard errors.
- **Multiple-comparison correction:** Applying the Benjamini–Hochberg procedure to interaction-term p-values.
- **Sensitivity analyses:** Examining severe stunting, sampling-weighted regression, selection bias, and inverse-probability weighting.
- **Diagnostics:** Assessing multicollinearity, regional prediction errors, and differences in anthropometric measurement availability.

## Data Availability

**The DHS microdata are not included in this repository.**

In accordance with the DHS Program's data-access and terms-of-use requirements, the original survey microdata cannot be redistributed through this repository.

Researchers can request access to the Nigeria DHS 2023–24 dataset through the [DHS Program website](https://dhsprogram.com/). Access is subject to DHS approval and applicable data-use conditions.

The pipeline requires the following files:

| Filename | Dataset |
|---|---|
| `NGKR8BFL.dta` | Children's Recode |
| `NGHR8BFL.dta` | Household Recode |
| `NGIR8BFL.dta` | Individual Women's Recode |

After obtaining access, place the three files in a local `data/` directory. The directory is intended for local use and should not be committed to the public repository.

## Repository Structure

```text
Child-Stunting-Analysis/
│
├── README.md
├── LICENSE
├── requirements.txt
├── analysis_pipeline.py
│
├── data/
│   ├── NGKR8BFL.dta
│   ├── NGHR8BFL.dta
│   └── NGIR8BFL.dta
│
├── figures/
│   └── Generated visualisations
│
└── outputs/
    ├── Analytical datasets
    ├── Descriptive statistics
    ├── Model evaluation results
    ├── Regression summaries
    └── Sensitivity analysis results
```

The `data/`, `figures/`, and `outputs/` directories may be created locally when running the analysis. Raw DHS data and generated outputs are not necessarily distributed with the source code.

## Reproducing the Analysis

### 1. Obtain the DHS data

Register at the [DHS Program](https://dhsprogram.com/) and request access to the Nigeria DHS 2023–24 datasets.

Download the KR, HR, and IR recode files listed in the Data Availability section.

### 2. Set up the project

Clone the repository:

```bash
git clone https://github.com/RBED001/Child-Stunting-Analysis.git
cd Child-Stunting-Analysis
```

Place the three DHS `.dta` files in the `data/` directory.

### 3. Install dependencies

It is recommended to use a Python virtual environment.

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Run the analysis

Execute the pipeline from the repository root:

```bash
python analysis_pipeline.py
```

The script performs the analysis sequentially, from DHS data loading and preprocessing through predictive modelling, explainability, confirmatory regression, and sensitivity analyses.

### 5. Access generated results

The pipeline saves generated figures in `figures/` and numerical results, tables, and regression summaries in `outputs/`.

These outputs include:

- Descriptive statistics and sample summaries
- Predictive model performance metrics
- ROC and precision-recall curves
- Confusion matrices and calibration plots
- Bootstrap AUC confidence intervals
- Cross-validation and hyperparameter-tuning results
- SHAP and permutation feature importance
- Confirmatory regression coefficients and interaction results
- Severe-stunting regression results
- Weighted regression and IPW sensitivity results
- Selection-bias and regional prediction-error diagnostics

## Methodological Notes

**Analytical sample:** The primary analysis uses children with valid height-for-age measurements and complete observations for the selected analytical predictors. Missing or flagged anthropometric measurements are excluded from the primary complete-case analysis.

**Predictive modelling:** Models are evaluated using a stratified 80:20 train-test split with a fixed random seed. Hyperparameter tuning is conducted using cross-validation on the training data.

**Confirmatory regression:** The primary regression examines interactions between geopolitical zone and household wealth, maternal education, and maternal age. Standard errors are clustered at the survey cluster level.

**Survey weighting:** The weighted regression is implemented as a sensitivity analysis using DHS sampling weights and cluster-robust covariance estimation. It should not be interpreted as a complete complex-survey design analysis incorporating all survey design components.

**Inverse probability weighting:** The selection-bias analysis estimates the probability of obtaining valid anthropometric measurements. Child-level record identifiers are used when attaching estimated weights to analytical observations to avoid multiplying child records through maternal respondent identifiers. The resulting analysis remains dependent on the selection model and its assumptions.

**Interpretation:** Predictive feature importance and SHAP values describe model behaviour; they do not establish causal effects. Regression associations should likewise be interpreted in the context of the observational survey design.

## Software and Dependencies

The analysis was developed using Python 3.12 and the following principal package versions:

| Package | Version |
|---|---|
| Python | 3.12 |
| pandas | 2.2.2 |
| scikit-learn | 1.6.1 |
| XGBoost | 3.3.0 |
| SHAP | 0.52.0 |
| statsmodels | 0.14.6 |

Additional dependencies are listed in `requirements.txt`.

## Citation

If you use this code or build upon the analysis, please cite the associated manuscript:

*Explainable Machine Learning Analysis of Child Stunting Risk and Regional Disparities in Nigeria.*

Full bibliographic details will be added following publication.

## License

The analysis code in this repository is released under the MIT License. See the `LICENSE` file for details.

The MIT License applies only to the source code and does not extend to the DHS microdata. The original survey data remain subject to the DHS Program's terms of use and access conditions.
