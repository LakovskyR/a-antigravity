"""
Statistical Engine - Module 2.
Executes selected statistical modules on cleaned survey codes.
Outputs: statistical_results.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def load_config() -> dict:
    """Load run configuration."""
    config_path = Path(__file__).parent.parent / "tmp" / "run_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict):
    """Save updated configuration."""
    config_path = Path(__file__).parent.parent / "tmp" / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_data():
    """Load cleaned codes and mapping dictionary."""
    tmp_dir = Path(__file__).parent.parent / "tmp"
    df = pd.read_csv(tmp_dir / "cleaned_codes.csv")
    with open(tmp_dir / "mapping_dict.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return df, mapping


def save_results(results: dict):
    """Save statistical results to tmp/."""
    tmp_dir = Path(__file__).parent.parent / "tmp"
    results_path = tmp_dir / "statistical_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    try:
        print(f"Saved: {results_path}")
    except UnicodeEncodeError:
        print(f"Saved: {results_path.name}")


def bootstrap_ci(data: np.ndarray, statistic=np.mean, n_bootstrap: int = 1000, ci: float = 0.95):
    """Calculate bootstrap confidence interval."""
    n = len(data)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_stats, alpha * 100)
    upper = np.percentile(boot_stats, (1 - alpha) * 100)
    return float(statistic(data)), float(lower), float(upper)


def run_descriptive(df: pd.DataFrame, mapping: dict) -> dict:
    """Run descriptive statistics module."""
    print("Running Descriptive Statistics...")
    results = {"frequencies": {}, "means": {}, "crosstabs": {}}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "respondent_id"]

    for col in numeric_cols:
        freq = df[col].value_counts(dropna=False).sort_index()
        results["frequencies"][col] = {
            "labels": [str(x) for x in freq.index.tolist()],
            "counts": freq.values.tolist(),
            "percentages": (freq / len(df) * 100).round(2).tolist(),
        }

        valid_data = df[col].dropna().values
        if len(valid_data) > 0:
            mean, ci_low, ci_high = bootstrap_ci(valid_data)
            results["means"][col] = {
                "mean": round(mean, 3),
                "ci_95_low": round(ci_low, 3),
                "ci_95_high": round(ci_high, 3),
                "n": len(valid_data),
            }

    if len(numeric_cols) >= 2:
        ct = pd.crosstab(df[numeric_cols[0]], df[numeric_cols[1]])
        results["crosstabs"][f"{numeric_cols[0]}_x_{numeric_cols[1]}"] = {
            "row_variable": numeric_cols[0],
            "col_variable": numeric_cols[1],
            "table": ct.to_dict(),
        }

    return results


def run_pca(df: pd.DataFrame, mapping: dict) -> dict:
    """
    PCA - Principal Component Analysis.
    Reduces dimensionality before clustering, identifies main variance axes.
    Outputs: explained variance, loadings, scree data, transformed coordinates.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("Running PCA...")
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["respondent_id"], errors="ignore").copy()
    if numeric_df.empty:
        return {
            "status": "incompatible_data",
            "reason": "No numeric indicators available for PCA.",
        }

    min_non_null = 20
    coverage = numeric_df.notna().sum()
    keep_cols = coverage[coverage >= min_non_null].index.tolist()
    if len(keep_cols) < 3:
        return {
            "status": "incompatible_data",
            "reason": (
                f"Need at least 3 numeric columns with >= {min_non_null} non-null values. "
                f"Found {len(keep_cols)}."
            ),
        }
    numeric_df = numeric_df[keep_cols]

    variances = numeric_df.var(numeric_only=True, skipna=True)
    usable_cols = variances[variances > 0].index.tolist()
    if len(usable_cols) < 3:
        return {
            "status": "incompatible_data",
            "reason": "Need at least 3 non-constant numeric columns for PCA.",
        }
    numeric_df = numeric_df[usable_cols]

    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(numeric_df)
    if x_imputed.shape[0] < 10:
        return {
            "status": "incompatible_data",
            "reason": "Need >=10 rows and >=3 numeric columns",
        }

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    pca_full = PCA(random_state=42)
    pca_full.fit(x_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, 0.80)) + 1
    n_components = min(n_components, 10, len(numeric_df.columns))

    pca = PCA(n_components=n_components, random_state=42)
    x_pca = pca.fit_transform(x_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=numeric_df.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    top_vars = {}
    for pc in loadings.columns:
        sorted_vars = loadings[pc].abs().sort_values(ascending=False)
        top_vars[pc] = sorted_vars.head(5).index.tolist()

    tmp_dir = Path(__file__).parent.parent / "tmp"
    pca_coords_path = tmp_dir / "pca_coordinates.csv"
    pca_df = pd.DataFrame(x_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df.to_csv(pca_coords_path, index=False)

    return {
        "status": "ok",
        "n_components_selected": n_components,
        "variance_explained_pct": [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_],
        "cumulative_variance_pct": [
            round(float(v) * 100, 2) for v in np.cumsum(pca.explained_variance_ratio_)
        ],
        "scree_data": {
            "all_components": list(range(1, len(pca_full.explained_variance_ratio_) + 1)),
            "all_variance": [round(float(v) * 100, 2) for v in pca_full.explained_variance_ratio_],
        },
        "top_variables_per_pc": top_vars,
        "loadings": loadings.round(3).to_dict(),
        "coordinates_file": str(pca_coords_path),
        "note": f"PCA coordinates saved to {pca_coords_path} - used by clustering modules.",
    }


def select_optimal_k(x_scaled: np.ndarray, k_range: range) -> dict:
    """
    Compare 4 k-selection methods and return consensus k.
    Methods: Silhouette (max), Davies-Bouldin (min), Calinski-Harabasz (max), Elbow (2nd diff).
    Consensus = median of the 4 recommendations.
    """
    metrics = {
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": [],
        "inertia": [],
    }
    k_list = list(k_range)
    if not k_list:
        return {
            "k_silhouette": 2,
            "k_davies_bouldin": 2,
            "k_calinski_harabasz": 2,
            "k_elbow": 2,
            "k_consensus": 2,
            "metrics_by_k": {"k": [], **metrics},
        }

    for k in k_list:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(x_scaled)
        metrics["inertia"].append(float(model.inertia_))

        # Some metrics require >1 cluster and sufficient separation.
        if len(np.unique(labels)) > 1:
            metrics["silhouette"].append(float(silhouette_score(x_scaled, labels)))
            metrics["davies_bouldin"].append(float(davies_bouldin_score(x_scaled, labels)))
            metrics["calinski_harabasz"].append(float(calinski_harabasz_score(x_scaled, labels)))
        else:
            metrics["silhouette"].append(float("-inf"))
            metrics["davies_bouldin"].append(float("inf"))
            metrics["calinski_harabasz"].append(float("-inf"))

    k_silhouette = k_list[int(np.argmax(metrics["silhouette"]))]
    k_db = k_list[int(np.argmin(metrics["davies_bouldin"]))]
    k_ch = k_list[int(np.argmax(metrics["calinski_harabasz"]))]

    inertias = np.array(metrics["inertia"], dtype=float)
    if len(inertias) >= 3:
        second_diffs = np.diff(np.diff(inertias))
        k_elbow = k_list[int(np.argmax(np.abs(second_diffs))) + 2]
    else:
        k_elbow = k_silhouette

    k_consensus = int(np.median([k_silhouette, k_db, k_ch, k_elbow]))

    return {
        "k_silhouette": int(k_silhouette),
        "k_davies_bouldin": int(k_db),
        "k_calinski_harabasz": int(k_ch),
        "k_elbow": int(k_elbow),
        "k_consensus": int(k_consensus),
        "metrics_by_k": {
            "k": [int(k) for k in k_list],
            "silhouette": [float(v) for v in metrics["silhouette"]],
            "davies_bouldin": [float(v) for v in metrics["davies_bouldin"]],
            "calinski_harabasz": [float(v) for v in metrics["calinski_harabasz"]],
            "inertia": [float(v) for v in metrics["inertia"]],
        },
    }


def run_typology(df: pd.DataFrame, mapping: dict) -> dict:
    """Run K-Means typology with automated K selection."""
    print("Running Typology (K-Means)...")
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["respondent_id"], errors="ignore").copy()
    if numeric_df.empty:
        return {"status": "incompatible_data", "reason": "No numeric data available for clustering."}

    min_non_null = 20
    coverage = numeric_df.notna().sum()
    keep_cols = coverage[coverage >= min_non_null].index.tolist()
    if len(keep_cols) < 2:
        return {
            "status": "incompatible_data",
            "reason": f"Need >=2 numeric columns with >= {min_non_null} non-null values.",
        }
    numeric_df = numeric_df[keep_cols]

    variances = numeric_df.var(numeric_only=True, skipna=True)
    usable_cols = variances[variances > 0].index.tolist()
    if len(usable_cols) < 2:
        return {"status": "incompatible_data", "reason": "Need >=2 non-constant numeric columns."}
    numeric_df = numeric_df[usable_cols]

    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(numeric_df)
    if x_imputed.shape[0] < 10:
        return {"status": "incompatible_data", "reason": "Insufficient rows (minimum 10 observations)."}

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    max_k = min(8, max(2, len(x_scaled) // 10 + 1))
    k_range = range(2, max_k + 1)
    k_selection = select_optimal_k(x_scaled, k_range)
    best_k = int(k_selection["k_consensus"])

    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = final_model.fit_predict(x_scaled)

    results = {
        "status": "ok",
        "selection_method": "4_metric_consensus_median",
        "optimal_k": int(best_k),
        "k_recommendations": {
            "silhouette": int(k_selection["k_silhouette"]),
            "davies_bouldin": int(k_selection["k_davies_bouldin"]),
            "calinski_harabasz": int(k_selection["k_calinski_harabasz"]),
            "elbow": int(k_selection["k_elbow"]),
            "consensus": int(k_selection["k_consensus"]),
        },
        "cluster_sizes": pd.Series(cluster_labels).value_counts().sort_index().to_dict(),
        "metrics_by_k": k_selection["metrics_by_k"],
        # Backward-compatible fields used by existing downstream views.
        "elbow_data": {
            "k": k_selection["metrics_by_k"]["k"],
            "inertia": k_selection["metrics_by_k"]["inertia"],
        },
        "silhouette_data": {
            "k": k_selection["metrics_by_k"]["k"],
            "scores": k_selection["metrics_by_k"]["silhouette"],
        },
        "silhouette_score": float(max(k_selection["metrics_by_k"]["silhouette"])),
    }

    if UMAP_AVAILABLE and len(x_scaled) > 10:
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(x_scaled)
        results["umap_coordinates"] = {
            "x": embedding[:, 0].tolist(),
            "y": embedding[:, 1].tolist(),
            "cluster": cluster_labels.tolist(),
        }

    return results


def run_modeling(df: pd.DataFrame, mapping: dict) -> dict:
    """Run model comparison: RF vs XGBoost vs CatBoost."""
    print("Running Model Comparison...")
    numeric_df = df.select_dtypes(include=[np.number]).dropna()

    if len(numeric_df.columns) < 2:
        return {"error": "Insufficient variables"}

    x = numeric_df.iloc[:, :-1].drop(columns=["respondent_id"], errors="ignore")
    y = numeric_df.iloc[:, -1]

    if y.nunique() > 5:
        y = (y > y.median()).astype(int)

    if x.empty:
        return {"error": "No features available"}

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    results = {"candidates": []}
    scores: Dict[str, float] = {}

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    rf_proba = rf.predict_proba(x_test)[:, 1] if len(np.unique(y)) == 2 else None

    rf_scores = {
        "accuracy": float(accuracy_score(y_test, rf_pred)),
        "f1": float(f1_score(y_test, rf_pred, average="weighted")),
        "auc": float(roc_auc_score(y_test, rf_proba)) if rf_proba is not None else None,
    }
    results["candidates"].append({"model": "random_forest", "scores": rf_scores})
    scores["random_forest"] = rf_scores["f1"]

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )
    cv_auc_mean = None
    cv_auc_std = None
    try:
        cv_folds = min(5, max(2, int(y.value_counts().min())))
        if len(np.unique(y)) == 2 and cv_folds >= 2:
            cv_scores = cross_val_score(gb, x, y, cv=cv_folds, scoring="roc_auc")
            cv_auc_mean = float(cv_scores.mean())
            cv_auc_std = float(cv_scores.std())
    except Exception:
        cv_auc_mean = None
        cv_auc_std = None

    gb.fit(x_train, y_train)
    gb_pred = gb.predict(x_test)
    gb_proba = gb.predict_proba(x_test)[:, 1] if len(np.unique(y)) == 2 else None
    gb_scores = {
        "accuracy": float(accuracy_score(y_test, gb_pred)),
        "f1": float(f1_score(y_test, gb_pred, average="weighted")),
        "auc": float(roc_auc_score(y_test, gb_proba)) if gb_proba is not None else None,
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
    }
    results["candidates"].append({"model": "gradient_boosting", "scores": gb_scores})
    scores["gradient_boosting"] = gb_scores["f1"]

    if XGB_AVAILABLE:
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
        xgb_model.fit(x_train, y_train)
        xgb_pred = xgb_model.predict(x_test)
        xgb_proba = xgb_model.predict_proba(x_test)[:, 1] if len(np.unique(y)) == 2 else None
        xgb_scores = {
            "accuracy": float(accuracy_score(y_test, xgb_pred)),
            "f1": float(f1_score(y_test, xgb_pred, average="weighted")),
            "auc": float(roc_auc_score(y_test, xgb_proba)) if xgb_proba is not None else None,
        }
        results["candidates"].append({"model": "xgboost", "scores": xgb_scores})
        scores["xgboost"] = xgb_scores["f1"]

    if CATBOOST_AVAILABLE:
        cb_model = cb.CatBoostClassifier(iterations=100, verbose=False, random_state=42)
        cb_model.fit(x_train, y_train)
        cb_pred = cb_model.predict(x_test)
        cb_proba = cb_model.predict_proba(x_test)[:, 1] if len(np.unique(y)) == 2 else None
        cb_scores = {
            "accuracy": float(accuracy_score(y_test, cb_pred)),
            "f1": float(f1_score(y_test, cb_pred, average="weighted")),
            "auc": float(roc_auc_score(y_test, cb_proba)) if cb_proba is not None else None,
        }
        results["candidates"].append({"model": "catboost", "scores": cb_scores})
        scores["catboost"] = cb_scores["f1"]

    winner = max(scores, key=scores.get)
    results["winner"] = winner
    results["winner_metric"] = "f1"
    results["winner_score"] = float(scores[winner])

    if SHAP_AVAILABLE:
        try:
            if winner == "random_forest":
                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(x_test)
            elif winner == "xgboost" and XGB_AVAILABLE:
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(x_test)
            elif winner == "gradient_boosting":
                explainer = shap.TreeExplainer(gb)
                shap_values = explainer.shap_values(x_test)
            else:
                explainer = shap.TreeExplainer(cb_model)
                shap_values = explainer.shap_values(x_test)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            mean_shap = np.abs(shap_values).mean(axis=0)
            top_idx = np.argsort(mean_shap)[-5:]
            results["shap_summary"] = {
                "top_features": x.columns[top_idx].tolist(),
                "mean_shap_values": mean_shap[top_idx].tolist(),
            }
        except Exception as exc:
            results["shap_summary"] = {"error": str(exc)}

    return results


def run_regression(df: pd.DataFrame, mapping: dict) -> dict:
    """Run logistic regression with bootstrap confidence intervals."""
    print("Running Logistic Regression...")
    from sklearn.linear_model import LogisticRegression

    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if len(numeric_df.columns) < 2:
        return {"error": "Insufficient variables"}

    x = numeric_df.iloc[:, :-1].drop(columns=["respondent_id"], errors="ignore")
    y = numeric_df.iloc[:, -1]
    if y.nunique() > 2:
        y = (y > y.median()).astype(int)

    if x.empty:
        return {"error": "No features available"}

    model = LogisticRegression(max_iter=1000)
    model.fit(x, y)

    coefs = []
    for i, col in enumerate(x.columns):
        boot_coefs = []
        for _ in range(500):
            idx = np.random.choice(len(x), size=len(x), replace=True)
            x_boot = x.iloc[idx]
            y_boot = y.iloc[idx]
            try:
                boot_model = LogisticRegression(max_iter=1000)
                boot_model.fit(x_boot, y_boot)
                boot_coefs.append(float(boot_model.coef_[0][i]))
            except Exception:
                continue

        ci_low = np.percentile(boot_coefs, 2.5) if boot_coefs else None
        ci_high = np.percentile(boot_coefs, 97.5) if boot_coefs else None
        coefs.append(
            {
                "variable": col,
                "coefficient": float(model.coef_[0][i]),
                "ci_95_low": float(ci_low) if ci_low is not None else None,
                "ci_95_high": float(ci_high) if ci_high is not None else None,
            }
        )

    return {
        "intercept": float(model.intercept_[0]),
        "coefficients": coefs,
        "accuracy": float(model.score(x, y)),
    }


def run_anova(df: pd.DataFrame, mapping: dict) -> dict:
    """Run one-way ANOVA with optional Tukey HSD post-hoc."""
    print("Running ANOVA...")
    work_df = df.copy()
    numeric_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return {
            "status": "incompatible_data",
            "reason": "ANOVA needs at least two numeric columns (group + outcome).",
        }

    y_col = numeric_cols[-1]
    candidate_groups = [c for c in work_df.columns if c != y_col]
    group_col = None

    for col in candidate_groups:
        nunique = work_df[col].nunique(dropna=True)
        if 2 <= nunique <= 12:
            group_col = col
            break

    if group_col is None:
        group_col = numeric_cols[0]
        try:
            work_df["_anova_group"] = pd.qcut(
                pd.to_numeric(work_df[group_col], errors="coerce"),
                q=min(4, work_df[group_col].nunique(dropna=True)),
                duplicates="drop",
            )
            group_col = "_anova_group"
        except Exception:
            return {
                "status": "incompatible_data",
                "reason": "Could not derive a valid grouping variable for ANOVA.",
            }

    anova_df = work_df[[group_col, y_col]].dropna()
    if len(anova_df) < 8:
        return {
            "status": "incompatible_data",
            "reason": "Insufficient non-null rows for ANOVA (minimum 8).",
        }

    grouped = [grp[y_col].values for _, grp in anova_df.groupby(group_col)]
    grouped = [g for g in grouped if len(g) > 1]
    if len(grouped) < 2:
        return {
            "status": "incompatible_data",
            "reason": "Need at least two groups with more than one observation.",
        }

    f_stat, p_val = stats.f_oneway(*grouped)
    out = {
        "status": "ok",
        "method": "one_way_anova",
        "group_variable": str(group_col),
        "target_variable": str(y_col),
        "n_observations": int(len(anova_df)),
        "n_groups": int(len(grouped)),
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "significant_0_05": bool(p_val < 0.05),
    }

    if STATSMODELS_AVAILABLE:
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            tukey = pairwise_tukeyhsd(
                endog=anova_df[y_col].astype(float),
                groups=anova_df[group_col].astype(str),
                alpha=0.05,
            )
            tukey_rows = []
            for row in tukey.summary().data[1:]:
                tukey_rows.append(
                    {
                        "group1": str(row[0]),
                        "group2": str(row[1]),
                        "mean_diff": float(row[2]),
                        "p_adj": float(row[3]),
                        "reject_h0": bool(row[6]),
                    }
                )
            out["tukey_hsd"] = tukey_rows
        except Exception as exc:
            out["tukey_hsd_error"] = str(exc)

    return out


def run_ols(df: pd.DataFrame, mapping: dict) -> dict:
    """Run OLS linear regression with p-values and confidence intervals."""
    print("Running OLS...")
    from sklearn.impute import SimpleImputer

    numeric_df = (
        df.select_dtypes(include=[np.number]).drop(columns=["respondent_id"], errors="ignore").copy()
    )
    if numeric_df.empty:
        return {
            "status": "incompatible_data",
            "reason": "OLS needs at least one target and one predictor (numeric columns not found).",
        }

    min_non_null = 20
    coverage = numeric_df.notna().sum()
    keep_cols = coverage[coverage >= min_non_null].index.tolist()
    if len(keep_cols) < 2:
        return {
            "status": "incompatible_data",
            "reason": (
                f"Need at least 2 numeric columns with >= {min_non_null} non-null values. "
                f"Found {len(keep_cols)}."
            ),
        }
    numeric_df = numeric_df[keep_cols]

    variances = numeric_df.var(numeric_only=True, skipna=True).sort_values(ascending=False)
    usable_cols = variances[variances > 0].index.tolist()
    if len(usable_cols) < 2:
        return {"status": "incompatible_data", "reason": "No usable numeric variation for OLS."}

    # Keep the model stable on high-dimensional survey matrices.
    usable_cols = usable_cols[:20]
    numeric_df = numeric_df[usable_cols]

    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)

    y_col = imputed.columns[0]
    x_df = imputed.drop(columns=[y_col], errors="ignore")
    y = imputed[y_col]

    if x_df.empty:
        return {"status": "incompatible_data", "reason": "No predictors available for OLS."}

    if STATSMODELS_AVAILABLE:
        try:
            x_const = sm.add_constant(x_df, has_constant="add")
            model = sm.OLS(y, x_const).fit()
            conf = model.conf_int()
            coef_rows = []
            for var in model.params.index:
                if var == "const":
                    continue
                coef_rows.append(
                    {
                        "variable": str(var),
                        "coefficient": float(model.params[var]),
                        "p_value": float(model.pvalues[var]),
                        "ci_95_low": float(conf.loc[var, 0]),
                        "ci_95_high": float(conf.loc[var, 1]),
                    }
                )
            return {
                "status": "ok",
                "method": "statsmodels.OLS",
                "target_variable": str(y_col),
                "n_observations": int(len(imputed)),
                "r_squared": float(model.rsquared),
                "r_squared_adj": float(model.rsquared_adj),
                "f_statistic": float(model.fvalue) if model.fvalue is not None else None,
                "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else None,
                "coefficients": coef_rows,
            }
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(x_df, y)
    preds = lr.predict(x_df)
    mse = float(np.mean((y - preds) ** 2))
    return {
        "status": "ok",
        "method": "sklearn.LinearRegression",
        "target_variable": str(y_col),
        "n_observations": int(len(imputed)),
        "r_squared": float(lr.score(x_df, y)),
        "mse": mse,
        "coefficients": [
            {"variable": str(col), "coefficient": float(coef)}
            for col, coef in zip(x_df.columns, lr.coef_)
        ],
        "note": "statsmodels not available; p-values/CI omitted.",
    }


def run_cah(df: pd.DataFrame, mapping: dict) -> dict:
    """Run hierarchical agglomerative clustering (Ward)."""
    print("Running CAH (Hierarchical Clustering)...")
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["respondent_id"], errors="ignore").dropna()
    if len(numeric_df) < 10 or len(numeric_df.columns) < 2:
        return {"status": "incompatible_data", "reason": "Need >=10 rows and >=2 numeric columns."}

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(numeric_df)
    max_k = min(8, max(3, len(numeric_df) // 10 + 1))
    k_range = range(2, max_k)

    sil_scores = {}
    labels_by_k = {}
    for k in k_range:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(x_scaled)
        labels_by_k[k] = labels
        try:
            sil_scores[k] = float(silhouette_score(x_scaled, labels))
        except Exception:
            sil_scores[k] = -1.0

    best_k = max(sil_scores, key=sil_scores.get)
    best_labels = labels_by_k[best_k]
    uniq, cnts = np.unique(best_labels, return_counts=True)

    return {
        "status": "ok",
        "method": "AgglomerativeClustering(ward)",
        "optimal_k": int(best_k),
        "silhouette_by_k": {str(k): float(v) for k, v in sil_scores.items()},
        "cluster_sizes": {int(k): int(v) for k, v in zip(uniq, cnts)},
    }


def run_lda(df: pd.DataFrame, mapping: dict) -> dict:
    """Run linear discriminant analysis for segment profiling."""
    print("Running LDA...")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if len(numeric_df.columns) < 3:
        return {"status": "incompatible_data", "reason": "Need at least 3 numeric columns for LDA."}

    y_raw = numeric_df.iloc[:, -1]
    x_df = numeric_df.iloc[:, :-1].drop(columns=["respondent_id"], errors="ignore")
    if x_df.empty:
        return {"status": "incompatible_data", "reason": "No feature columns available for LDA."}

    if y_raw.nunique() > 8:
        y = pd.qcut(y_raw, q=min(3, y_raw.nunique()), labels=False, duplicates="drop")
    else:
        y = y_raw.astype("category").cat.codes

    if len(np.unique(y)) < 2:
        return {"status": "incompatible_data", "reason": "LDA requires at least 2 classes."}

    x_train, x_test, y_train, y_test = train_test_split(x_df, y, test_size=0.2, random_state=42, stratify=y)
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    pred = lda.predict(x_test)

    out = {
        "status": "ok",
        "method": "LinearDiscriminantAnalysis",
        "n_observations": int(len(numeric_df)),
        "n_classes": int(len(np.unique(y))),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
    }

    if hasattr(lda, "explained_variance_ratio_"):
        out["explained_variance_ratio"] = [float(v) for v in lda.explained_variance_ratio_]
    if hasattr(lda, "coef_"):
        out["coefficients_shape"] = list(lda.coef_.shape)
    return out


def run_decision_tree(df: pd.DataFrame, mapping: dict) -> dict:
    """Run interpretable decision tree and return human-readable rules."""
    print("Running Decision Tree...")
    from sklearn.tree import DecisionTreeClassifier, export_text

    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if len(numeric_df.columns) < 2:
        return {"status": "incompatible_data", "reason": "Need at least one target and one feature."}

    x_df = numeric_df.iloc[:, :-1].drop(columns=["respondent_id"], errors="ignore")
    y_raw = numeric_df.iloc[:, -1]
    if x_df.empty:
        return {"status": "incompatible_data", "reason": "No features available for decision tree."}

    if y_raw.nunique() > 6:
        y = (y_raw > y_raw.median()).astype(int)
    else:
        y = y_raw.astype("category").cat.codes

    if len(np.unique(y)) < 2:
        return {"status": "incompatible_data", "reason": "Need at least 2 classes for tree classifier."}

    x_train, x_test, y_train, y_test = train_test_split(x_df, y, test_size=0.2, random_state=42, stratify=y)
    tree = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_leaf=5)
    tree.fit(x_train, y_train)
    pred = tree.predict(x_test)

    rules = export_text(tree, feature_names=list(x_df.columns))
    importances = {
        str(col): float(imp)
        for col, imp in zip(x_df.columns, tree.feature_importances_)
        if imp > 0
    }

    return {
        "status": "ok",
        "method": "DecisionTreeClassifier(max_depth=4)",
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
        "n_leaves": int(tree.get_n_leaves()),
        "tree_depth": int(tree.get_depth()),
        "feature_importances": dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True)),
        "rules": rules,
    }


def run_cbc(df: pd.DataFrame, mapping: dict) -> dict:
    """
    Run Conjoint-Based Choice analysis.

    Required structure:
    Columns: choice_set_id, alternative_id, chosen (0/1), attribute columns.
    """
    print("Running CBC Analysis...")
    required_cols = ["choice_set_id", "alternative_id", "chosen"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        return {
            "status": "incompatible_data",
            "required_structure": "Columns: choice_set_id, alternative_id, chosen (0/1), attribute columns",
            "columns_found": df.columns.tolist(),
            "recommendation": "Run descriptive module first to inspect data structure",
            "missing_columns": missing,
        }

    if not STATSMODELS_AVAILABLE:
        return {
            "status": "incompatible_data",
            "required_structure": "Columns: choice_set_id, alternative_id, chosen (0/1), attribute columns",
            "columns_found": df.columns.tolist(),
            "recommendation": "Install statsmodels and rerun CBC",
            "missing_dependency": "statsmodels",
        }

    attribute_cols = [c for c in df.columns if c not in required_cols]
    if not attribute_cols:
        return {
            "status": "incompatible_data",
            "required_structure": "Columns: choice_set_id, alternative_id, chosen (0/1), attribute columns",
            "columns_found": df.columns.tolist(),
            "recommendation": "Add conjoint attribute columns and rerun",
        }

    model_df = df[required_cols + attribute_cols].dropna().copy()
    if model_df.empty:
        return {
            "status": "incompatible_data",
            "required_structure": "Columns: choice_set_id, alternative_id, chosen (0/1), attribute columns",
            "columns_found": df.columns.tolist(),
            "recommendation": "No usable non-null conjoint rows found",
        }

    # Assumption: each row is an alternative inside a choice set and chosen marks selected alternative.
    # We fit MNLogit with alternative_id as class and attributes as predictors.
    x = pd.get_dummies(model_df[attribute_cols], drop_first=True)
    if x.empty:
        return {
            "status": "incompatible_data",
            "required_structure": "Columns: choice_set_id, alternative_id, chosen (0/1), attribute columns",
            "columns_found": df.columns.tolist(),
            "recommendation": "Attributes are not suitable for modeling after encoding",
        }

    y_codes = model_df["alternative_id"].astype("category").cat.codes
    x = sm.add_constant(x, has_constant="add")

    try:
        mnlogit = sm.MNLogit(y_codes, x)
        fit = mnlogit.fit(disp=False)
        params = fit.params

        feature_importance = (
            params.abs().mean(axis=1).sort_values(ascending=False).head(10).to_dict()
        )
        utilities = {str(k): float(v) for k, v in feature_importance.items()}
        utilities_path = Path(__file__).parent.parent / "tmp" / "cbc_utilities.json"
        with open(utilities_path, "w", encoding="utf-8") as f:
            json.dump(utilities, f, indent=2, ensure_ascii=False)

        return {
            "status": "ok",
            "method": "statsmodels.MNLogit",
            "assumption": "Input rows represent alternatives within choice sets; chosen is available for diagnostics",
            "n_rows": int(len(model_df)),
            "n_choice_sets": int(model_df["choice_set_id"].nunique()),
            "n_alternatives": int(model_df["alternative_id"].nunique()),
            "pseudo_r2": float(getattr(fit, "prsquared", np.nan)),
            "log_likelihood": float(getattr(fit, "llf", np.nan)),
            "top_attribute_utilities": [
                {"feature": k, "importance": float(v)} for k, v in feature_importance.items()
            ],
            "utilities_file": str(utilities_path),
        }
    except Exception as exc:
        return {
            "status": "incompatible_data",
            "required_structure": "Columns: choice_set_id, alternative_id, chosen (0/1), attribute columns",
            "columns_found": df.columns.tolist(),
            "recommendation": "Run descriptive module first to inspect data structure",
            "error": str(exc),
        }


def run_cbc_simulator(df: pd.DataFrame, mapping: dict) -> dict:
    """
    CBC Share-of-Preference Simulator.

    Input: tmp/cbc_utilities.json (written by run_cbc if successful)
           + tmp/cbc_scenarios.json (analyst-defined product profiles)
    """
    print("Running CBC Simulator...")
    tmp_dir = Path(__file__).parent.parent / "tmp"
    utilities_path = tmp_dir / "cbc_utilities.json"
    scenarios_path = tmp_dir / "cbc_scenarios.json"

    if utilities_path.exists():
        with open(utilities_path, encoding="utf-8") as f:
            utilities = json.load(f)
    else:
        results_path = tmp_dir / "statistical_results.json"
        if results_path.exists():
            with open(results_path, encoding="utf-8") as f:
                prev = json.load(f)
            cbc_result = prev.get("results", {}).get("cbc", {})
            if cbc_result.get("status") == "ok":
                utilities = {
                    item["feature"]: item["importance"]
                    for item in cbc_result.get("top_attribute_utilities", [])
                }
            else:
                return {
                    "status": "incompatible_data",
                    "reason": "No CBC utilities found. Run 'cbc' module first to estimate utilities.",
                    "required_file": "tmp/cbc_utilities.json",
                    "format": {
                        "description": "dict mapping attribute levels to utility values",
                        "example": {
                            "efficacy_high": 1.8,
                            "efficacy_low": -0.4,
                            "price_low": 0.9,
                            "price_high": -1.2,
                            "admin_oral": 0.6,
                            "admin_iv": -0.6,
                        },
                    },
                }
        else:
            return {
                "status": "incompatible_data",
                "reason": "No CBC utilities found. Run 'cbc' module first.",
                "required_file": "tmp/cbc_utilities.json",
            }

    if not utilities:
        return {
            "status": "incompatible_data",
            "reason": "Utilities are empty. Run 'cbc' with compatible conjoint data first.",
        }

    if scenarios_path.exists():
        with open(scenarios_path, encoding="utf-8") as f:
            scenarios = json.load(f)
    else:
        attr_keys = list(utilities.keys())
        mid = max(1, len(attr_keys) // 2)
        scenarios = [
            {
                "name": "Product_A",
                "attributes": {k: 1 for k in attr_keys[:mid]},
            },
            {
                "name": "Product_B",
                "attributes": {k: 1 for k in attr_keys[mid:]},
            },
            {
                "name": "Status_Quo",
                "attributes": {k: 0 for k in attr_keys},
            },
        ]

        with open(scenarios_path, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)
        print(f"  Auto-generated scenarios saved to {scenarios_path.name} - edit to customize.")

    def scenario_utility(scenario: dict) -> float:
        total = 0.0
        for attr, value in scenario.get("attributes", {}).items():
            total += float(utilities.get(attr, 0.0)) * float(value)
        return total

    raw_utils = [scenario_utility(s) for s in scenarios]
    max_u = max(raw_utils) if raw_utils else 0.0
    exp_utils = [np.exp(u - max_u) for u in raw_utils]
    total_exp = sum(exp_utils) or 1.0
    shares = [round(e / total_exp * 100, 2) for e in exp_utils]

    share_results = [
        {
            "scenario": s["name"],
            "utility": round(raw_utils[i], 4),
            "market_share_pct": shares[i],
        }
        for i, s in enumerate(scenarios)
    ]

    winner_idx = int(np.argmax(shares))
    winner = scenarios[winner_idx]
    sensitivity = []

    for attr in winner.get("attributes", {}):
        if attr not in utilities:
            continue
        modified = {
            "name": f"{winner['name']}_no_{attr}",
            "attributes": {k: (0 if k == attr else v) for k, v in winner["attributes"].items()},
        }
        mod_util = scenario_utility(modified)
        mod_exp = [
            np.exp(mod_util - max_u) if i == winner_idx else exp_utils[i]
            for i in range(len(scenarios))
        ]
        mod_total = sum(mod_exp) or 1.0
        mod_share = round(mod_exp[winner_idx] / mod_total * 100, 2)
        sensitivity.append(
            {
                "attribute_removed": attr,
                "share_without": mod_share,
                "share_delta": round(mod_share - shares[winner_idx], 2),
            }
        )

    sensitivity.sort(key=lambda x: x["share_delta"])

    wtp = {}
    price_attrs = [
        k for k in utilities if "price" in k.lower() or "prix" in k.lower() or "cost" in k.lower()
    ]
    perf_attrs = [k for k in utilities if k not in price_attrs]
    if price_attrs and perf_attrs:
        price_util = abs(float(utilities[price_attrs[0]]))
        if price_util > 0:
            for attr in perf_attrs:
                wtp[attr] = round(float(utilities[attr]) / price_util, 3)

    result = {
        "status": "ok",
        "method": "Multinomial Logit (softmax)",
        "n_scenarios": len(scenarios),
        "market_shares": share_results,
        "winner": scenarios[winner_idx]["name"] if scenarios else None,
        "winner_share_pct": shares[winner_idx] if shares else None,
        "sensitivity_analysis": sensitivity[:10],
        "note": (
            "Edit tmp/cbc_scenarios.json to define custom product profiles. "
            "Attributes must match keys in tmp/cbc_utilities.json."
        ),
    }

    if wtp:
        result["willingness_to_pay_ratios"] = wtp
        result["wtp_note"] = (
            "WTP = utility(attribute) / |utility(price)|. "
            "Interpret as relative monetary value of each attribute level."
        )

    with open(utilities_path, "w", encoding="utf-8") as f:
        json.dump(utilities, f, indent=2, ensure_ascii=False)

    return result


def run_cas_vignettes(df: pd.DataFrame, mapping: dict) -> dict:
    """
    Cas Vignettes - Factorial Survey / Vignette Study module.
    """
    print("Running Cas Vignettes...")
    from itertools import product as iterproduct
    import math

    tmp_dir = Path(__file__).parent.parent / "tmp"
    vig_cols = [c for c in df.columns if c.lower().startswith("vig_") or c.lower().startswith("vignette_")]

    if vig_cols:
        print(f"  MODE 2: Analyzing {len(vig_cols)} vignette columns...")
        rating_col = vig_cols[-1]
        attr_cols = vig_cols[:-1]

        if not attr_cols:
            return {
                "status": "incompatible_data",
                "reason": "Only one vignette column found - need attribute columns + rating column",
                "expected_pattern": "VIG_attr1, VIG_attr2, ..., VIG_rating",
            }

        work_df = df[vig_cols].dropna()
        if len(work_df) < 10:
            return {
                "status": "incompatible_data",
                "reason": "Insufficient non-null vignette rows (minimum 10)",
            }

        if not STATSMODELS_AVAILABLE:
            return {
                "status": "error",
                "reason": "statsmodels required for vignette OLS. Run: uv pip install statsmodels",
            }

        x = pd.get_dummies(work_df[attr_cols], drop_first=True)
        y = pd.to_numeric(work_df[rating_col], errors="coerce")
        valid = y.notna()
        x = x.loc[valid]
        y = y.loc[valid]
        x_const = sm.add_constant(x)

        try:
            model = sm.OLS(y, x_const).fit()
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

        part_worths = [
            {
                "variable": str(var),
                "utility": float(model.params[var]),
                "p_value": float(model.pvalues[var]),
                "significant": bool(model.pvalues[var] < 0.05),
            }
            for var in model.params.index
            if var != "const"
        ]

        importances = {}
        for col in attr_cols:
            related = [pw for pw in part_worths if pw["variable"].startswith(col)]
            if related:
                utils = [pw["utility"] for pw in related]
                importances[col] = round(max(utils) - min(utils), 4)

        total_range = sum(importances.values()) or 1
        rel_importance = {k: round(v / total_range * 100, 2) for k, v in importances.items()}

        return {
            "status": "ok",
            "mode": "analysis",
            "target_variable": rating_col,
            "n_observations": int(len(work_df)),
            "r_squared": float(model.rsquared),
            "r_squared_adj": float(model.rsquared_adj),
            "part_worth_utilities": part_worths,
            "relative_importance_pct": rel_importance,
            "most_important_attribute": max(rel_importance, key=rel_importance.get) if rel_importance else None,
            "note": "Relative importance = range of part-worths / total range. Higher = more influential.",
        }

    print("  MODE 1: Generating vignette design matrix...")
    attr_path = tmp_dir / "vignette_attributes.json"
    if attr_path.exists():
        with open(attr_path, encoding="utf-8") as f:
            attributes = json.load(f)
    else:
        attributes = {
            "Efficacite": ["Forte", "Moderee", "Faible"],
            "Tolerance": ["Bonne", "Moderee"],
            "Voie_admin": ["Oral", "IV", "SC"],
            "Prix": ["Faible", "Eleve"],
            "Donnees_VR": ["Disponibles", "Non_disponibles"],
        }
        with open(attr_path, "w", encoding="utf-8") as f:
            json.dump(attributes, f, indent=2, ensure_ascii=False)
            print(f"  Default attributes written to {attr_path.name} - edit to customize.")

    levels = [v for v in attributes.values()]
    full_factorial_n = 1
    for lvl in levels:
        full_factorial_n *= len(lvl)

    n_vignettes = max(8, min(32, int(math.ceil(math.sqrt(full_factorial_n)))))
    all_combos = list(iterproduct(*levels))

    if len(all_combos) > n_vignettes:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_combos), size=n_vignettes, replace=False)
        selected = [all_combos[i] for i in idx]
    else:
        selected = all_combos

    attr_names = list(attributes.keys())
    design_df = pd.DataFrame(selected, columns=attr_names)
    design_df.insert(0, "vignette_id", range(1, len(design_df) + 1))

    design_path = tmp_dir / "vignette_design.csv"
    design_df.to_csv(design_path, index=False, encoding="utf-8")

    balance = {}
    for col in attr_names:
        counts = design_df[col].value_counts().to_dict()
        balance[col] = {str(k): int(v) for k, v in counts.items()}

    return {
        "status": "ok",
        "mode": "design",
        "n_vignettes": len(design_df),
        "n_attributes": len(attr_names),
        "full_factorial_size": full_factorial_n,
        "design_method": "fractional factorial (random sample, seed=42)",
        "attribute_balance": balance,
        "design_file": str(design_path),
        "next_steps": [
            f"1. Review/edit {attr_path} to define your real attributes and levels",
            "2. Re-run this module to regenerate design",
            f"3. Use {design_path} to program your survey vignettes",
            "4. After data collection, re-run with VIG_-prefixed columns for analysis",
        ],
        "note": (
            "For true D-optimal design, install pyDOE2: uv pip install pyDOE2. "
            "This module will automatically use it if available."
        ),
    }


def run_latent_class(df: pd.DataFrame, mapping: dict) -> dict:
    """
    Latent Class Analysis / Gaussian Mixture Model (Mixte typology).
    """
    print("Running Latent Class / Mixture Model...")
    from sklearn.impute import SimpleImputer
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score

    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["respondent_id"], errors="ignore").copy()
    if numeric_df.empty:
        return {
            "status": "incompatible_data",
            "reason": "No numeric indicators available for latent class.",
        }

    min_non_null = 20
    coverage = numeric_df.notna().sum()
    kept_cols = coverage[coverage >= min_non_null].index.tolist()
    if len(kept_cols) < 2:
        return {
            "status": "incompatible_data",
            "reason": (
                f"Need at least 2 numeric indicators with >= {min_non_null} non-null values. "
                f"Found {len(kept_cols)}."
            ),
        }
    numeric_df = numeric_df[kept_cols]

    variances = numeric_df.var(numeric_only=True, skipna=True)
    usable_cols = variances[variances > 0].index.tolist()
    if len(usable_cols) < 2:
        return {
            "status": "incompatible_data",
            "reason": "Need at least 2 numeric indicators with non-zero variance.",
        }
    numeric_df = numeric_df[usable_cols]

    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(numeric_df)
    if x_imputed.shape[0] < 20:
        return {
            "status": "incompatible_data",
            "reason": "Insufficient rows (minimum 20 observations).",
        }

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    max_k = min(7, max(2, x_scaled.shape[0] // 10 + 1))
    k_range = range(2, max_k + 1)
    bic_scores = {}
    aic_scores = {}
    gmm_models = {}

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=42)
        gmm.fit(x_scaled)
        bic_scores[k] = float(gmm.bic(x_scaled))
        aic_scores[k] = float(gmm.aic(x_scaled))
        gmm_models[k] = gmm

    best_k = min(bic_scores, key=bic_scores.get)
    best_gmm = gmm_models[best_k]
    labels = best_gmm.predict(x_scaled)
    probs = best_gmm.predict_proba(x_scaled)

    eps = 1e-10
    entropy = -np.sum(probs * np.log(probs + eps), axis=1).mean()
    try:
        sil = float(silhouette_score(x_scaled, labels))
    except Exception:
        sil = None

    numeric_df_reset = pd.DataFrame(x_imputed, columns=numeric_df.columns).reset_index(drop=True)
    uniq, cnts = np.unique(labels, return_counts=True)
    class_sizes = {int(k): int(v) for k, v in zip(uniq, cnts)}
    class_profiles = {}
    for cls in sorted(class_sizes.keys()):
        mask = labels == cls
        n_cls = int(mask.sum())
        size_pct = round(n_cls / len(labels) * 100, 1)
        profile = {
            col: round(float(numeric_df_reset.loc[mask, col].mean()), 3)
            for col in numeric_df_reset.columns[:15]
        }
        class_profiles[f"class_{cls}"] = {
            "n": n_cls,
            "share_pct": size_pct,
            "mean_probability": round(float(probs[mask, cls].mean()), 3),
            "profile": profile,
        }

    class_means = pd.DataFrame(
        {
            cls: [numeric_df_reset.loc[labels == cls, col].mean() for col in numeric_df_reset.columns]
            for cls in range(best_k)
        },
        index=numeric_df_reset.columns,
    )
    discriminating_vars = class_means.std(axis=1).sort_values(ascending=False)

    result = {
        "status": "ok",
        "method": "Gaussian Mixture Model (sklearn)",
        "optimal_k": int(best_k),
        "selection_criterion": "BIC (lower is better)",
        "bic_by_k": {str(k): round(v, 2) for k, v in bic_scores.items()},
        "aic_by_k": {str(k): round(v, 2) for k, v in aic_scores.items()},
        "silhouette_score": sil,
        "classification_entropy": round(float(entropy), 4),
        "entropy_note": "< 0.5 = good class separation; > 1.0 = overlapping classes",
        "n_observations": int(x_imputed.shape[0]),
        "n_indicators": int(numeric_df.shape[1]),
        "class_sizes": class_sizes,
        "class_profiles": class_profiles,
        "top_discriminating_variables": {
            str(k): float(v) for k, v in discriminating_vars.head(10).to_dict().items()
        },
        "note": (
            "Compare with K-Means (typology) and CAH. "
            "GMM assigns soft probabilistic membership - use mean_probability to flag "
            "ambiguous respondents. "
            "For binary/ordinal items, a true LCA (poLCA in R) gives sharper results."
        ),
    }

    max_val = numeric_df_reset.max().max()
    min_val = numeric_df_reset.min().min()
    looks_like_likert = (
        min_val >= 0
        and max_val <= 10
        and numeric_df.dtypes.apply(lambda d: np.issubdtype(d, np.integer)).any()
    )

    if looks_like_likert and STATSMODELS_AVAILABLE:
        result["data_type_detected"] = "Likert/ordinal codes"
        result["recommendation"] = (
            "Data looks like survey codes (0-10 range). "
            "For true Latent Class Analysis on categorical items, "
            "consider using R::poLCA or Python::lca package. "
            "GMM results above are valid but treat responses as continuous."
        )
    else:
        result["data_type_detected"] = "continuous/mixed"

    return result


def run_forecasting(df: pd.DataFrame, mapping: dict) -> dict:
    """Run basic linear trend forecasting when wave/timestamp data exists."""
    print("Running Forecasting...")
    lower_cols = {col.lower(): col for col in df.columns}

    time_col = None
    if "wave" in lower_cols:
        time_col = lower_cols["wave"]
    else:
        for col in df.columns:
            name = col.lower()
            if "time" in name or "date" in name or "timestamp" in name:
                time_col = col
                break

    if time_col is None:
        return {
            "status": "incompatible_data",
            "required_structure": "Column 'wave' or a timestamp/date column plus at least one numeric metric",
            "columns_found": df.columns.tolist(),
            "recommendation": "Add a wave/date column and rerun forecasting",
        }

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns.tolist() if col != time_col
    ]
    if not numeric_cols:
        return {
            "status": "incompatible_data",
            "required_structure": "Column 'wave' or timestamp plus at least one numeric metric",
            "columns_found": df.columns.tolist(),
            "recommendation": "Add numeric KPI columns for trend modeling",
        }

    target_col = numeric_cols[0]
    work_df = df[[time_col, target_col]].dropna().copy()

    if work_df.empty or len(work_df) < 3:
        return {
            "status": "incompatible_data",
            "required_structure": "At least 3 non-null observations for wave/date and numeric metric",
            "columns_found": df.columns.tolist(),
            "recommendation": "Collect more waves before forecasting",
        }

    if time_col.lower() == "wave":
        work_df["_time_numeric"] = pd.to_numeric(work_df[time_col], errors="coerce")
    else:
        parsed = pd.to_datetime(work_df[time_col], errors="coerce")
        work_df["_time_numeric"] = (parsed - parsed.min()).dt.days

    work_df = work_df.dropna(subset=["_time_numeric", target_col])
    if len(work_df) < 3:
        return {
            "status": "incompatible_data",
            "required_structure": "Usable numeric wave/date index and metric values",
            "columns_found": df.columns.tolist(),
            "recommendation": "Ensure wave/date values are parseable",
        }

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        work_df["_time_numeric"], work_df[target_col]
    )

    x_vals = work_df["_time_numeric"].to_numpy()
    y_hat = intercept + slope * x_vals
    ci_band = 1.96 * std_err if np.isfinite(std_err) else None

    return {
        "status": "ok",
        "time_column": time_col,
        "target_column": target_col,
        "method": "scipy.stats.linregress",
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "confidence_band_95": float(ci_band) if ci_band is not None else None,
        "fitted_values": [float(v) for v in y_hat.tolist()],
    }


def run_statistical_skills(df: pd.DataFrame, config: dict) -> dict:
    """Execute enabled statistical_engine skills from registry."""
    root_dir = Path(__file__).parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    try:
        from skills.skill_loader import apply_skills

        return apply_skills("statistical_engine", df, config)
    except Exception as exc:
        return {"status": "skill_loader_unavailable", "error": str(exc)}


def apply_labels_to_results(results: dict, mapping: dict) -> dict:
    """Apply label metadata marker to statistical results."""
    labeled_results = results.copy()
    labeled_results["_label_mapping_applied"] = True
    labeled_results["_mapping_timestamp"] = datetime.now().isoformat()
    return labeled_results


def main():
    """Main statistical analysis workflow."""
    import io

    if (
        sys.stdout.encoding
        and sys.stdout.encoding.lower() not in ("utf-8", "utf8")
        and hasattr(sys.stdout, "buffer")
    ):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 70)
    print("MODULE 2: STATISTICAL ENGINE")
    print("=" * 70)

    config = load_config()
    df, mapping = load_data()

    print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Modules: {', '.join(config['project_metadata']['selected_modules'])}")

    config["run_info"]["status"] = "analyzing"
    config["run_info"]["analysis_started_at"] = datetime.now().isoformat()
    save_config(config)

    modules_to_run = config["project_metadata"]["selected_modules"]
    all_results: Dict[str, Any] = {}

    module_functions = {
        "descriptive": run_descriptive,
        "pca": run_pca,
        "anova": run_anova,
        "ols": run_ols,
        "typology": run_typology,
        "cah": run_cah,
        "lda": run_lda,
        "decision_tree": run_decision_tree,
        "latent_class": run_latent_class,
        "modeling": run_modeling,
        "regression": run_regression,
        "cbc": run_cbc,
        "cbc_simulator": run_cbc_simulator,
        "cas_vignettes": run_cas_vignettes,
        "forecasting": run_forecasting,
    }

    for module_name in modules_to_run:
        if module_name not in module_functions:
            all_results[module_name] = {"error": f"Unknown module: {module_name}"}
            continue

        try:
            all_results[module_name] = module_functions[module_name](df, mapping)
        except Exception as exc:
            all_results[module_name] = {"error": str(exc)}

    all_results["skills_registry"] = run_statistical_skills(df, config)

    labeled_results = apply_labels_to_results(all_results, mapping)

    final_results = {
        "project_metadata": config["project_metadata"],
        "run_info": config["run_info"],
        "results": labeled_results,
    }

    save_results(final_results)

    config["run_info"]["status"] = "analyzed"
    config["run_info"]["modules_completed"] = list(all_results.keys())
    config["run_info"]["analysis_completed_at"] = datetime.now().isoformat()
    save_config(config)

    print("\n" + "=" * 70)
    print("STATISTICAL ENGINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
