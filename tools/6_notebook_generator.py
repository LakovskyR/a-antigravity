"""
Notebook Generator - Module 6.
Generates an executable .ipynb for data scientist exploration.
"""

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def load_context() -> dict:
    """Load all available context from tmp/."""
    tmp = Path(__file__).parent.parent / "tmp"
    ctx: dict[str, Any] = {}

    config_path = tmp / "run_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            ctx["config"] = json.load(f)

    schema_path = tmp / "project_schema.json"
    if schema_path.exists():
        with open(schema_path, encoding="utf-8") as f:
            ctx["schema"] = json.load(f)

    brief_path = tmp / "analytics_brief.json"
    if brief_path.exists():
        with open(brief_path, encoding="utf-8") as f:
            ctx["brief"] = json.load(f)

    return ctx


def ask_setup(
    ctx: dict,
    non_interactive: bool = False,
    goal: str | None = None,
    target: str | None = None,
    k: int | None = None,
) -> dict:
    """
    Interactive or CLI setup questions.
    Returns setup dict with: goal, target_variable, k_segments, modules.
    """
    goals = {
        "1": ("segmentation", "Segmentation - find patient/prescriber profiles"),
        "2": ("drivers", "Driver analysis - what drives a key outcome"),
        "3": ("descriptive", "Descriptive only - frequencies and crosstabs"),
        "4": ("conjoint", "Conjoint/CBC - product preference and WTP"),
        "5": ("full", "Full analysis - run everything"),
    }

    goal_modules = {
        "segmentation": ["descriptive", "pca", "typology", "cah", "lda", "latent_class"],
        "drivers": ["descriptive", "pca", "ols", "modeling", "decision_tree", "anova"],
        "descriptive": ["descriptive", "anova"],
        "conjoint": ["descriptive", "cbc", "cbc_simulator", "cas_vignettes"],
        "full": [
            "descriptive",
            "pca",
            "anova",
            "ols",
            "typology",
            "cah",
            "lda",
            "latent_class",
            "modeling",
            "decision_tree",
            "cbc",
            "cbc_simulator",
        ],
    }

    if non_interactive:
        selected_goal = goal or "segmentation"
        modules = goal_modules.get(selected_goal, goal_modules["segmentation"])
        confirmed = ctx.get("brief", {}).get("confirmed_modules")
        if isinstance(confirmed, list) and confirmed:
            merged = []
            seen = set()
            for m in confirmed + modules:
                if m not in seen:
                    merged.append(m)
                    seen.add(m)
            modules = merged
        return {
            "goal": selected_goal,
            "target_variable": target or "auto",
            "k_segments": k or 4,
            "modules": modules,
        }

    print("\n" + "=" * 60)
    print("ANTIGRAVITY - Notebook Generator")
    print("=" * 60)
    print("\nWhat do you want to achieve?\n")
    for key, (_, label) in goals.items():
        print(f"  {key}. {label}")
    print()

    while True:
        choice = input("Select goal (1-5): ").strip()
        if choice in goals:
            selected_goal, _ = goals[choice]
            break
        print("  Please enter 1-5.")

    modules = goal_modules[selected_goal]

    schema_questions = ctx.get("schema", {}).get("questions", {})
    numeric_vars = [
        q for q, info in schema_questions.items() if info.get("type") in ("numericlist", "numeric", "ranking", "grid")
    ]
    print(f"\nDetected numeric variables: {numeric_vars[:10]}")

    target_input = input("\nTarget variable for driver/OLS analysis (press Enter for auto-detect): ").strip()
    target_variable = target_input if target_input else "auto"

    k_input = input("\nNumber of segments for clustering (press Enter for 4): ").strip()
    k_segments = int(k_input) if k_input.isdigit() else 4

    print(f"\nSelected: goal={selected_goal}, k={k_segments}, target={target_variable}")
    print(f"Modules: {modules}")
    confirm = input("\nProceed? (Enter=yes, n=change): ").strip().lower()
    if confirm == "n":
        return ask_setup(ctx, non_interactive=False)

    return {
        "goal": selected_goal,
        "target_variable": target_variable,
        "k_segments": k_segments,
        "modules": modules,
    }


AGENT_MARKER = "# AGENT: comment this cell - interpret results for the analyst"


def md(text: str):
    return new_markdown_cell(text)


def code(src: str, agent_comment: bool = True):
    if agent_comment:
        src = AGENT_MARKER + "\n" + src
    return new_code_cell(src)


def cell_title(ctx: dict, setup: dict) -> list:
    project = ctx.get("config", {}).get("project_metadata", {}).get("project_name", "Survey")
    brief_ctx = ctx.get("brief", {}).get("project_context", "")
    goal_label = setup["goal"].replace("_", " ").title()
    ts = datetime.now().strftime("%Y-%m-%d")
    return [
        md(
            f"""# {project} - Exploratory Analysis
**Goal:** {goal_label}  
**Generated:** {ts}  
**Context:** {brief_ctx}

> This notebook was auto-generated by Antigravity.  
> Edit the **PARAMETERS** cell below to change any settings.  
> Run cells top to bottom. Ask your AI agent to comment any cell.
"""
        )
    ]


def cell_parameters(setup: dict) -> list:
    target = f'"{setup["target_variable"]}"' if setup["target_variable"] != "auto" else "None  # auto-detected"
    modules_str = json.dumps(setup["modules"], indent=4)
    return [
        md("## Parameters\nEdit all configuration here. Changes propagate through the notebook."),
        new_code_cell(
            f"""# == PARAMETERS - edit here ==================================
# Clustering
K_SEGMENTS       = {setup["k_segments"]}   # number of K-Means clusters
K_SEGMENTS_MAX   = 8    # max K to evaluate in auto-selection scan
CAH_MAX_K        = 6    # max k to evaluate in hierarchical clustering
LATENT_K_MAX     = 6    # max k for Gaussian Mixture Model

# PCA
PCA_VARIANCE_TARGET = 0.80  # keep components explaining this % of variance
PCA_MAX_COMPONENTS  = 10

# Driver analysis
TARGET_VARIABLE  = {target}
SIGNIFICANCE     = 0.05     # p-value threshold for significance

# Modeling
N_TREES          = 100      # RandomForest / XGBoost n_estimators
TEST_SIZE        = 0.20     # train/test split ratio

# General
RANDOM_STATE     = 42
MIN_SEGMENT_SIZE = 10       # drop segments smaller than this
TOP_N_FEATURES   = 10       # how many features to show in charts

# Market positioning
TARGET_BRAND        = "A"          # brand to highlight
AWARENESS_PREFIX    = "Awareness_"
SATISFACTION_PREFIX = "Satisfaction_"

# Active modules
MODULES = {modules_str}

# Data paths
DATA_PATH   = "../tmp/cleaned_codes.csv"
SCHEMA_PATH = "../tmp/project_schema.json"
PCA_PATH    = "../tmp/pca_coordinates.csv"
"""
        ),
    ]


def cell_setup() -> list:
    return [
        md("## 0. Setup"),
        new_code_cell(
            """import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY = True
except ImportError:
    PLOTLY = False

try:
    import shap
    SHAP = True
except ImportError:
    SHAP = False

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid")

df = pd.read_csv(DATA_PATH)
with open(SCHEMA_PATH, encoding="utf-8") as f:
    schema = json.load(f)

print(f"Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Schema questions: {len(schema.get('questions', {}))}")
"""
        ),
    ]


def cell_overview() -> list:
    return [
        md("## 1. Data Overview"),
        code(
            """print(df.dtypes.value_counts())
print("\\nMissing values per column:")
missing = df.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0].round(3).to_string())
df.head(5)
"""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(16, 4))
sns.heatmap(
    df.isnull().T, cbar=False, ax=ax, xticklabels=False, yticklabels=True, cmap="YlOrRd"
)
ax.set_title("Missing Values Map (yellow = missing)")
plt.tight_layout()
plt.show()
"""
        ),
    ]


def cell_descriptive(schema: dict) -> list:
    question_map = json.dumps(
        {k: v.get("label", k)[:50] for k, v in list(schema.get("questions", {}).items())[:30]},
        indent=4,
        ensure_ascii=False,
    )
    return [
        md("## 2. Descriptive Statistics"),
        code(
            f"""QUESTION_LABELS = {question_map}

def label(col):
    return QUESTION_LABELS.get(col, col)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
print(f"Numeric columns: {{len(numeric_cols)}}")
df[numeric_cols].describe().round(2)
"""
        ),
        code(
            """means = df[numeric_cols].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(14, max(4, len(means) * 0.3)))
means.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(means)))
ax.set_title("Mean score per variable")
ax.set_xlabel("Mean")
plt.tight_layout()
plt.show()
"""
        ),
        code(
            """corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
ax.set_title("Variable Correlation Matrix")
plt.tight_layout()
plt.show()
"""
        ),
    ]


def cell_market_positioning(schema: dict) -> list:
    return [
        md(
            """## Market Positioning Analysis
Brand positioning map (Awareness x Satisfaction) with 95% CI error bars.
Competitive attribute gap: Brand A vs market average.
Configure TARGET_BRAND and AWARENESS_PREFIX / SATISFACTION_PREFIX in PARAMETERS.
"""
        ),
        code(
            """from scipy.stats import sem, t as t_dist

def ci95(series):
    n = len(series.dropna())
    if n < 2:
        mean = float(series.mean()) if len(series) else np.nan
        return mean, mean, mean
    mean = float(series.mean())
    margin = float(sem(series.dropna()) * t_dist.ppf(0.975, n - 1))
    return mean, mean - margin, mean + margin

aw_cols = [c for c in df.columns if c.lower().startswith(AWARENESS_PREFIX.lower())]
sat_cols = [c for c in df.columns if c.lower().startswith(SATISFACTION_PREFIX.lower())]

if not aw_cols or not sat_cols:
    print("No awareness/satisfaction columns detected.")
    print("Set AWARENESS_PREFIX and SATISFACTION_PREFIX in PARAMETERS to match your column names.")
else:
    positions = []
    for aw_col in aw_cols:
        suffix = aw_col[len(AWARENESS_PREFIX):]
        sat_col = SATISFACTION_PREFIX + suffix
        if sat_col not in df.columns:
            continue
        aw_mean, aw_low, aw_high = ci95(df[aw_col])
        sat_mean, sat_low, sat_high = ci95(df[sat_col])
        positions.append(
            {
                "brand": suffix,
                "awareness": aw_mean,
                "aw_low": aw_low,
                "aw_high": aw_high,
                "satisfaction": sat_mean,
                "sat_low": sat_low,
                "sat_high": sat_high,
            }
        )
    pos_df = pd.DataFrame(positions)

    if pos_df.empty:
        print("Awareness/satisfaction prefixes found, but no aligned brand suffix pairs.")
    else:
        brand_col = next(
            (
                c for c in df.columns
                if "brand" in c.lower() or "prescri" in c.lower() or "recommend" in c.lower()
            ),
            None,
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        ax = axes[0]
        for _, row in pos_df.iterrows():
            is_target = str(row["brand"]).upper() == TARGET_BRAND.upper()
            color = "red" if is_target else "steelblue"
            size = 300 if is_target else 150
            ax.scatter(
                row["awareness"], row["satisfaction"], s=size, c=color, alpha=0.7,
                edgecolors="black", linewidth=2, zorder=5
            )
            ax.errorbar(
                row["awareness"], row["satisfaction"],
                xerr=[[row["awareness"] - row["aw_low"]], [row["aw_high"] - row["awareness"]]],
                yerr=[[row["satisfaction"] - row["sat_low"]], [row["sat_high"] - row["satisfaction"]]],
                fmt="none", ecolor="gray", alpha=0.4, capsize=4
            )
            ax.annotate(
                row["brand"], (row["awareness"], row["satisfaction"]),
                fontsize=10, fontweight="bold" if is_target else "normal",
                ha="center", va="bottom"
            )

        ax.axhline(pos_df["satisfaction"].mean(), color="gray", linestyle="--", alpha=0.4)
        ax.axvline(pos_df["awareness"].mean(), color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("Awareness Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("Satisfaction Score", fontsize=12, fontweight="bold")
        ax.set_title("Brand Positioning Map (95% CI)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        if brand_col:
            brand_counts = df[brand_col].value_counts()
            shares = brand_counts / len(df) * 100
            colors_bar = [
                "red" if str(b).upper() == TARGET_BRAND.upper() else "lightsteelblue"
                for b in shares.index
            ]
            bars = ax2.barh(shares.index.astype(str), shares.values, color=colors_bar, edgecolor="black")
            for bar, val in zip(bars, shares.values):
                ax2.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontweight="bold")
            ax2.set_xlabel("Market Share (%)", fontweight="bold")
            ax2.set_title("Prescription Market Share", fontsize=14, fontweight="bold")
        else:
            ax2.text(
                0.5, 0.5, "No brand/prescription column detected\\nSet brand_col manually",
                ha="center", va="center", transform=ax2.transAxes
            )
        plt.tight_layout()
        plt.show()
"""
        ),
        code(
            """attr_target_cols = [
    c for c in df.columns
    if c.endswith(f"_{TARGET_BRAND.upper()}") and not any(c.startswith(p) for p in [AWARENESS_PREFIX, SATISFACTION_PREFIX])
]
attr_all_cols = {}

for col in attr_target_cols:
    base = col[: col.rfind("_")]
    market_cols = [c for c in df.columns if c.startswith(base) and c != col]
    if market_cols:
        market_avg = df[market_cols].mean(axis=1).mean()
        target_avg = df[col].mean()
        attr_all_cols[base] = {
            "target": target_avg,
            "market": market_avg,
            "gap": target_avg - market_avg,
        }

if attr_all_cols:
    gap_df = pd.DataFrame(attr_all_cols).T.sort_values("gap", ascending=False)
    fig, ax = plt.subplots(figsize=(12, max(4, len(gap_df) * 0.5)))
    colors_gap = ["green" if g > 0 else "red" for g in gap_df["gap"]]
    ax.barh(gap_df.index, gap_df["gap"], color=colors_gap, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Gap vs Market Average", fontweight="bold")
    ax.set_title(f"{TARGET_BRAND} Competitive Attribute Gap (positive = advantage)", fontweight="bold")
    plt.tight_layout()
    plt.show()
    print("Strengths (top 3):", gap_df.head(3).index.tolist())
    print("Weaknesses (bottom 3):", gap_df.tail(3).index.tolist())
else:
    print("No brand attribute columns detected matching pattern {base}_{TARGET_BRAND}")
"""
        ),
    ]


def cell_pca() -> list:
    return [
        md(
            """## 3. PCA - Dimensionality Reduction
PCA reduces the question space to uncorrelated axes. Use the scree plot to
validate component selection. Top loadings show which questions drive each axis.
"""
        ),
        code(
            """from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = df.select_dtypes(include="number").drop(columns=["respondent_id"], errors="ignore").fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X_scaled)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_comp = int(np.searchsorted(cumvar, PCA_VARIANCE_TARGET)) + 1
n_comp = min(n_comp, PCA_MAX_COMPONENTS, X.shape[1])
print(f"Selected {n_comp} components explaining {cumvar[n_comp-1]*100:.1f}% of variance")

pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_comp)])
pca_df.to_csv(PCA_PATH, index=False)
print(f"PCA coordinates saved to {PCA_PATH}")
"""
        ),
        code(
            """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(range(1, len(pca_full.explained_variance_ratio_)+1), pca_full.explained_variance_ratio_ * 100, color="#4C72B0")
ax1.axvline(n_comp + 0.5, color="red", linestyle="--", label=f"Selected: {n_comp}")
ax1.set_xlabel("Component")
ax1.set_ylabel("Variance explained (%)")
ax1.set_title("Scree Plot")
ax1.legend()

ax2.plot(range(1, len(cumvar)+1), cumvar * 100, marker="o", color="#4C72B0")
ax2.axhline(PCA_VARIANCE_TARGET * 100, color="red", linestyle="--", label=f"{PCA_VARIANCE_TARGET*100:.0f}% target")
ax2.axvline(n_comp + 0.5, color="orange", linestyle="--")
ax2.set_xlabel("Components")
ax2.set_ylabel("Cumulative variance (%)")
ax2.set_title("Cumulative Variance")
ax2.legend()

plt.tight_layout()
plt.show()
"""
        ),
        code(
            """loadings = pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=[f"PC{i+1}" for i in range(n_comp)],
)
top_vars = loadings.abs().max(axis=1).sort_values(ascending=False).head(20).index
fig, ax = plt.subplots(figsize=(n_comp * 1.5 + 2, 8))
sns.heatmap(loadings.loc[top_vars], annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
ax.set_title("PCA Loadings - Top 20 Variables")
plt.tight_layout()
plt.show()
"""
        ),
    ]


def cell_pca_biplot() -> list:
    return [
        md("### PCA Biplot - Correlation Circle + Segment Projection"),
        code(
            """from matplotlib.patches import Ellipse

if X_pca.shape[1] < 2:
    print("Need at least 2 PCA components for biplot.")
else:
    loadings_biplot = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_matrix = loadings_biplot[:, :2]

    feature_importance_pca = np.sum(loading_matrix ** 2, axis=1)
    top_idx = np.argsort(feature_importance_pca)[-15:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    circle = plt.Circle((0, 0), 1, color="navy", fill=False, linestyle="--", linewidth=2, alpha=0.5)
    ax1.add_patch(circle)
    for r in [0.25, 0.5, 0.75]:
        ax1.add_patch(plt.Circle((0, 0), r, color="gray", fill=False, linestyle=":", linewidth=0.5, alpha=0.3))
    ax1.axhline(0, color="black", linewidth=1, alpha=0.3)
    ax1.axvline(0, color="black", linewidth=1, alpha=0.3)

    colors_pca = plt.cm.viridis(np.linspace(0, 1, len(top_idx)))
    feature_names_display = X.columns.tolist()
    for feat_idx, color in zip(top_idx, colors_pca):
        x_l, y_l = loading_matrix[feat_idx]
        ax1.arrow(0, 0, x_l, y_l, head_width=0.04, head_length=0.04, fc=color, ec=color, linewidth=2, alpha=0.8)
        label = str(feature_names_display[feat_idx])[:25]
        ax1.text(
            x_l * 1.18, y_l * 1.18, label, fontsize=8,
            ha="left" if x_l > 0 else "right",
            va="bottom" if y_l > 0 else "top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.4)
        )

    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=13, fontweight="bold")
    ax1.set_title("Correlation Circle - Feature Contributions", fontsize=14, fontweight="bold")
    ax1.set_aspect("equal")
    ax1.text(
        0.02,
        0.98,
        "Long vector = high contribution\\nClose vectors = correlated\\nOpposite = negative corr",
        transform=ax1.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    if "segment" in df.columns:
        scatter = ax2.scatter(
            X_pca[:, 0], X_pca[:, 1], c=df["segment"], cmap="viridis", s=60, alpha=0.6, edgecolors="black", linewidth=0.5
        )
        for seg in sorted(df["segment"].unique()):
            pts = X_pca[df["segment"] == seg][:, :2]
            if len(pts) < 3:
                continue
            mean = pts.mean(axis=0)
            ax2.scatter(*mean, s=500, marker="*", c="red", edgecolors="black", linewidth=2, zorder=10)
            ax2.text(
                mean[0], mean[1] + 0.3, f"Seg {seg}", fontsize=13, fontweight="bold",
                ha="center", bbox=dict(boxstyle="round", facecolor="red", alpha=0.3)
            )
            cov = np.cov(pts.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 1][::-1]))
            w, h = 4 * np.sqrt(np.maximum(eigenvalues, 1e-9))
            ellipse = Ellipse(
                mean, w, h, angle=angle, facecolor="none",
                edgecolor=plt.cm.viridis(seg / max(df["segment"].max(), 1)),
                linewidth=2.5, linestyle="--", alpha=0.8
            )
            ax2.add_patch(ellipse)
        plt.colorbar(scatter, ax=ax2, label="Segment")
    else:
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=40)

    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=13, fontweight="bold")
    ax2.set_title("Respondent Projection + 2-sigma Confidence Ellipses", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
"""
        ),
    ]


def cell_segmentation(k: int) -> list:
    return [
        md(f"## 4. Segmentation\nK-Means on PCA coordinates. Default k={k} - change **K_SEGMENTS** in Parameters."),
        code(
            """from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

try:
    X_seg = pd.read_csv(PCA_PATH).values
    print("Using PCA coordinates for clustering")
except FileNotFoundError:
    X_seg = df.select_dtypes(include="number").fillna(0).values
    print("PCA file not found - using raw data")

# -- K Selection: 4-metric consensus -------------------------------------
k_range = range(2, min(K_SEGMENTS_MAX + 1, len(X_seg) // 10 + 2))
metrics_k = {"silhouette": [], "davies_bouldin": [], "calinski_harabasz": [], "inertia": []}

for k in k_range:
    model_k = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_k = model_k.fit_predict(X_seg)
    metrics_k["silhouette"].append(silhouette_score(X_seg, labels_k))
    metrics_k["davies_bouldin"].append(davies_bouldin_score(X_seg, labels_k))
    metrics_k["calinski_harabasz"].append(calinski_harabasz_score(X_seg, labels_k))
    metrics_k["inertia"].append(model_k.inertia_)

k_list = list(k_range)
k_sil = k_list[np.argmax(metrics_k["silhouette"])] if k_list else K_SEGMENTS
k_db = k_list[np.argmin(metrics_k["davies_bouldin"])] if k_list else K_SEGMENTS
k_ch = k_list[np.argmax(metrics_k["calinski_harabasz"])] if k_list else K_SEGMENTS
inertias = np.array(metrics_k["inertia"])
if len(inertias) >= 3:
    k_elbow = k_list[np.argmax(np.abs(np.diff(np.diff(inertias)))) + 2]
else:
    k_elbow = k_sil

k_recommended = int(np.median([k_sil, k_db, k_ch, k_elbow]))
print(
    f"K recommendations -> Silhouette: {k_sil}, Davies-Bouldin: {k_db}, "
    f"Calinski-Harabasz: {k_ch}, Elbow: {k_elbow}"
)
print(f"Consensus K (median): {k_recommended}")
print(f"K_SEGMENTS parameter: {K_SEGMENTS}  <- override consensus if you disagree")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plot_configs = [
    ("silhouette", axes[0, 0], "Silhouette Score", "max", k_sil, "bo-"),
    ("davies_bouldin", axes[0, 1], "Davies-Bouldin Index", "min", k_db, "go-"),
    ("calinski_harabasz", axes[1, 0], "Calinski-Harabasz Score", "max", k_ch, "mo-"),
    ("inertia", axes[1, 1], "Inertia (Elbow)", "elbow", k_elbow, "ro-"),
]
for key, ax, title, direction, k_best, style in plot_configs:
    ax.plot(k_list, metrics_k[key], style, linewidth=2, markersize=8)
    ax.axvline(x=k_best, color="red", linestyle="--", alpha=0.7, label=f"Best K={k_best}")
    ax.set_xlabel("K", fontweight="bold")
    ax.set_ylabel(title, fontweight="bold")
    ax.set_title(f"{title} ({direction} = better)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
plt.suptitle(f"K Selection Validation - Consensus K={k_recommended}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Use consensus unless analyst changed default value manually
K_FINAL = K_SEGMENTS if K_SEGMENTS != 4 else k_recommended
print(f"Using K={K_FINAL} for final clustering")
"""
        ),
        code(
            """kmeans = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=10)
df["segment"] = kmeans.fit_predict(X_seg)

sizes = df["segment"].value_counts().sort_index()
print("Segment sizes:")
print(sizes.to_string())

small = sizes[sizes < MIN_SEGMENT_SIZE].index.tolist()
if small:
    print(f"Segments {small} have < {MIN_SEGMENT_SIZE} respondents - consider reducing K_SEGMENTS")
"""
        ),
        code(
            """numeric_cols = df.select_dtypes(include="number").drop(columns=["respondent_id", "segment"], errors="ignore").columns.tolist()
profile = df.groupby("segment")[numeric_cols].mean()
profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

fig, ax = plt.subplots(figsize=(min(len(numeric_cols), 20), K_FINAL + 1))
sns.heatmap(
    profile_norm.T.head(25),
    annot=False,
    cmap="YlOrRd",
    ax=ax,
    linewidths=0.5,
    xticklabels=[f"Seg {i}" for i in profile_norm.index],
)
ax.set_title("Segment Profiles (normalized means)")
plt.tight_layout()
plt.show()
profile.round(2)
"""
        ),
        code(
            """if X_seg.shape[1] >= 2:
    n_seg = int(df["segment"].nunique())
    palette = sns.color_palette("tab10", n_seg)
    fig, ax = plt.subplots(figsize=(10, 7))
    for seg in sorted(df["segment"].unique()):
        mask = df["segment"] == seg
        ax.scatter(
            X_seg[mask, 0],
            X_seg[mask, 1],
            label=f"Seg {seg} (n={mask.sum()})",
            color=palette[seg],
            alpha=0.6,
            s=40,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Segments in PCA Space")
    ax.legend()
    plt.tight_layout()
    plt.show()
"""
        ),
    ]


def cell_cah() -> list:
    return [
        md("## 4b. CAH - Hierarchical Clustering\nCompare dendrogram with K-Means k choice."),
        code(
            """from scipy.cluster.hierarchy import dendrogram, linkage

sample_size = min(200, len(X_seg))
X_sample = X_seg[:sample_size]
Z = linkage(X_sample, method="ward")

fig, ax = plt.subplots(figsize=(16, 5))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, show_leaf_counts=True, leaf_rotation=90)
ax.axhline(y=Z[-(K_SEGMENTS-1), 2], color="red", linestyle="--", label=f"Cut for k={K_SEGMENTS}")
ax.set_title("CAH Dendrogram (Ward linkage, sample n=200)")
ax.set_ylabel("Distance")
ax.legend()
plt.tight_layout()
plt.show()
"""
        ),
    ]


def cell_latent_class() -> list:
    return [
        md(
            """## 4c. Latent Class / Mixte
GMM finds probabilistic hidden segments. BIC selects optimal k.
Compare with K-Means: if GMM suggests a different k, investigate.
"""
        ),
        code(
            """from sklearn.mixture import GaussianMixture

bic_scores, aic_scores = {}, {}
k_range_lc = range(2, LATENT_K_MAX + 1)

for k in k_range_lc:
    gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=RANDOM_STATE)
    gmm.fit(X_seg)
    bic_scores[k] = gmm.bic(X_seg)
    aic_scores[k] = gmm.aic(X_seg)

best_k_gmm = min(bic_scores, key=bic_scores.get)
print(f"GMM optimal k by BIC: {best_k_gmm} (K_SEGMENTS = {K_SEGMENTS})")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(list(bic_scores.keys()), list(bic_scores.values()), marker="o", label="BIC", color="#4C72B0")
ax.plot(list(aic_scores.keys()), list(aic_scores.values()), marker="s", label="AIC", color="#DD8452")
ax.axvline(best_k_gmm, color="red", linestyle="--", label=f"BIC optimal k={best_k_gmm}")
ax.set_xlabel("k")
ax.set_ylabel("Score (lower = better)")
ax.set_title("GMM - BIC / AIC by k")
ax.legend()
plt.tight_layout()
plt.show()
"""
        ),
        code(
            """gmm_final = GaussianMixture(n_components=best_k_gmm, covariance_type="full", n_init=10, random_state=RANDOM_STATE)
gmm_final.fit(X_seg)
df["latent_class"] = gmm_final.predict(X_seg)
probs = gmm_final.predict_proba(X_seg)

eps = 1e-10
entropy = -(probs * np.log(probs + eps)).sum(axis=1).mean()
print(f"Classification entropy: {entropy:.3f} (< 0.5 = clean separation)")
df.groupby("latent_class")[numeric_cols[:10]].mean().round(2)
"""
        ),
    ]


def cell_lda() -> list:
    return [
        md("## 5. Segment Discrimination - LDA"),
        code(
            """from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X_lda = df.select_dtypes(include="number").drop(columns=["respondent_id", "segment", "latent_class"], errors="ignore").fillna(0)
y_lda = df["segment"] if "segment" in df.columns else df.iloc[:, 0]

lda = LinearDiscriminantAnalysis(n_components=min(K_SEGMENTS - 1, 3))
X_ld = lda.fit_transform(X_lda.values, y_lda)

print(f"Explained variance by LD axes: {[f'{v*100:.1f}%' for v in lda.explained_variance_ratio_]}")

loadings_lda = pd.Series(lda.scalings_[:, 0], index=X_lda.columns).abs().sort_values(ascending=False).head(TOP_N_FEATURES)
fig, ax = plt.subplots(figsize=(10, 5))
loadings_lda.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("Top Discriminating Variables - LD1")
ax.set_xlabel("|Loading|")
plt.tight_layout()
plt.show()
"""
        ),
    ]


def cell_drivers(target_var: str) -> list:
    target_code = f'"{target_var}"' if target_var != "auto" else 'df.select_dtypes(include="number").columns[-1]'
    return [
        md("## 6. Driver Analysis"),
        code(
            f"""import statsmodels.api as sm

target_col = {target_code}
print(f"Target: {{target_col}}")

feature_cols = [
    c for c in df.select_dtypes(include="number").columns
    if c not in ("respondent_id", "segment", "latent_class", target_col)
]

X_ols = df[feature_cols].fillna(df[feature_cols].median())
y_ols = df[target_col].fillna(df[target_col].median())
var_mask = X_ols.var() > 0.01
X_ols = X_ols.loc[:, var_mask]

X_const = sm.add_constant(X_ols)
model = sm.OLS(y_ols, X_const).fit()

print(f"R2 = {{model.rsquared:.3f}}, adj R2 = {{model.rsquared_adj:.3f}}")
print(f"F-statistic p-value = {{model.f_pvalue:.4f}}")

coef_df = pd.DataFrame({{
    "coef": model.params[1:],
    "p_value": model.pvalues[1:],
    "ci_low": model.conf_int().iloc[1:, 0],
    "ci_high": model.conf_int().iloc[1:, 1],
}})
coef_df["significant"] = coef_df["p_value"] < SIGNIFICANCE
coef_df = coef_df.sort_values("coef", key=abs, ascending=False)
coef_df.head(TOP_N_FEATURES).round(3)
"""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(10, max(5, len(coef_df.head(TOP_N_FEATURES)) * 0.5)))
y_pos = range(len(coef_df.head(TOP_N_FEATURES)))
sub = coef_df.head(TOP_N_FEATURES)

colors = ["#4C72B0" if s else "#C0C0C0" for s in sub["significant"]]
ax.barh(list(y_pos), sub["coef"], color=colors)
ax.errorbar(
    sub["coef"],
    list(y_pos),
    xerr=[sub["coef"] - sub["ci_low"], sub["ci_high"] - sub["coef"]],
    fmt="none",
    color="black",
    capsize=3,
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(sub.index)
ax.set_title("OLS Coefficients (blue = significant at p<0.05)")
plt.tight_layout()
plt.show()
"""
        ),
    ]


def cell_rf_shap() -> list:
    return [
        md("## 6b. Random Forest + SHAP Feature Importances"),
        code(
            """from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

y_rf = df[target_col].fillna(df[target_col].median())
if y_rf.nunique() != 2:
    y_rf = (y_rf > y_rf.median()).astype(int)

X_rf = df[feature_cols].fillna(df[feature_cols].median())
X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y_rf, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

rf = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test)))

importances = pd.Series(rf.feature_importances_, index=X_rf.columns)
importances = importances.sort_values(ascending=False).head(TOP_N_FEATURES)
fig, ax = plt.subplots(figsize=(10, 5))
importances.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(importances)))
ax.set_title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=RANDOM_STATE,
)
cv_scores = cross_val_score(gb, X_rf, y_rf, cv=5, scoring="roc_auc")
print(f"GradientBoosting CV AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
gb.fit(X_train, y_train)
"""
        ),
        code(
            """if SHAP:
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap.summary_plot(shap_values, X_test, max_display=TOP_N_FEATURES, show=True, plot_type="bar")
else:
    print("Install shap for SHAP plots: pip install shap")

# Propensity scores
df["propensity_score"] = rf.predict_proba(X_rf)[:, 1]
propensity_path = "../tmp/output_propensity_scores.csv"
df[["propensity_score"]].to_csv(propensity_path, index=False)
print(f"Propensity scores saved: {propensity_path}")

# ROC curve
from sklearn.metrics import roc_auc_score, roc_curve
fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    fpr,
    tpr,
    "r-",
    linewidth=2,
    label=f"RandomForest (AUC={roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.3f})",
)
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.fill_between(fpr, 0, tpr, alpha=0.15, color="red")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
plt.tight_layout()
plt.show()
"""
        ),
    ]


def cell_anova(schema: dict) -> list:
    cat_cols = [q for q, info in schema.get("questions", {}).items() if info.get("type") == "single"][:3]
    cat_str = json.dumps(cat_cols)
    return [
        md("## 7. ANOVA - Group Comparisons"),
        code(
            f"""from scipy.stats import f_oneway

GROUP_VARS = {cat_str}

numeric_cols_anova = df.select_dtypes(include="number").drop(columns=["respondent_id", "segment", "latent_class"], errors="ignore").columns.tolist()
results_anova = []
for group_col in GROUP_VARS:
    if group_col not in df.columns:
        continue
    groups = df[group_col].dropna().unique()
    for col in numeric_cols_anova[:15]:
        group_data = [df[df[group_col] == g][col].dropna().values for g in groups]
        group_data = [g for g in group_data if len(g) >= 3]
        if len(group_data) < 2:
            continue
        f_stat, p_val = f_oneway(*group_data)
        results_anova.append({{
            "group_var": group_col,
            "metric": col,
            "f_stat": round(f_stat, 3),
            "p_value": round(p_val, 4),
            "significant": p_val < SIGNIFICANCE
        }})

anova_df = pd.DataFrame(results_anova)
sig_anova = anova_df[anova_df["significant"]].sort_values("p_value")
print(f"Significant results: {{len(sig_anova)}} / {{len(anova_df)}}")
sig_anova.head(20)
"""
        ),
    ]


def cell_decision_tree() -> list:
    return [
        md("## 8. Decision Tree - Interpretable Rules"),
        code(
            """from sklearn.tree import DecisionTreeClassifier, plot_tree

y_tree = df[target_col].fillna(df[target_col].median())
if y_tree.nunique() > 5:
    y_tree = (y_tree > y_tree.median()).astype(int)
X_tree = df[feature_cols].fillna(df[feature_cols].median())

tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE)
tree.fit(X_tree, y_tree)
print(f"Accuracy: {tree.score(X_tree, y_tree):.3f}")
print(f"Leaves: {tree.get_n_leaves()}")

fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(tree, feature_names=X_tree.columns, class_names=True, filled=True, rounded=True, ax=ax, fontsize=8)
plt.tight_layout()
plt.show()
"""
        ),
        code(
            """from sklearn.tree import export_text
rules = export_text(tree, feature_names=list(X_tree.columns))
print(rules)
"""
        ),
    ]


def cell_summary() -> list:
    return [
        md("## Summary & Conclusions"),
        code(
            """summary = {}

if "segment" in df.columns:
    summary["n_segments_kmeans"] = int(K_SEGMENTS)
    summary["segment_sizes"] = df["segment"].value_counts().to_dict()

if "latent_class" in df.columns:
    summary["n_classes_gmm"] = int(df["latent_class"].nunique())

if "model" in dir():
    summary["ols_r2"] = round(model.rsquared, 3)
    n_sig = (model.pvalues[1:] < SIGNIFICANCE).sum()
    summary["ols_significant_drivers"] = int(n_sig)

if "rf" in dir():
    summary["rf_accuracy"] = round(rf.score(X_test, y_test), 3)

for k, v in summary.items():
    print(f"  {k}: {v}")
"""
        ),
    ]


def build_notebook(ctx: dict, setup: dict) -> nbformat.NotebookNode:
    modules = setup["modules"]
    schema = ctx.get("schema", {})
    k = setup["k_segments"]
    target = setup["target_variable"]

    cells = []
    cells += cell_title(ctx, setup)
    cells += cell_parameters(setup)
    cells += cell_setup()
    cells += cell_overview()
    cells += cell_descriptive(schema)
    # Always add market positioning; it auto-detects relevant columns and skips gracefully if absent.
    cells += cell_market_positioning(schema)

    if "pca" in modules:
        cells += cell_pca()
    if "pca" in modules and ("typology" in modules or "cah" in modules):
        cells += cell_pca_biplot()

    seg_modules = {"typology", "cah", "latent_class"} & set(modules)
    if seg_modules:
        cells += cell_segmentation(k)
    if "cah" in modules:
        cells += cell_cah()
    if "latent_class" in modules:
        cells += cell_latent_class()
    if "lda" in modules:
        cells += cell_lda()

    driver_modules = {"ols", "regression", "modeling"} & set(modules)
    if driver_modules:
        cells += cell_drivers(target)
    if "modeling" in modules:
        cells += cell_rf_shap()

    if "anova" in modules:
        cells += cell_anova(schema)
    if "decision_tree" in modules:
        cells += cell_decision_tree()

    cells += cell_summary()

    nb = new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
    return nb


def main():
    if (
        sys.stdout.encoding
        and sys.stdout.encoding.lower() not in ("utf-8", "utf8")
        and hasattr(sys.stdout, "buffer")
    ):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Antigravity Notebook Generator")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument(
        "--goal",
        default=None,
        choices=["segmentation", "drivers", "descriptive", "conjoint", "full"],
    )
    parser.add_argument("--target", default=None, help="Target variable name")
    parser.add_argument("--k", type=int, default=None, help="Number of segments")
    args = parser.parse_args()

    ctx = load_context()
    setup = ask_setup(
        ctx,
        non_interactive=args.non_interactive,
        goal=args.goal,
        target=args.target,
        k=args.k,
    )
    nb = build_notebook(ctx, setup)

    project = ctx.get("config", {}).get("project_metadata", {}).get("project_name", "survey")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent.parent / "tmp" / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{project}_{setup['goal']}_{ts}.ipynb"

    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"\nNotebook generated: {out_path}")
    print(f"   Goal:     {setup['goal']}")
    print(f"   Modules:  {setup['modules']}")
    print(f"   Segments: {setup['k_segments']}")
    print(f"\nOpen with: jupyter lab {out_path}")
    print("Or run:    jupyter nbconvert --to notebook --execute " f"{out_path} --output {out_path}")


if __name__ == "__main__":
    main()
