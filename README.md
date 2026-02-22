# Untitled

# Project Antigravity: Forsta Survey Analytics Engine

## 🎯 Overview

Project Antigravity is a modular, analyst-first automation engine that transforms raw Forsta survey exports (~300 columns, patient & physician studies) into clean statistical deliverables. It handles the heavy lifting — ingestion, cleaning, modeling, commentary — so your data scientists and Power BI developers can focus entirely on design and client presentation.

An analyst opens Antigravity, points it at a SharePoint survey folder, selects a workflow (full pipeline or specific module), and receives a ready-to-use deliverable in their format of choice.

## Environment Setup (required - Python 3.12 for SHAP compatibility)

```
uv venv --python 3.12
uv sync
```

> Do NOT use Python 3.13+ - SHAP has a known incompatibility.
> 

## 🛡️ Core Principles

1. **RGPD Compliance & Privacy by Design:** All math runs on anonymized survey **codes**. Human-readable **labels** are applied only at final delivery. No PII or raw survey data ever reaches an LLM API — only aggregated statistical outputs.
2. **Deterministic Processing:** LLMs are forbidden from performing data manipulation, math, or aggregations. Python (Pandas/Scikit-Learn) handles all transformations. LLMs are used exclusively for code generation, error assistance, and text commentary.
3. **Analyst-First UX:** The system exposes a menu-driven interface. Analysts select a survey folder, pick a workflow or individual module, and let the engine run. No scripting required.
4. **B.L.A.S.T. Protocol:** Architecture follows Blueprint → Link → Architect → Stylize → Trigger, ensuring modular, testable, self-healing code.
5. **Versioning & Traceability:** Every completed analysis is committed to a dedicated GitHub repository branch, creating a full audit trail per project wave.

## 🏗️ Architecture & Tech Stack

|

| **Layer** | **Technology** |
| Environment & Deps | Python 3.12, `uv` |
| Exploration | `Marimo` notebooks (`.ipynb` compatible) |
| Storage / Source of Truth | Microsoft SharePoint (Projects & Analytics folders) |
| Data Processing | Pandas, Scikit-Learn Pipelines |
| Statistical Engine | Scikit-Learn, Statsmodels, XGBoost, CatBoost, LightGBM, `conjoint` / `pymer4` for CBC |
| Model Evaluation | SHAP, Bootstrap CI, automated model comparison (RF vs XGBoost vs CatBoost) |
| Vector Memory | Pinecone (survey mapping embeddings, past analysis retrieval) |
| AI Commentary | Multi-LLM router: **Claude**, **ChatGPT**, **Kimi Code**, **Perplexity** (selectable per task) |
| Design Integration | Canva API (auto-generate slide shells from deliverable templates) |
| Delivery Formats | `final_payload.json`, Streamlit Web App, `.ipynb`, PowerPoint (`.pptx`) |
| Deployment | Docker containers |
| Versioning | GitHub (per-project branches, automated commit on completion) |

## 🔄 Workflow

### Full Pipeline Mode

1. **Ingest** — Pull raw Forsta files + data mapping dictionaries from SharePoint.
2. **Clean** — Drop sparse columns (>90% empty), normalize types, handle missing values per mapping rules.
3. **Analyze** — Run the selected statistical module(s) on survey codes.
4. **Translate** — Map codes → labels to produce a human-readable JSON statistical payload.
5. **Commentary** — Route the JSON to the selected LLM to generate **two distinct insights**: Academic (methodological) and Business (strategic).
6. **Deliver** — Export in the selected format (JSON / App / Notebook / PowerPoint), save back to SharePoint, commit to GitHub.

### Module-Only Mode

An analyst can trigger any single module in isolation:

- Descriptive statistics only
- Typology (K-Means) only
- Model comparison only
- Commentary only (on a pre-existing payload)

## 📦 Statistical Modules

Each module is a standalone, atomic Python tool. Modules are selected at launch or auto-suggested by a template engine based on survey type.

| **Module** | **Methods** |
| `descriptive` | Frequencies, cross-tabs, means, medians, confidence intervals (bootstrap) |
| `typology` | K-Means with automated K selection (Elbow + Silhouette), UMAP visualization |
| `modeling` | Automated comparison: Random Forest, XGBoost, CatBoost — SHAP explainability for winner |
| `cbc` | Conjoint-Based Choice modeling: compare across `pymer4`, `statsmodels`, `conjoint` — select best fit |
| `forecasting` | Time-series or wave-over-wave trend analysis |
| `regression` | Logistic regression with bootstrap CIs |

## 🤖 LLM Router

The AI layer is model-agnostic. At launch (or per-task), the analyst selects which LLM to use:

| **Provider** | **Best used for** |
| **Claude** | Long-context analysis, structured JSON commentary, RGPD-sensitive synthesis |
| **ChatGPT** | General business summaries, PowerPoint narrative |
| **Kimi Code** | Code generation assistance, mapping resolution |
| **Perplexity** | External benchmarking, literature-grounded academic commentary |

Commentary is always generated in two personas:

- **Academic/Statistical** — methodology, validity, statistical significance
- **Business/Marketing** — strategic implications, client-facing language

Power BI integration: the LLM router can also be called from Power Query (M) to generate synthesis commentary directly inside a PBI table (e.g., country-level summaries).

## 📌 Versioning & Collaboration

- Each survey project maps to a GitHub repository with branches per wave (e.g., `wave-1`, `wave-2`).
- When a new capability is added, the system commits only /skills changes with a structured SKILL commit message.
- Analysts can review, annotate, and merge via standard GitHub PR workflow.
- Pinecone stores embeddings of past survey mappings and findings — enabling semantic search across historical projects.

## 🎨 Design Integration (Canva API)

Once the statistical payload and AI commentary are finalized, the system can push structured content to Canva via API to auto-populate slide templates, freeing analysts from manual deck building.

## 🗂️ Project File Structure

```
/antigravity/
├── README.md
├── task_plan.md
├── findings.md
├── progress.md
├── gemini.md               # Project rules + JSON Schema
├── .env                    # API keys (SharePoint, LLMs, GitHub, Pinecone, Canva)
├── app.py                  # Web App entry point (Streamlit/FastAPI)
├── Dockerfile
├── /arch/                  # SOPs and workflow templates
├── /tools/                 # Atomic Python scripts (pipeline modules)
│   ├── 1_ingest_clean.py
│   ├── 2_statistical_engine.py
│   ├── 3_ai_commentator.py
│   ├── 4_export_deliver.py
│   └── 5_skill_commit.py
├── /skills/                # Per-step skill definitions (for agent routing)
│   ├── skill_clean.md
│   ├── skill_model.md
│   └── skill_comment.md
└── /tmp/                   # Local test outputs
```