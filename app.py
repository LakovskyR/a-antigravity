"""
Antigravity - Streamlit Web Application.
Four-step wizard for schema-aware survey analysis.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import importlib.util

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
TMP = ROOT / "tmp"


st.set_page_config(
    page_title="Antigravity Analytics",
    page_icon="AG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --bg: #f4efe7;
    --ink: #1d2a32;
    --card: #fffdf8;
    --accent: #0f766e;
    --accent-2: #e07a5f;
    --line: #d8cfc1;
}
.stApp {
    background: radial-gradient(circle at 15% 10%, #f3e9d7 0%, #f4efe7 40%, #ece6dc 100%);
    color: var(--ink);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.ag-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 8px 22px rgba(29, 42, 50, 0.06);
}
.ag-kicker {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #55616a;
    font-size: 0.75rem;
}
.ag-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #172029;
}
.ag-step {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    border: 1px solid #c4b8a7;
    background: #f8f4ed;
    margin-right: 0.35rem;
    margin-bottom: 0.25rem;
    font-size: 0.8rem;
}
.ag-step-active {
    border-color: var(--accent);
    background: #d5f0ed;
    color: #0a5b55;
    font-weight: 600;
}
.ag-log {
    font-family: Consolas, Menlo, monospace;
    font-size: 0.82rem;
    background: #1e2933;
    color: #dbf5ee;
    border-radius: 10px;
    border: 1px solid #3a4b59;
    padding: 0.8rem;
    min-height: 240px;
    white-space: pre-wrap;
}
</style>
""",
    unsafe_allow_html=True,
)


def _load_module(module_path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource
def get_schema_module():
    return _load_module(ROOT / "tools" / "0_schema_parser.py", "schema_parser_mod")


@st.cache_resource
def get_ingest_module():
    return _load_module(ROOT / "tools" / "1_ingest_clean.py", "ingest_clean_mod")


def init_state():
    defaults = {
        "step": 1,
        "source_path": "",
        "project_name": "Survey",
        "wave": datetime.now().strftime("%Y-%m"),
        "llm_provider": "openai",
        "detected_schemas": {},
        "selected_schema": None,
        "df_preview": None,
        "all_columns": [],
        "schema_vars": {},
        "selected_columns": [],
        "selected_modules": [],
        "goal": "segmentation",
        "target_variable": "auto",
        "k_segments": 4,
        "generate_notebook": True,
        "generate_powerbi": True,
        "run_complete": False,
        "run_success": False,
        "run_logs": [],
        "last_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()
s = st.session_state


def _ensure_tmp_dirs():
    (TMP / "raw_data").mkdir(parents=True, exist_ok=True)
    (TMP / "outputs").mkdir(parents=True, exist_ok=True)


def detect_schemas(source_path: str) -> dict:
    schema_mod = get_schema_module()
    config = {"project_metadata": {"source_path": source_path}}
    found = schema_mod.find_all_schema_files(config)
    schemas = {}
    if found.get("responseid"):
        schemas["responseid"] = found["responseid"]
    schemas.update(found.get("loops", {}))
    return schemas


def parse_schema_variables(schema_content: str) -> dict:
    schema_mod = get_schema_module()
    questions = schema_mod.parse_schema(schema_content)
    var_map = {}
    for qid, info in questions.items():
        for var in info.get("variables", []):
            if var not in var_map:
                var_map[var] = {
                    "question_id": qid,
                    "label": info.get("label", var),
                    "type": info.get("type", "unknown"),
                }
    return var_map


def suggest_modules_from_schema(schema_content: str) -> list[str]:
    schema_mod = get_schema_module()
    questions = schema_mod.parse_schema(schema_content)
    type_counts = {}
    for info in questions.values():
        if not info.get("is_content"):
            continue
        q_type = info.get("type", "unknown")
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
    suggested = schema_mod._recommend_modules(type_counts, questions)
    return [item.get("module") for item in suggested if item.get("module")]


def load_preview(source_path: str, selected_schema: str) -> pd.DataFrame | None:
    ingest_mod = get_ingest_module()
    try:
        effective_source = source_path
        if selected_schema != "responseid" and selected_schema.startswith("lp"):
            loop_file = ingest_mod.find_loop_data_file(source_path, selected_schema)
            if loop_file:
                effective_source = str(loop_file)
        df, _fmt = ingest_mod.load_source(effective_source)
        return df.head(300).astype(str)
    except Exception:
        return None


def current_payload_path() -> Path | None:
    candidates = [
        TMP / "outputs" / "current_payload.json",
        TMP / "final_delivery_payload.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def latest_notebook_path() -> Path | None:
    nb_dir = TMP / "notebooks"
    if not nb_dir.exists():
        return None
    notebooks = sorted(nb_dir.glob("*.ipynb"), key=lambda p: p.stat().st_mtime, reverse=True)
    return notebooks[0] if notebooks else None


def write_run_config() -> dict:
    workflow_type = "module"
    run_config = {
        "project_metadata": {
            "project_name": s.project_name,
            "survey_type": "forsta",
            "wave": s.wave,
            "selected_modules": s.selected_modules,
            "goal": s.goal,
            "k_segments": s.k_segments,
            "workflow_type": workflow_type,
            "delivery_format": "json",
            "llm_provider": s.llm_provider,
            "source_path": s.source_path,
            "selected_schema": s.selected_schema,
            "modules_source": "webapp",
            "target_variable": None if s.target_variable == "auto" else s.target_variable,
        },
        "generate_notebook": bool(s.generate_notebook),
        "generate_powerbi": bool(s.generate_powerbi),
        "schema_column_filter": {
            "keep_variables": s.selected_columns,
            "drop_types": ["date", "time", "quantity", "opentext"],
            "total_content_questions": len(s.schema_vars),
            "total_variables": len(s.selected_columns),
        },
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "run_id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "status": "configured",
            "pipeline_steps": [],
            "prefer_schema_suggestion": False,
        },
    }
    TMP.mkdir(parents=True, exist_ok=True)
    with open(TMP / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    return run_config


def _enforce_column_selection_after_schema_step():
    run_cfg_path = TMP / "run_config.json"
    if not run_cfg_path.exists():
        return
    with open(run_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["schema_column_filter"] = {
        "keep_variables": s.selected_columns,
        "drop_types": ["date", "time", "quantity", "opentext"],
        "total_content_questions": len(s.schema_vars),
        "total_variables": len(s.selected_columns),
    }
    cfg.setdefault("project_metadata", {})["selected_modules"] = s.selected_modules
    cfg["project_metadata"]["modules_source"] = "webapp"

    with open(run_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def run_pipeline(log_placeholder, progress_bar, status_placeholder):
    steps = [
        "0_schema_parser.py",
        "1_ingest_clean.py",
        "2_statistical_engine.py",
        "3_ai_commentator.py",
        "4_export_deliver.py",
        "5_archive_outputs.py",
    ]
    if s.generate_notebook:
        steps.append("6_notebook_generator.py")
    if s.generate_powerbi:
        steps.append("7_powerbi_export.py")

    with open(TMP / "run_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["run_info"]["pipeline_steps"] = steps
    with open(TMP / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    logs: list[str] = []
    child_env = dict(**os.environ)
    child_env.setdefault("PYTHONIOENCODING", "utf-8")

    for idx, step in enumerate(steps, start=1):
        status_placeholder.markdown(f"**Running {idx}/{len(steps)}:** `{step}`")
        logs.append(f"[RUN] {step}")
        log_placeholder.markdown(
            "<div class='ag-log'>" + "\n".join(logs[-80:]) + "</div>",
            unsafe_allow_html=True,
        )
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "tools" / step)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            logs.append(line.rstrip())
            log_placeholder.markdown(
                "<div class='ag-log'>" + "\n".join(logs[-80:]) + "</div>",
                unsafe_allow_html=True,
            )
        rc = process.wait()
        progress_bar.progress(idx / len(steps))

        if step == "0_schema_parser.py":
            _enforce_column_selection_after_schema_step()

        if rc != 0:
            return False, logs, f"Step failed: {step} (exit code {rc})"

    return True, logs, ""


def show_header():
    st.markdown("<div class='ag-kicker'>Pharma Market Research Workflow</div>", unsafe_allow_html=True)
    st.markdown("<div class='ag-title'>Antigravity Analytics Webapp</div>", unsafe_allow_html=True)
    st.caption("Schema-aware Forsta processing from upload to notebook and Power BI package.")


def show_step_indicator():
    labels = [
        "1. Upload + Schema",
        "2. Columns",
        "3. Modules",
        "4. Run + Results",
    ]
    row = []
    for i, label in enumerate(labels, start=1):
        cls = "ag-step ag-step-active" if s.step == i else "ag-step"
        row.append(f"<span class='{cls}'>{label}</span>")
    st.markdown("".join(row), unsafe_allow_html=True)


def step_upload():
    st.markdown("<div class='ag-card'>", unsafe_allow_html=True)
    st.subheader("Step 1 - Data Upload and Schema Detection")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        uploaded = st.file_uploader(
            "Upload data source",
            type=["zip", "xlsx", "xls", "csv", "txt", "tsv"],
            help="Recommended: Forsta ZIP export containing responseid/lp files.",
        )
        manual_path = st.text_input("Or provide existing file/folder path", value=s.source_path or "")

    with c2:
        s.project_name = st.text_input("Project name", value=s.project_name)
        s.wave = st.text_input("Wave", value=s.wave)
        s.llm_provider = st.selectbox(
            "LLM provider",
            ["openai", "claude", "kimi", "perplexity"],
            index=["openai", "claude", "kimi", "perplexity"].index(
                s.llm_provider if s.llm_provider in ["openai", "claude", "kimi", "perplexity"] else "openai"
            ),
        )

    detect_clicked = st.button("Detect Schemas", type="primary")
    if detect_clicked:
        _ensure_tmp_dirs()
        source = None
        if uploaded is not None:
            dest = TMP / "raw_data" / uploaded.name
            with open(dest, "wb") as f:
                f.write(uploaded.getbuffer())
            source = str(dest)
            st.success(f"Uploaded file saved to {dest}")
        elif manual_path.strip():
            p = Path(manual_path.strip())
            if p.exists():
                source = str(p)
            else:
                st.error(f"Path not found: {manual_path.strip()}")

        if source:
            s.source_path = source
            with st.spinner("Scanning schema files..."):
                schemas = detect_schemas(source)
            if not schemas:
                st.error("No schema files found. Provide source containing responseid.txt or lp*.txt/lp*.xlsx.")
            else:
                s.detected_schemas = schemas
                options = list(schemas.keys())
                if s.selected_schema not in options:
                    loop_opts = [k for k in options if k.startswith("lp")]
                    s.selected_schema = loop_opts[0] if loop_opts else options[0]
                st.success(f"Detected {len(options)} schema file(s).")

    if s.detected_schemas:
        options = list(s.detected_schemas.keys())
        default_idx = options.index(s.selected_schema) if s.selected_schema in options else 0
        s.selected_schema = st.radio(
            "Choose schema to analyze",
            options,
            index=default_idx,
            format_func=lambda k: f"{k} ({'main metadata schema' if k == 'responseid' else 'loop schema'})",
        )

        if st.button("Load Preview"):
            with st.spinner("Loading data preview..."):
                preview = load_preview(s.source_path, s.selected_schema)
            s.df_preview = preview
            if preview is not None and not preview.empty:
                s.all_columns = preview.columns.tolist()
                s.schema_vars = parse_schema_variables(s.detected_schemas[s.selected_schema])
                st.success(f"Preview loaded: {len(preview)} rows x {len(preview.columns)} columns")
            else:
                s.all_columns = []
                s.schema_vars = parse_schema_variables(s.detected_schemas[s.selected_schema])
                st.warning("Preview not available; continuing with schema-only variable list.")

    if isinstance(s.df_preview, pd.DataFrame) and not s.df_preview.empty:
        st.dataframe(s.df_preview.head(8), use_container_width=True)

    ready = bool(s.source_path and s.detected_schemas and s.selected_schema)
    if st.button("Next: Column Selection", disabled=not ready):
        if not s.schema_vars and s.selected_schema in s.detected_schemas:
            s.schema_vars = parse_schema_variables(s.detected_schemas[s.selected_schema])
        s.step = 2
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def step_columns():
    st.markdown("<div class='ag-card'>", unsafe_allow_html=True)
    st.subheader("Step 2 - Column Selection")

    schema_cols = list(s.schema_vars.keys())
    available_cols = s.all_columns or schema_cols
    matched = [c for c in available_cols if c in s.schema_vars]
    unmatched = [c for c in available_cols if c not in s.schema_vars]

    st.info(f"Matched to schema: {len(matched)} columns | extra columns: {len(unmatched)}")

    default_cols = s.selected_columns or matched or available_cols
    select_all = st.checkbox("Select all available columns", value=not s.selected_columns)

    if select_all:
        selected = available_cols
    else:
        selected = st.multiselect(
            "Choose columns to keep",
            options=available_cols,
            default=[c for c in default_cols if c in available_cols],
        )

    st.caption(f"Selected columns: {len(selected)}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back"):
            s.step = 1
            st.rerun()
    with c2:
        if st.button("Next: Module Setup", type="primary", disabled=not selected):
            s.selected_columns = selected
            s.step = 3
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def step_modules():
    st.markdown("<div class='ag-card'>", unsafe_allow_html=True)
    st.subheader("Step 3 - Module and Variable Selection")

    goal_defaults = {
        "segmentation": ["descriptive", "pca", "typology", "cah", "lda", "latent_class"],
        "drivers": ["descriptive", "anova", "ols", "regression", "modeling", "decision_tree"],
        "descriptive": ["descriptive", "anova"],
        "conjoint": ["descriptive", "cbc", "cbc_simulator", "cas_vignettes"],
        "full": [
            "descriptive", "anova", "ols", "regression", "typology", "pca", "cah", "lda",
            "latent_class", "modeling", "decision_tree", "cbc", "cbc_simulator", "cas_vignettes", "forecasting"
        ],
    }

    all_modules = [
        "descriptive", "anova", "ols", "regression", "typology", "pca", "cah", "lda",
        "latent_class", "modeling", "decision_tree", "cbc", "cbc_simulator", "cas_vignettes", "forecasting"
    ]

    s.goal = st.radio(
        "Analysis goal",
        ["segmentation", "drivers", "descriptive", "conjoint", "full"],
        index=["segmentation", "drivers", "descriptive", "conjoint", "full"].index(s.goal),
        horizontal=True,
    )

    ai_suggested = []
    if s.selected_schema in s.detected_schemas:
        ai_suggested = suggest_modules_from_schema(s.detected_schemas[s.selected_schema])

    default_modules = s.selected_modules or ai_suggested or goal_defaults[s.goal]
    selected_mods = st.multiselect(
        "Statistical modules",
        options=all_modules,
        default=[m for m in default_modules if m in all_modules],
    )

    if "cbc_simulator" in selected_mods and "cbc" not in selected_mods:
        selected_mods.insert(selected_mods.index("cbc_simulator"), "cbc")
        st.warning("cbc_simulator requires cbc. Added automatically.")

    target_options = ["auto"] + [c for c in s.selected_columns if c in s.schema_vars]
    if s.target_variable not in target_options:
        s.target_variable = "auto"
    s.target_variable = st.selectbox("Target variable (optional)", target_options)

    s.k_segments = st.slider("Max clusters/segments", min_value=2, max_value=10, value=int(s.k_segments))
    s.generate_notebook = st.checkbox("Generate notebook", value=bool(s.generate_notebook))
    s.generate_powerbi = st.checkbox("Generate Power BI package", value=bool(s.generate_powerbi))

    st.caption(f"Modules selected: {len(selected_mods)}")
    if ai_suggested:
        st.caption("AI suggestion from schema: " + ", ".join(ai_suggested))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back"):
            s.step = 2
            st.rerun()
    with c2:
        if st.button("Next: Run Pipeline", type="primary", disabled=not selected_mods):
            s.selected_modules = selected_mods
            s.step = 4
            s.run_complete = False
            s.run_success = False
            s.run_logs = []
            s.last_error = ""
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def step_run():
    st.markdown("<div class='ag-card'>", unsafe_allow_html=True)
    st.subheader("Step 4 - Run and Results")

    st.write(
        {
            "source_path": s.source_path,
            "schema": s.selected_schema,
            "columns": len(s.selected_columns),
            "modules": s.selected_modules,
            "llm_provider": s.llm_provider,
            "generate_notebook": s.generate_notebook,
            "generate_powerbi": s.generate_powerbi,
        }
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back"):
            s.step = 3
            st.rerun()
    with c2:
        run_clicked = st.button("Run Full Pipeline", type="primary", disabled=not s.selected_modules)

    log_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()

    if run_clicked:
        write_run_config()
        with st.spinner("Running pipeline..."):
            ok, logs, err = run_pipeline(log_placeholder, progress_bar, status_placeholder)
        s.run_complete = True
        s.run_success = ok
        s.run_logs = logs
        s.last_error = err

    if s.run_logs:
        log_placeholder.markdown(
            "<div class='ag-log'>" + "\n".join(s.run_logs[-120:]) + "</div>",
            unsafe_allow_html=True,
        )

    if s.run_complete:
        if s.run_success:
            st.success("Pipeline completed successfully.")
        else:
            st.error(s.last_error or "Pipeline failed.")

        payload = current_payload_path()
        nb_path = latest_notebook_path()
        pbi_dir = TMP / "powerbi"

        st.markdown("### Downloads")
        cols = st.columns(3)

        with cols[0]:
            if payload and payload.exists():
                st.download_button(
                    "Download payload JSON",
                    data=payload.read_bytes(),
                    file_name=payload.name,
                    mime="application/json",
                )

        with cols[1]:
            if nb_path and nb_path.exists():
                st.download_button(
                    "Download notebook",
                    data=nb_path.read_bytes(),
                    file_name=nb_path.name,
                    mime="application/x-ipynb+json",
                )

        with cols[2]:
            if pbi_dir.exists():
                summary = "\n".join(f"- {p.name}" for p in sorted(pbi_dir.iterdir()) if p.is_file())
                st.markdown("Power BI files generated:")
                st.code(summary or "(none)")

        if payload and payload.exists():
            with st.expander("View payload preview"):
                try:
                    st.json(json.loads(payload.read_text(encoding="utf-8")))
                except Exception:
                    st.text(payload.read_text(encoding="utf-8", errors="replace")[:4000])

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    show_header()
    show_step_indicator()
    with st.expander("Help"):
        st.markdown(
            "1. Upload a survey ZIP/file and detect schemas.\n"
            "2. Choose columns to keep for analysis.\n"
            "3. Select modules, target, and output options.\n"
            "4. Run pipeline and download outputs."
        )

    if s.step == 1:
        step_upload()
    elif s.step == 2:
        step_columns()
    elif s.step == 3:
        step_modules()
    else:
        step_run()


if __name__ == "__main__":
    main()
