"""
Launcher - Entry Point for Project Antigravity.
Interactive menu for configuring and launching the pipeline.
Supports --non-interactive mode for automated/scripted runs.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


AVAILABLE_MODULES = [
    # --- Core ---
    ("descriptive", "Descriptive statistics - frequencies, crosstabs, bootstrap CI", "Core"),
    ("anova", "ANOVA - one-way + Tukey HSD post-hoc", "Core"),
    ("ols", "OLS - linear regression with p-values, R², confidence intervals", "Core"),
    ("regression", "Logistic regression - binary outcome, bootstrap CI", "Core"),
    # --- Segmentation ---
    ("typology", "Typology - K-Means clustering + UMAP visualization", "Segmentation"),
    ("pca", "PCA - dimensionality reduction, scree plot, loadings", "Segmentation"),
    ("cah", "CAH - hierarchical agglomerative clustering (Ward)", "Segmentation"),
    ("lda", "LDA - linear discriminant analysis, segment profiling", "Segmentation"),
    ("latent_class", "Latent Class / Mixte - GMM hidden subgroups, BIC-optimal k", "Segmentation"),
    # --- Modeling ---
    ("modeling", "Model comparison - RF / XGBoost / CatBoost + SHAP importances", "Modeling"),
    ("decision_tree", "Decision tree - interpretable rules, max depth 4", "Modeling"),
    # --- Conjoint / Choice ---
    ("cbc", "CBC - conjoint choice estimation (MNLogit)", "Conjoint / Choice"),
    ("cbc_simulator", "CBC Simulator - share-of-preference + WTP (requires cbc)", "Conjoint / Choice"),
    ("cas_vignettes", "Cas Vignettes - factorial design generator or part-worth analysis", "Conjoint / Choice"),
    # --- Time / Trend ---
    ("forecasting", "Forecasting - linear trend by wave", "Trend"),
]

DEFAULT_MODULES = ["descriptive", "anova", "typology", "ols"]
SUPPORTED_SOURCE_EXTENSIONS = {".zip", ".xlsx", ".xls", ".csv", ".txt", ".tsv"}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Antigravity Pipeline Launcher")
    parser.add_argument("--non-interactive", action="store_true", help="Run with defaults, no prompts")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["claude", "openai", "kimi", "perplexity"],
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument("--modules", nargs="+", default=None, help="Override module list")
    parser.add_argument(
        "--source",
        default="tmp/raw_data",
        help="Data source: folder path OR file path (.zip/.xlsx/.csv/.txt)",
    )
    parser.add_argument(
        "--goal",
        default="segmentation",
        choices=["segmentation", "drivers", "descriptive", "conjoint", "full"],
        help="Analysis goal (default: segmentation)",
    )
    parser.add_argument("--k", type=int, default=4, help="Segment count for clustering modules (default: 4)")
    parser.add_argument("--target", default=None, help="Optional target variable for supervised modules")
    parser.add_argument("--with-notebook", action="store_true", help="Force notebook generation")
    parser.add_argument("--with-powerbi", action="store_true", help="Enable Power BI package generation")
    return parser.parse_args()


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    """Print application header - safe for all console encodings."""
    try:
        print("=" * 70)
        print("   PROJECT ANTIGRAVITY — Forsta Survey Analytics Engine")
        print("=" * 70)
        print()
    except UnicodeEncodeError:
        print("=" * 70)
        print("   PROJECT ANTIGRAVITY - Forsta Survey Analytics Engine")
        print("=" * 70)
        print()


def all_module_keys() -> List[str]:
    return [m[0] for m in AVAILABLE_MODULES]


def module_groups() -> List[Tuple[str, List[Tuple[str, str]]]]:
    groups = []
    seen = []
    for _, _, cat in AVAILABLE_MODULES:
        if cat not in seen:
            seen.append(cat)
    for cat in seen:
        items = [(k, desc) for (k, desc, c) in AVAILABLE_MODULES if c == cat]
        groups.append((cat, items))
    return groups


def normalize_modules(modules: List[str]) -> List[str]:
    """Keep valid modules, preserve order, dedupe."""
    valid = set(all_module_keys())
    out = []
    seen = set()
    for m in modules:
        key = str(m).strip()
        if not key:
            continue
        if key not in valid:
            print(f"  Warning: unknown module '{key}' ignored.")
            continue
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def enforce_module_dependencies(modules: List[str], announce: bool = True) -> List[str]:
    """Apply module dependency rules."""
    out = list(modules)
    if "cbc_simulator" in out and "cbc" not in out:
        idx = out.index("cbc_simulator")
        out.insert(idx, "cbc")
        if announce:
            print("  ⚠ cbc_simulator requires cbc — added automatically.")
    return out


def load_ai_suggested_modules(tmp_dir: Path) -> List[str]:
    """Load AI-confirmed modules from tmp/analytics_brief.json if available."""
    brief_path = tmp_dir / "analytics_brief.json"
    if not brief_path.exists():
        return []

    try:
        with open(brief_path, encoding="utf-8") as f:
            brief = json.load(f)

        llm = brief.get("llm_insights", {})
        mods = llm.get("confirmed_modules") or brief.get("confirmed_modules") or []
        if isinstance(mods, list):
            return normalize_modules(mods)
    except Exception as exc:
        print(f"  Warning: could not read AI module suggestions ({exc}).")

    return []


def prompt_source_path() -> str:
    """
    Ask user for data source.
    Accepts folder path or direct file path (.zip/.xlsx/.xls/.csv/.txt/.tsv).
    """
    print("\n-- Data Source ------------------------------------------------")
    print("  Supported formats: ZIP (Forsta export), XLSX, CSV, TXT/TSV")
    print("  You can enter:")
    print("    1. A folder path  -> app finds the file automatically")
    print("    2. A file path    -> .zip / .xlsx / .csv / .txt")
    print()

    default = "tmp/raw_data"
    raw = input(f"  Path [{default}]: ").strip()
    path = raw if raw else default

    p = Path(path)
    if not p.exists():
        print(f"  Path not found: {path}")
        retry = input("  Try again (y) or use default (Enter): ").strip().lower()
        if retry == "y":
            return prompt_source_path()
        return default

    if p.is_file():
        ext = p.suffix.lower()
        if ext not in SUPPORTED_SOURCE_EXTENSIONS:
            print(f"  Unsupported file type: {ext}")
            print("  Supported: .zip .xlsx .xls .csv .txt .tsv")
            return prompt_source_path()
        print(f"  File: {p.name} ({ext})")
    else:
        found = []
        for ext in sorted(SUPPORTED_SOURCE_EXTENSIONS):
            found.extend(p.glob(f"*{ext}"))
        if found:
            print(f"  Folder contains: {[f.name for f in found[:5]]}")
        else:
            print("  No supported files found in folder")

    return str(path)


def get_sharepoint_folder(non_interactive=False, source=None):
    """Backward-compatible wrapper for source path selection."""
    if non_interactive:
        path = source or "tmp/raw_data"
        print(f"[AUTO] Data source: {path}")
        return path

    return prompt_source_path()


def select_workflow(non_interactive=False, modules_override=None):
    """Present workflow selection menu."""
    if non_interactive:
        modules = modules_override or DEFAULT_MODULES
        modules = enforce_module_dependencies(normalize_modules(modules), announce=True)
        print(f"[AUTO] Workflow: full | Modules: {', '.join(modules)}")
        return "full", modules

    print("STEP 2: Select Workflow")
    print("-" * 50)
    print("[1] Full Pipeline - Run all modules")
    print("[2] Select Module - Run specific analysis only")
    print()

    while True:
        choice = input("Select (1-2): ").strip()
        if choice == "1":
            modules = enforce_module_dependencies(all_module_keys(), announce=False)
            print("Full Pipeline selected\n")
            return "full", modules
        if choice == "2":
            print()
            return "module", select_modules()
        print("Invalid choice. Please enter 1 or 2.")


def select_modules() -> List[str]:
    """Present module selection menu with categories."""
    index_to_key = {}
    idx = 1

    print("Available modules:")
    for cat, items in module_groups():
        print(f"  -- {cat} {'-' * max(1, 42 - len(cat))}")
        for key, desc in items:
            index_to_key[idx] = key
            short = desc.split(" - ")[0]
            print(f"  {idx:>2}. {key:<13} {short}")
            idx += 1

    print()
    print("Enter numbers (e.g. 1 3 5) or 'all' or press Enter for AI suggestion:")

    while True:
        raw = input("Select: ").strip()
        if not raw:
            print("Using AI suggestion after schema parser.\n")
            return []

        if raw.lower() == "all":
            selected = all_module_keys()
            selected = enforce_module_dependencies(selected, announce=True)
            print(f"Selected modules: {', '.join(selected)}\n")
            return selected

        tokens = raw.replace(",", " ").split()
        selected = []
        invalid = []
        for tok in tokens:
            if not tok.isdigit():
                invalid.append(tok)
                continue
            n = int(tok)
            if n in index_to_key:
                selected.append(index_to_key[n])
            else:
                invalid.append(tok)

        selected = normalize_modules(selected)
        selected = enforce_module_dependencies(selected, announce=True)

        if invalid:
            print(f"Invalid entries ignored: {', '.join(invalid)}")

        if selected:
            print(f"Selected modules: {', '.join(selected)}\n")
            return selected

        print("No valid modules selected. Try again.")


def select_delivery_format(non_interactive=False):
    """Present delivery format selection menu."""
    if non_interactive:
        print("[AUTO] Delivery format: json")
        return "json"

    print("STEP 3: Select Delivery Format")
    print("-" * 50)

    formats = [
        ("json", "JSON - Raw payload for API/integration"),
        ("webapp", "WebApp - Launch Streamlit dashboard"),
        ("notebook", "Notebook - Generate .ipynb file"),
        ("powerpoint", "PowerPoint - Generate .pptx slides"),
    ]

    for idx, (_, desc) in enumerate(formats, 1):
        print(f"[{idx}] {desc}")
    print()

    while True:
        choice = input("Select (1-4): ").strip()
        format_map = {"1": "json", "2": "webapp", "3": "notebook", "4": "powerpoint"}
        if choice in format_map:
            selected = format_map[choice]
            print(f"Delivery format: {selected}\n")
            return selected
        print("Invalid choice. Please enter 1-4.")


def select_llm_provider(non_interactive=False, provider_override=None):
    """Present LLM provider selection menu."""
    if non_interactive:
        provider = provider_override or "openai"
        print(f"[AUTO] LLM provider: {provider}")
        return provider

    print("STEP 4: Select LLM Provider for Commentary")
    print("-" * 50)

    providers = [
        ("claude", "Claude - Best for long-context, structured analysis"),
        ("openai", "OpenAI - General summaries, PowerPoint narratives"),
        ("kimi", "Kimi - Code generation, mapping assistance"),
        ("perplexity", "Perplexity - External benchmarking, literature grounding"),
    ]

    for idx, (_, desc) in enumerate(providers, 1):
        print(f"[{idx}] {desc}")
    print()

    while True:
        choice = input("Select (1-4): ").strip()
        provider_map = {"1": "claude", "2": "openai", "3": "kimi", "4": "perplexity"}
        if choice in provider_map:
            selected = provider_map[choice]
            print(f"LLM provider: {selected}\n")
            return selected
        print("Invalid choice. Please enter 1-4.")


def extract_project_info(sharepoint_path: str) -> dict:
    """Extract project name and wave from SharePoint path."""
    parts = [p for p in sharepoint_path.split("/") if p]
    project_name = parts[1] if len(parts) > 1 else "Unknown"
    wave = parts[2] if len(parts) > 2 else "Unknown"
    return {"project_name": project_name, "wave": wave}


def create_config(
    sharepoint_path: str,
    workflow_type: str,
    modules: List[str],
    delivery_format: str,
    llm_provider: str,
    goal: str = "segmentation",
    k_segments: int = 4,
    target_variable: str | None = None,
    modules_source: str = "manual",
    prefer_schema_suggestion: bool = False,
    generate_notebook: bool = False,
    generate_powerbi: bool = False,
) -> dict:
    """Create the run configuration dictionary."""
    project_info = extract_project_info(sharepoint_path)

    return {
        "project_metadata": {
            "project_name": project_info["project_name"],
            "survey_type": "forsta",
            "wave": project_info["wave"],
            "selected_modules": modules,
            "goal": goal,
            "k_segments": int(k_segments),
            "workflow_type": workflow_type,
            "delivery_format": delivery_format,
            "llm_provider": llm_provider,
            "source_path": sharepoint_path,
            "modules_source": modules_source,
            "target_variable": target_variable,
        },
        "generate_notebook": generate_notebook,
        "generate_powerbi": generate_powerbi,
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "run_id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "status": "configured",
            "pipeline_steps": [],
            "prefer_schema_suggestion": prefer_schema_suggestion,
        },
    }


def save_config(config: dict) -> Path:
    """Save configuration to tmp/run_config.json."""
    tmp_dir = Path(__file__).parent.parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    config_path = tmp_dir / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return config_path


def display_summary(config: dict):
    """Display configuration summary."""
    meta = config["project_metadata"]

    print()
    print("=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Project:      {meta['project_name']}")
    print(f"Wave:         {meta['wave']}")
    print(f"Source:       {meta['source_path']}")
    print(f"Workflow:     {meta['workflow_type']}")
    print(f"Modules:      {', '.join(meta['selected_modules']) if meta['selected_modules'] else '[AI suggestion]'}")
    print(f"Delivery:     {meta['delivery_format']}")
    print(f"LLM Provider: {meta['llm_provider']}")
    print("=" * 70)
    print()


def confirm_launch(non_interactive=False):
    """Ask user to confirm pipeline launch."""
    if non_interactive:
        print("[AUTO] Proceeding with pipeline launch...")
        return True
    print("Ready to launch the pipeline.")
    choice = input("Proceed? [Y/n]: ").strip().lower()
    return choice in ("", "y", "yes")


def build_pipeline_steps(config: dict) -> list:
    """Schema parser runs first; statistical engine always runs after ingest."""
    steps = [
        "0_schema_parser.py",
        "1_ingest_clean.py",
        "2_statistical_engine.py",
        "3_ai_commentator.py",
        "4_export_deliver.py",
        "5_archive_outputs.py",
        "5_skill_commit.py",
    ]
    if config.get("generate_notebook", True):
        steps.append("6_notebook_generator.py")
    if config.get("generate_powerbi", False):
        steps.append("7_powerbi_export.py")
    return steps


def present_schema_suggestions(tmp_dir: Path, non_interactive=False, prefer_schema_suggestion=False) -> str:
    """
    After schema parser runs, show LLM module suggestions to analyst.
    Returns "suggested", "original", or "abort".
    """
    brief_path = tmp_dir / "analytics_brief.json"
    if not brief_path.exists():
        return "original"

    with open(brief_path, encoding="utf-8") as f:
        brief = json.load(f)

    insights = brief.get("llm_insights", {})
    suggested = insights.get("confirmed_modules") or brief.get("confirmed_modules") or []
    context = insights.get("project_context") or brief.get("project_context") or ""
    notes = insights.get("analyst_notes") or brief.get("analyst_notes") or []

    if not suggested:
        run_config_path = tmp_dir / "run_config.json"
        if run_config_path.exists():
            try:
                with open(run_config_path, encoding="utf-8") as f:
                    run_cfg = json.load(f)
                meta = run_cfg.get("project_metadata", {})
                if meta.get("modules_source") == "schema_parser_llm":
                    suggested = meta.get("selected_modules", [])
                if not context:
                    context = run_cfg.get("project_context", "")
                if not notes:
                    notes = run_cfg.get("analyst_notes", [])
            except Exception:
                pass

    suggested = enforce_module_dependencies(normalize_modules(suggested), announce=False)
    if not suggested:
        return "original"

    print()
    print("=" * 70)
    print("SCHEMA ANALYSIS COMPLETE - AI Module Recommendations")
    print("=" * 70)
    if context:
        print(f"Project: {context}")
        print()
    print(f"Recommended modules: {', '.join(suggested)}")
    if notes:
        print("Analyst notes:")
        for note in notes:
            print(f"  - {note}")
    print()

    if non_interactive:
        if prefer_schema_suggestion:
            print("[AUTO] Using AI-suggested modules.")
            return "suggested"
        print("[AUTO] Keeping provided module selection.")
        return "original"

    print("[1] Use AI-suggested modules")
    print("[2] Keep my original selection")
    print("[3] Abort - review tmp/analytics_brief.json manually")
    print()

    while True:
        choice = input("Select (1-3): ").strip()
        if choice == "1":
            return "suggested"
        if choice == "2":
            return "original"
        if choice == "3":
            return "abort"
        print("Enter 1, 2 or 3.")


def launch_pipeline(config: dict, non_interactive=False) -> bool:
    """Centralized orchestration with schema confirmation step after step 0b."""
    print("\nLaunching pipeline...\n")
    root_dir = Path(__file__).parent.parent
    tmp_dir = root_dir / "tmp"
    steps = build_pipeline_steps(config)
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    pinned_modules = list(config.get("project_metadata", {}).get("selected_modules", []))
    pinned_source = config.get("project_metadata", {}).get("modules_source", "manual")

    config["run_info"]["pipeline_steps"] = steps
    save_config(config)

    for step in steps:
        print(f"[RUN] tools/{step}")
        if step == "6_notebook_generator.py":
            cmd = [sys.executable, f"tools/{step}"]
            if non_interactive:
                goal = str(config.get("project_metadata", {}).get("goal", "segmentation"))
                k_segments = int(config.get("project_metadata", {}).get("k_segments", 4))
                cmd.extend(["--non-interactive", "--goal", goal, "--k", str(k_segments)])
            result = subprocess.run(cmd, cwd=root_dir, env=child_env)
        else:
            result = subprocess.run([sys.executable, f"tools/{step}"], cwd=root_dir, env=child_env)
        if result.returncode != 0:
            print(f"Pipeline stopped at {step}")
            return False

        if step == "0_schema_parser.py":
            decision = present_schema_suggestions(
                tmp_dir,
                non_interactive=non_interactive,
                prefer_schema_suggestion=bool(config.get("run_info", {}).get("prefer_schema_suggestion", False)),
            )
            if decision == "abort":
                return False
            if decision == "suggested":
                with open(tmp_dir / "run_config.json", encoding="utf-8") as f:
                    config.update(json.load(f))
                config["project_metadata"]["selected_modules"] = enforce_module_dependencies(
                    normalize_modules(config["project_metadata"].get("selected_modules", [])),
                    announce=True,
                )
                save_config(config)
                print(f"Using suggested modules: {config['project_metadata']['selected_modules']}")
            else:
                run_cfg_path = tmp_dir / "run_config.json"
                if run_cfg_path.exists():
                    try:
                        with open(run_cfg_path, encoding="utf-8") as f:
                            config.update(json.load(f))
                    except Exception:
                        pass

                if pinned_modules:
                    kept = enforce_module_dependencies(normalize_modules(pinned_modules), announce=False)
                    config["project_metadata"]["selected_modules"] = kept
                    config["project_metadata"]["modules_source"] = pinned_source
                    save_config(config)
                    print(f"Keeping original modules: {kept}")

                current = config["project_metadata"].get("selected_modules", [])
                if not current:
                    fallback = DEFAULT_MODULES.copy()
                    config["project_metadata"]["selected_modules"] = fallback
                    config["project_metadata"]["modules_source"] = "default_fallback"
                    save_config(config)
                    print(f"No modules selected; using fallback: {fallback}")

    return True


def resolve_non_interactive_modules(args) -> Tuple[List[str], str, bool]:
    """Resolve modules for --non-interactive runs."""
    if args.modules:
        modules = enforce_module_dependencies(normalize_modules(args.modules), announce=True)
        return modules, "cli", False

    tmp_dir = Path(__file__).parent.parent / "tmp"
    ai_modules = load_ai_suggested_modules(tmp_dir)
    if ai_modules:
        modules = enforce_module_dependencies(ai_modules, announce=True)
        print(f"[AUTO] Modules from analytics_brief.json: {', '.join(modules)}")
        return modules, "analytics_brief", True

    modules = DEFAULT_MODULES.copy()
    print(f"[AUTO] Modules default set: {', '.join(modules)}")
    return modules, "default", True


def main():
    """Main entry point."""
    import io

    if (
        sys.stdout.encoding
        and sys.stdout.encoding.lower() not in ("utf-8", "utf8")
        and hasattr(sys.stdout, "buffer")
    ):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    args = parse_args()
    non_interactive = args.non_interactive

    if not non_interactive:
        clear_screen()
    print_header()

    modules = None
    modules_source = "manual"
    prefer_schema_suggestion = False
    generate_notebook = bool(non_interactive) or bool(args.with_notebook)
    generate_powerbi = bool(args.with_powerbi)

    if non_interactive:
        print("=" * 70)
        print("NON-INTERACTIVE MODE — Using defaults")
        print("=" * 70)
        print()
        modules, modules_source, prefer_schema_suggestion = resolve_non_interactive_modules(args)

    sharepoint_path = get_sharepoint_folder(non_interactive=non_interactive, source=args.source)
    workflow_type, selected_modules = select_workflow(
        non_interactive=non_interactive,
        modules_override=modules,
    )
    delivery_format = select_delivery_format(non_interactive=non_interactive)
    llm_provider = select_llm_provider(non_interactive=non_interactive, provider_override=args.provider)

    if selected_modules is None:
        selected_modules = []

    selected_modules = enforce_module_dependencies(normalize_modules(selected_modules), announce=True)

    if not non_interactive and selected_modules:
        modules_source = "interactive"

    config = create_config(
        sharepoint_path=sharepoint_path,
        workflow_type=workflow_type,
        modules=selected_modules,
        delivery_format=delivery_format,
        llm_provider=llm_provider,
        goal=args.goal,
        k_segments=max(2, int(args.k)),
        target_variable=args.target,
        modules_source=modules_source,
        prefer_schema_suggestion=prefer_schema_suggestion,
        generate_notebook=generate_notebook,
        generate_powerbi=generate_powerbi or (delivery_format == "powerbi"),
    )

    display_summary(config)

    config_path = save_config(config)
    print(f"Configuration saved to: {config_path}\n")

    if confirm_launch(non_interactive=non_interactive):
        success = launch_pipeline(config, non_interactive=non_interactive)
        if success:
            print("\nPipeline completed successfully.")
            if not non_interactive:
                print("\n" + "-" * 60)
                gen_nb = input("Generate a data scientist notebook? (Enter=yes / n=skip): ").strip().lower()
                if gen_nb != "n":
                    root_dir = Path(__file__).parent.parent
                    subprocess.run(
                        [
                            sys.executable,
                            "tools/6_notebook_generator.py",
                            "--goal",
                            config["project_metadata"].get("goal", "segmentation"),
                            "--k",
                            str(config["project_metadata"].get("k_segments", 4)),
                        ],
                        cwd=root_dir,
                    )
        else:
            print("\nPipeline failed. Check logs for details.")
            sys.exit(1)
    else:
        print("\nPipeline launch cancelled.")
        print(f"Configuration saved to {config_path}")
        print("You can resume by running: python tools/1_ingest_clean.py")


if __name__ == "__main__":
    main()
