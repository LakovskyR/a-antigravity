"""
Schema Parser — Module 0b (runs after launcher, before ingest)
Reads the Forsta responseid.txt schema (from ZIP or direct file) and produces:
  - tmp/project_schema.json   : structured question map (type, label, answer codes)
  - tmp/analytics_brief.json  : AI-suggested analytics per question + project context

Then prompts the selected LLM to:
  1. Confirm/enrich the analytics suggestions
  2. Generate a one-paragraph project brief for the commentator
  3. Recommend which modules to run given the question mix

INPUTS expected in tmp/run_config.json:
  - source_path          : SharePoint/local folder path
  - llm_provider         : which LLM to use for the brief
  - project_summary_file : optional path to the .xlsx project summary (APLUSA convention)

FORSTA SCHEMA FORMAT (responseid.txt, tab-separated):
  Question ID | Variable ID | Type | Start | Finish | Answer Code | Question Label | Answer Label

FORSTA QUESTION TYPES MAPPING:
  single      -> categorical (1 answer from list)    -> freq table, chi-square
  multi       -> binary flags (0/1 per option)       -> freq table, multi-response %
  ranking     -> ordinal rank (1=best)               -> mean rank, top-box
  grid        -> scale per row (Likert/NPS-style)    -> mean, distribution, drivers
  numericlist -> open numeric (count, %, volume)     -> mean, median, CI, regression
  opentext    -> verbatim text                       -> skip (no stats) / flag for NLP
  date/time   -> metadata                            -> skip
  quantity    -> system ID / count                   -> skip
"""

import io
import json
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MAIN_SCHEMA_PATTERN = "responseid.txt"
LOOP_SCHEMA_PATTERN = re.compile(r".*lp\w+\.txt$", re.IGNORECASE)
SCHEMA_EXCEL_PATTERN = re.compile(r".*lp\w+\.xlsx$", re.IGNORECASE)
SCHEMA_ZIP_PATTERN = "schemas_"

# Which question ID prefixes are "real" survey content vs metadata
CONTENT_PREFIXES = ("Q", "S", "P")
METADATA_PREFIXES = ("h", "H", "bg", "lang", "resp", "interview", "last", "first",
                     "Dacima", "Sample", "Agency", "Survey", "Dial", "CATI",
                     "NOM", "PRENOM", "MONTANT", "DUREE", "DPO")

# Analytics recommended per Forsta question type
TYPE_TO_ANALYTICS = {
    "single":      ["frequency_table", "cross_tab", "chi_square"],
    "multi":       ["multi_response_frequency", "cross_tab"],
    "ranking":     ["mean_rank", "top_box_analysis", "rank_distribution"],
    "grid":        ["mean_score", "score_distribution", "driver_analysis"],
    "numericlist": ["descriptive_stats", "bootstrap_ci", "regression"],
    "opentext":    ["verbatim_flag"],      # skip from stats, flag for NLP
    "date":        [],                     # skip
    "time":        [],                     # skip
    "quantity":    [],                     # skip
    "numeric":     ["descriptive_stats"],
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "tmp" / "run_config.json"
    if not config_path.exists():
        print("ERROR: run_config.json not found. Run 0_launcher.py first.")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict):
    config_path = Path(__file__).parent.parent / "tmp" / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# Step 1: Locate and read schema file
# ─────────────────────────────────────────────
def find_all_schema_files(config: dict) -> dict:
    """
    Scan source ZIP and tmp/raw_data/ for all schema files.
    Returns:
    {
      "responseid": "<content or None>",
      "loops": {"lpPatients": "<content>", ...}
    }
    """
    result = {"responseid": None, "loops": {}}

    raw_dir = Path(__file__).parent.parent / "tmp" / "raw_data"
    source_path = Path(config.get("project_metadata", {}).get("source_path", str(raw_dir)))
    scan_dirs = [raw_dir]
    if source_path.is_dir() and source_path != raw_dir:
        scan_dirs.append(source_path)
    elif source_path.is_file() and source_path.parent != raw_dir:
        scan_dirs.append(source_path.parent)

    explicit_schema = config.get("project_metadata", {}).get("schema_file_path")
    if explicit_schema:
        schema_path = Path(explicit_schema)
        if schema_path.exists():
            try:
                if schema_path.is_file() and schema_path.suffix.lower() == ".zip":
                    with zipfile.ZipFile(schema_path) as zf:
                        for name in zf.namelist():
                            base = Path(name).name
                            if base.lower() == MAIN_SCHEMA_PATTERN:
                                result["responseid"] = zf.read(name).decode(
                                    "utf-8-sig", errors="replace"
                                )
                            elif LOOP_SCHEMA_PATTERN.match(base):
                                loop_name = _detect_loop_name(base)
                                if loop_name and loop_name not in result["loops"]:
                                    result["loops"][loop_name] = zf.read(name).decode(
                                        "utf-8-sig", errors="replace"
                                    )
                            elif SCHEMA_EXCEL_PATTERN.match(base):
                                loop_name = _detect_loop_name(base)
                                if loop_name and loop_name not in result["loops"]:
                                    content = _read_xlsx_schema_bytes(zf.read(name))
                                    if content:
                                        result["loops"][loop_name] = content
                else:
                    content = _read_schema_content(schema_path)
                    if content:
                        loop_name = _detect_loop_name(schema_path.name)
                        if loop_name:
                            result["loops"][loop_name] = content
                        else:
                            result["responseid"] = content
                total = (1 if result["responseid"] else 0) + len(result["loops"])
                print(
                    f"  Schema files found from explicit path: {total} "
                    f"(main={'yes' if result['responseid'] else 'no'}, "
                    f"loops={list(result['loops'].keys())})"
                )
                return result
            except Exception as exc:
                print(f"Warning: Could not read configured schema file {schema_path}: {exc}")

    for scan_dir in scan_dirs:
        if not scan_dir.exists() or not scan_dir.is_dir():
            continue

        for schema_file in scan_dir.glob("*.txt"):
            _classify_and_store(schema_file, result)

        for schema_file in scan_dir.glob("*.xlsx"):
            if SCHEMA_EXCEL_PATTERN.match(schema_file.name):
                content = _read_xlsx_schema(schema_file)
                if content:
                    loop_name = _detect_loop_name(schema_file.name)
                    if loop_name:
                        result["loops"][loop_name] = content

        zip_candidates = list(scan_dir.glob("schemas_*.zip")) + list(scan_dir.glob("*.zip"))
        if source_path.is_file() and source_path.suffix.lower() == ".zip" and source_path.parent == scan_dir:
            zip_candidates.append(source_path)
        seen_zip = set()
        for zip_path in zip_candidates:
            if zip_path in seen_zip:
                continue
            seen_zip.add(zip_path)
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    for name in zf.namelist():
                        base = Path(name).name
                        if base.lower() == MAIN_SCHEMA_PATTERN:
                            if result["responseid"] is None:
                                result["responseid"] = zf.read(name).decode(
                                    "utf-8-sig", errors="replace"
                                )
                        elif LOOP_SCHEMA_PATTERN.match(base):
                            loop_name = _detect_loop_name(base)
                            if loop_name and loop_name not in result["loops"]:
                                result["loops"][loop_name] = zf.read(name).decode(
                                    "utf-8-sig", errors="replace"
                                )
                        elif SCHEMA_EXCEL_PATTERN.match(base):
                            loop_name = _detect_loop_name(base)
                            if loop_name and loop_name not in result["loops"]:
                                content = _read_xlsx_schema_bytes(zf.read(name))
                                if content:
                                    result["loops"][loop_name] = content
            except Exception as exc:
                print(f"  Warning: could not read {zip_path.name}: {exc}")

    total = (1 if result["responseid"] else 0) + len(result["loops"])
    print(
        f"  Schema files found: {total} "
        f"(main={'yes' if result['responseid'] else 'no'}, "
        f"loops={list(result['loops'].keys())})"
    )
    return result


def _detect_loop_name(filename: str) -> str | None:
    """Extract loop name from names like *_lpPatients.txt -> lpPatients."""
    stem = Path(filename).stem
    parts = stem.split("_")
    for part in parts:
        if part.lower().startswith("lp") and len(part) > 2:
            return part
    if stem.lower().startswith("lp"):
        return stem
    return None


def _read_schema_content(path: Path) -> str | None:
    """Read txt/xlsx schema content and normalize as tab-separated text."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8-sig", errors="replace")
    if suffix == ".xlsx":
        return _read_xlsx_schema(path)
    return None


def _classify_and_store(file_path: Path, result: dict):
    """Read a .txt schema file and store it in the proper slot."""
    name = file_path.name.lower()
    if MAIN_SCHEMA_PATTERN in name:
        if result["responseid"] is None:
            result["responseid"] = file_path.read_text(encoding="utf-8-sig", errors="replace")
    elif LOOP_SCHEMA_PATTERN.match(file_path.name):
        loop_name = _detect_loop_name(file_path.name)
        if loop_name and loop_name not in result["loops"]:
            result["loops"][loop_name] = file_path.read_text(
                encoding="utf-8-sig", errors="replace"
            )


def _read_xlsx_schema(path: Path) -> str | None:
    """Read loop schema .xlsx and convert to tab-separated schema text."""
    try:
        df = pd.read_excel(path, dtype=str).fillna("")
        return _xlsx_df_to_tsv(df)
    except Exception as exc:
        print(f"  Warning: could not read xlsx schema {path.name}: {exc}")
        return None


def _read_xlsx_schema_bytes(content: bytes) -> str | None:
    """Read loop schema from xlsx bytes (for zipped schema files)."""
    try:
        df = pd.read_excel(io.BytesIO(content), dtype=str).fillna("")
        return _xlsx_df_to_tsv(df)
    except Exception:
        return None


def _xlsx_df_to_tsv(df: pd.DataFrame) -> str | None:
    """Normalize xlsx column names to expected schema format and return TSV."""
    col_map = {}
    for col in df.columns:
        normalized = str(col).strip().lower().replace(" ", "_")
        if "question_id" in normalized or normalized == "questionid":
            col_map[col] = "Question ID"
        elif "variable_id" in normalized or normalized == "variableid":
            col_map[col] = "Variable ID"
        elif normalized == "type":
            col_map[col] = "Type"
        elif normalized == "start":
            col_map[col] = "Start"
        elif normalized in ("finish", "end"):
            col_map[col] = "Finish"
        elif "answer_code" in normalized or normalized == "answercode":
            col_map[col] = "Answer Code"
        elif "question_label" in normalized or normalized == "questionlabel":
            col_map[col] = "Question Label"
        elif "answer_label" in normalized or normalized == "answerlabel":
            col_map[col] = "Answer Label"

    df = df.rename(columns=col_map)
    required = ["Question ID", "Variable ID", "Type"]
    if not all(col in df.columns for col in required):
        return None
    return df.to_csv(sep="\t", index=False)


# ─────────────────────────────────────────────
# Step 2: Parse schema → structured dict
# ─────────────────────────────────────────────
def parse_schema(schema_content: str) -> dict:
    """
    Parse the Forsta responseid.txt TSV into a structured question map.

    Returns:
    {
      "Q1": {
        "type": "ranking",
        "label": "In your opinion, what are your 3 most important...",
        "variables": ["Q1_1", "Q1_2", ...],
        "answers": {"1": "More effective options...", "2": "..."},
        "analytics": ["mean_rank", "top_box_analysis", ...]
      },
      ...
    }
    """
    lines = schema_content.strip().splitlines()
    questions: dict = {}

    for line in lines[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) < 4:
            continue

        q_id = parts[0].strip()
        var_id = parts[1].strip()
        q_type = parts[2].strip().lower()
        answer_code = parts[5].strip() if len(parts) > 5 else ""
        q_label = parts[6].strip() if len(parts) > 6 else ""
        a_label = parts[7].strip() if len(parts) > 7 else ""

        # Skip loop identifier rows (e.g. lpPatients itself is not a question)
        if var_id and q_id and var_id == q_id and var_id.lower().startswith("lp"):
            continue
        # Skip loop identifier rows (e.g. lpPatients itself is not a question)
        if var_id.lower().startswith("lp") and q_type not in TYPE_TO_ANALYTICS:
            continue
        # Skip malformed/unsupported rows where the variable mirrors the question ID
        if var_id == q_id and (not q_type or q_type not in TYPE_TO_ANALYTICS):
            continue

        # Skip pure metadata
        if any(q_id.startswith(p) for p in METADATA_PREFIXES):
            continue
        if not q_id or q_type in ("date", "time", "quantity") and not q_id.startswith(CONTENT_PREFIXES):
            continue

        if q_id not in questions:
            # Clean label (remove Forsta ^logic^ tags)
            clean_label = q_label
            # Remove ^?'...'...^ conditional display logic (Forsta)
            clean_label = re.sub(r"\^\?'[^']*'[^)]*\)\^", "", clean_label)
            clean_label = re.sub(r"\^\?'[^']*'[^']*'[^']*'\^", "", clean_label)
            # Remove standard ^...^ logic tags
            clean_label = re.sub(r"\^[^^]*\^", "", clean_label)
            # Remove ^IT()? style tags
            clean_label = re.sub(r"\^[^)]+\)\^?", "", clean_label)
            # Remove leading ^ characters
            clean_label = re.sub(r"^\^+", "", clean_label)
            # Remove leading loop placeholder fragments (e.g. "AML patient #")
            clean_label = re.sub(r"^\s*[A-Za-z][A-Za-z\s]*patient\s*#\s*", "", clean_label, flags=re.IGNORECASE)
            # Strip whitespace, commas, question marks
            clean_label = clean_label.strip(" ,?")

            questions[q_id] = {
                "type": q_type,
                "label": clean_label or q_label[:120],
                "variables": [],
                "answers": {},
                "analytics": TYPE_TO_ANALYTICS.get(q_type, []),
                "is_content": q_id.startswith(CONTENT_PREFIXES),
            }

        if var_id and var_id not in questions[q_id]["variables"]:
            questions[q_id]["variables"].append(var_id)

        if answer_code and a_label:
            questions[q_id]["answers"][answer_code] = a_label

    return questions


# ─────────────────────────────────────────────
# Step 3: Infer column filter from schema
# ─────────────────────────────────────────────
def extract_variable_ids(questions: dict) -> dict:
    """
    Build the definitive list of variables to KEEP (content questions only).
    Returns {'keep_variables': [...], 'drop_types': [...]} for use in ingest.
    """
    keep_vars = []
    drop_types = ["date", "time", "quantity", "opentext"]

    for q_id, info in questions.items():
        if not info["is_content"]:
            continue
        if info["type"] in drop_types:
            continue
        keep_vars.extend(info["variables"])

    # Always keep key metadata (exclude respondent identifiers from analytics payload)
    always_keep = ["QCountry", "hWave", "SampleHouse",
                   "SPE", "InLoop1", "InLoop2"]
    keep_vars = list(dict.fromkeys(always_keep + keep_vars))  # dedup, preserve order

    return {
        "keep_variables": keep_vars,
        "drop_types": drop_types,
        "total_content_questions": sum(1 for q in questions.values() if q["is_content"]),
        "total_variables": len(keep_vars),
    }


# ─────────────────────────────────────────────
# Step 4: Build analytics brief for LLM
# ─────────────────────────────────────────────
def build_analytics_brief(questions: dict, config: dict) -> dict:
    """
    Build a structured brief that the LLM will use to:
    - Suggest which modules to run
    - Generate project context for the commentator
    """
    content_qs = {q: info for q, info in questions.items() if info["is_content"]}

    type_counts = {}
    all_analytics = set()
    question_summaries = []

    for q_id, info in content_qs.items():
        t = info["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
        for a in info["analytics"]:
            all_analytics.add(a)
        if info["analytics"]:
            question_summaries.append(
                {
                    "id": q_id,
                    "type": t,
                    "label": info["label"][:100],
                    "suggested_analytics": info["analytics"],
                    "n_answers": len(info["answers"]),
                }
            )

    # Defensive dedupe at brief level
    recommended = _recommend_modules(type_counts, questions)
    seen = set()
    deduped = []
    for r in recommended:
        if r["module"] not in seen:
            deduped.append(r)
            seen.add(r["module"])

    return {
        "project_name": config["project_metadata"]["project_name"],
        "wave": config["project_metadata"]["wave"],
        "question_type_breakdown": type_counts,
        "all_suggested_analytics": sorted(all_analytics),
        "questions": question_summaries,
        "recommended_modules": deduped,
    }


def _recommend_modules(type_counts: dict, questions: dict | None = None) -> list:
    """Heuristic module recommendation - deduplicated."""
    recommendations = []
    seen_modules = set()

    def add(module: str, reason: str):
        if module not in seen_modules:
            recommendations.append({"module": module, "reason": reason})
            seen_modules.add(module)

    single_multi = type_counts.get("single", 0) + type_counts.get("multi", 0)
    if single_multi > 3:
        add(
            "descriptive",
            f"{type_counts.get('single',0)} single + {type_counts.get('multi',0)} multi -> frequency tables",
        )

    if type_counts.get("numericlist", 0) > 0:
        add(
            "descriptive",
            f"{type_counts.get('numericlist',0)} numeric questions -> means and CIs",
        )

    if type_counts.get("grid", 0) > 0:
        add(
            "descriptive",
            f"{type_counts.get('grid',0)} grid/scale questions -> score distributions",
        )

    if type_counts.get("ranking", 0) > 1:
        add(
            "descriptive",
            f"{type_counts.get('ranking',0)} ranking questions -> mean rank, top-box",
        )

    total_vars = sum(type_counts.values())
    if total_vars > 15:
        add("typology", f"{total_vars} variables -> segmentation feasible")

    if type_counts.get("numericlist", 0) + type_counts.get("grid", 0) > 2:
        add("modeling", "numeric/scale mix -> driver analysis and prediction")

    if type_counts.get("ranking", 0) > 0:
        add("regression", "ranking data -> model adoption/barrier drivers")

    # Latent class: large grid or complex multi -> hidden segments
    if type_counts.get("grid", 0) >= 3 or type_counts.get("multi", 0) >= 3:
        add("latent_class", "complex grid/multi data -> latent class for hidden segments")

    # Cas vignettes: if VIG_ columns detected in schema
    if questions:
        vig_vars = [
            q
            for q in questions
            if str(q).startswith(("VIG_", "vignette_", "CAS_"))
            or str(q).lower().startswith(("vig_", "vignette_", "cas_"))
        ]
        if vig_vars:
            add("cas_vignettes", "vignette columns detected -> vignette design/analysis")

    # CBC simulator: only if CBC is also selected
    recommended_modules = [r["module"] for r in recommendations]
    if "cbc" in recommended_modules:
        add("cbc_simulator", "CBC present -> add simulator for share-of-preference")

    return recommendations
def enrich_with_llm(brief: dict, provider: str) -> dict:
    """
    Send the analytics brief to the selected LLM.
    Ask it to:
    1. Validate/adjust module recommendations
    2. Write a 2-sentence project context for the commentator
    3. Flag any questions that need special treatment
    Returns enriched brief with llm_insights added.
    """
    prompt = f"""You are a pharmaceutical market research analytics expert.

Below is a structured brief from a Forsta survey (project: {brief['project_name']}, {brief['wave']}).

QUESTION MIX:
{json.dumps(brief['question_type_breakdown'], indent=2)}

QUESTIONS SUMMARY (first 15):
{json.dumps(brief['questions'][:15], indent=2)}

CURRENTLY SUGGESTED MODULES:
{json.dumps([r['module'] for r in brief['recommended_modules']], indent=2)}

YOUR TASKS:
1. Confirm or adjust the recommended modules. Add any you think are missing given the question types.
2. Write a 2-sentence "project context" that the AI commentator will use to frame its academic and business commentary. What is this study measuring? What decisions will it inform?
3. Flag any specific questions that need non-standard treatment (e.g., patient loop questions P13/P14/P15, conditional logic, open-text that could use NLP).

Respond ONLY as valid JSON:
{{
  "confirmed_modules": ["descriptive", ...],
  "project_context": "This study measures...",
  "analyst_notes": ["Note about Q3...", "Note about P13 patient loop..."],
  "confidence": "high|medium|low"
}}
"""

    api_key_map = {
        "claude": "CLAUDE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "kimi": "KIMI_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
    }
    api_key = os.getenv(api_key_map.get(provider, "CLAUDE_API_KEY"), "")

    if not api_key:
        print(f"  WARNING: No API key for {provider}. LLM enrichment skipped.")
        return {"confirmed_modules": [r["module"] for r in brief["recommended_modules"]],
                "project_context": f"Survey study: {brief['project_name']}, {brief['wave']}.",
                "analyst_notes": [],
                "confidence": "low",
                "note": "LLM not available — using heuristic recommendations only"}

    try:
        if provider == "claude":
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 800,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"]

        elif provider == "openai":
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "gpt-4o-mini", "max_tokens": 800,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]

        elif provider == "kimi":
            resp = requests.post(
                "https://api.moonshot.cn/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "moonshot-v1-8k", "max_tokens": 800,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]

        elif provider == "perplexity":
            resp = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "sonar", "max_tokens": 800,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]

        else:
            return {"error": f"Unknown provider: {provider}"}

        # Parse JSON from response
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())

    except Exception as e:
        print(f"  WARNING: LLM enrichment failed ({e}). Using heuristics only.")
        return {"confirmed_modules": [r["module"] for r in brief["recommended_modules"]],
                "project_context": f"Survey study: {brief['project_name']}, {brief['wave']}.",
                "analyst_notes": [], "confidence": "low", "error": str(e)}


# ─────────────────────────────────────────────
# Step 6: Save outputs
# ─────────────────────────────────────────────
def save_outputs(questions: dict, brief: dict, llm_insights: dict,
                 variable_filter: dict, config: dict):
    tmp_dir = Path(__file__).parent.parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    # 1. Full schema map (used by ingest and statistical engine for label mapping)
    schema_path = tmp_dir / "project_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().isoformat(),
                   "questions": questions,
                   "type_counts": brief.get("question_type_breakdown", {})}, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {schema_path}")

    # 2. Analytics brief (used by launcher for module suggestions + commentator for context)
    brief_enriched = {
        **brief,
        "llm_insights": llm_insights,
        "confirmed_modules": llm_insights.get("confirmed_modules", []),
        "project_context": llm_insights.get("project_context", ""),
        "analyst_notes": llm_insights.get("analyst_notes", []),
        "generated_at": datetime.now().isoformat(),
    }
    brief_path = tmp_dir / "analytics_brief.json"
    with open(brief_path, "w", encoding="utf-8") as f:
        json.dump(brief_enriched, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {brief_path}")

    # 3. Inject recommended modules and variable filter into run_config
    llm_modules = llm_insights.get("confirmed_modules", [])
    if llm_modules and config["project_metadata"].get("workflow_type") == "full":
        config["project_metadata"]["selected_modules"] = llm_modules
        config["project_metadata"]["modules_source"] = "schema_parser_llm"
        print(f"  Modules updated from schema analysis: {llm_modules}")

    # Also inject schema-derived column filter so ingest knows exactly which vars to keep
    config["schema_column_filter"] = variable_filter
    config["project_context"] = llm_insights.get("project_context", "")
    config["analyst_notes"] = llm_insights.get("analyst_notes", [])
    save_config(config)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # Ensure UTF-8 output on Windows terminals
    import io
    if (
        sys.stdout.encoding
        and sys.stdout.encoding.lower() not in ("utf-8", "utf8")
        and hasattr(sys.stdout, "buffer")
    ):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 70)
    print("MODULE 0b: SCHEMA PARSER")
    print("Reads Forsta schema -> builds question map -> suggests analytics")
    print("=" * 70)

    load_env()
    config = load_config()

    provider = config["project_metadata"].get("llm_provider", "claude")
    print(f"\nLLM provider: {provider}")

    # 1. Find and select schema
    print("\n[1/5] Locating schema file(s)...")
    all_schemas = find_all_schema_files(config)

    available = {}
    if all_schemas["responseid"]:
        available["responseid"] = all_schemas["responseid"]
    available.update(all_schemas["loops"])

    if not available:
        print("ERROR: No schema files found.")
        sys.exit(1)

    selected_from_config = config.get("project_metadata", {}).get("selected_schema")
    if len(available) == 1:
        chosen_name, schema_content = next(iter(available.items()))
    elif selected_from_config:
        chosen_name = selected_from_config
        if chosen_name in available:
            schema_content = available[chosen_name]
        else:
            chosen_name, schema_content = next(iter(available.items()))
            print(
                f"  Warning: selected_schema='{selected_from_config}' not found. "
                f"Falling back to '{chosen_name}'."
            )
    elif all_schemas["loops"]:
        chosen_name = next(iter(all_schemas["loops"]))
        schema_content = all_schemas["loops"][chosen_name]
        print(f"  Auto-selected loop schema: {chosen_name}")
        print("  (Override with config.project_metadata.selected_schema)")
    else:
        chosen_name = "responseid"
        schema_content = all_schemas["responseid"]

    config["project_metadata"]["selected_schema"] = chosen_name
    config["project_metadata"]["available_schemas"] = list(available.keys())
    save_config(config)
    print(f"Using schema: {chosen_name}")

    # 2. Parse schema
    print("[2/5] Parsing schema...")
    questions = parse_schema(schema_content)
    content_q = {k: v for k, v in questions.items() if v["is_content"]}
    print(f"  Total questions parsed: {len(questions)}")
    print(f"  Content questions (analytics-relevant): {len(content_q)}")

    type_summary = {}
    for q in content_q.values():
        type_summary[q["type"]] = type_summary.get(q["type"], 0) + 1
    for t, n in sorted(type_summary.items(), key=lambda x: -x[1]):
        print(f"    {t:12s}: {n}")

    # 3. Extract variable filter
    print("[3/5] Building variable filter from schema...")
    variable_filter = extract_variable_ids(questions)
    print(f"  Variables to keep: {variable_filter['total_variables']}")

    # 4. Build analytics brief
    print("[4/5] Building analytics brief...")
    brief = build_analytics_brief(questions, config)
    print(f"  Recommended modules (heuristic): {[r['module'] for r in brief['recommended_modules']]}")

    # 5. LLM enrichment
    print(f"[5/5] Enriching with {provider.upper()}...")
    llm_insights = enrich_with_llm(brief, provider)
    if "confirmed_modules" in llm_insights:
        print(f"  LLM confirmed modules: {llm_insights['confirmed_modules']}")
    if "project_context" in llm_insights:
        print(f"  Project context: {llm_insights['project_context'][:100]}...")

    # 6. Save
    save_outputs(questions, brief, llm_insights, variable_filter, config)

    print("\n" + "=" * 70)
    print("SCHEMA PARSER COMPLETE")
    print("Analyst can review tmp/analytics_brief.json before launching modules.")
    print("=" * 70)


if __name__ == "__main__":
    main()
