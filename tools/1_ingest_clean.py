"""
Ingest & Clean - Module 1.
Downloads raw Forsta data, applies configurable column filtering, and outputs RGPD-safe data.
Outputs: cleaned_codes.csv (codes only), mapping_dict.json
"""

import io
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# ============================================================
# COLUMN FILTER - Edit this to match your Forsta export format
# ============================================================
KEEP_PREFIXES: List[str] = []
DROP_PREFIXES: List[str] = []
DROP_SUFFIXES: List[str] = ["_label", "_text", "_open"]
SPARSITY_THRESHOLD: float = 0.9
PII_PATTERNS: List[str] = ["email", "name", "phone", "address", "ip", "respondent"]
# ============================================================

# DATA_MODE - controls RGPD strictness
# "codes"  (default): Drop all text/label columns. Safe for any LLM provider.
# "labels" : Keep label columns. ONLY use when running fully locally with no cloud LLM.
#            Enables human-readable analysis without code->label mapping step.
DATA_MODE = os.getenv("ANTIGRAVITY_DATA_MODE", "codes")
SUPPORTED_EXTENSIONS = {".zip", ".xlsx", ".xls", ".csv", ".txt", ".tsv"}


def load_env():
    """Load environment variables from project root."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    global DATA_MODE
    DATA_MODE = os.getenv("ANTIGRAVITY_DATA_MODE", "codes").strip().lower() or "codes"


def load_config() -> dict:
    """Load run configuration from tmp/run_config.json."""
    config_path = Path(__file__).parent.parent / "tmp" / "run_config.json"
    if not config_path.exists():
        print("Configuration not found. Run 0_launcher.py first.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict):
    """Save updated configuration."""
    config_path = Path(__file__).parent.parent / "tmp" / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_graph_token(client_id: str, client_secret: str, tenant_id: str) -> str:
    """Get Microsoft Graph token using OAuth2 client credentials flow."""
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
    }
    response = requests.post(token_url, data=payload, timeout=30)
    response.raise_for_status()
    return response.json()["access_token"]


def resolve_sharepoint_file_path(config: dict) -> str:
    """Resolve remote file path used by the Graph /content endpoint."""
    explicit_path = os.getenv("SHAREPOINT_CSV_PATH", "").strip()
    if explicit_path:
        return explicit_path if explicit_path.startswith("/") else f"/{explicit_path}"

    base_path = config.get("project_metadata", {}).get("source_path", "/")
    filename = os.getenv("SHAREPOINT_CSV_FILENAME", "data.csv").strip() or "data.csv"
    if not base_path.endswith("/"):
        base_path += "/"
    return f"{base_path}{filename}"


def load_xlsx_from_zip(zip_path: Path) -> Optional[pd.DataFrame]:
    """
    Load the main survey data xlsx from a Forsta ZIP archive.
    Skips sub-table files: callhistoryinfo, Duration, lpPatients.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            xlsx_files = [
                name
                for name in zf.namelist()
                if name.endswith(".xlsx")
                and not any(
                    skip in name for skip in ["callhistory", "Duration", "lpPatients", "Patients"]
                )
            ]
            if not xlsx_files:
                return None

            main_file = xlsx_files[0]
            print(f"  Reading from zip: {zip_path.name} -> {main_file}")
            with zf.open(main_file) as handle:
                return pd.read_excel(io.BytesIO(handle.read()))
    except Exception as exc:
        print(f"  Warning: could not read zip {zip_path.name}: {exc}")
        return None


def _no_data_error():
    print()
    print("ERROR: No data source found in tmp/raw_data/")
    print("Expected files:")
    print("  tmp/raw_data/p{survey_id}*.zip      <- main Forsta data zip")
    print("  tmp/raw_data/schemas_*.zip           <- Forsta schema zip (optional but recommended)")
    sys.exit(1)


def _load_mapping_from_schema() -> dict:
    """Load code->label mapping from project_schema.json if schema parser has run."""
    schema_path = Path(__file__).parent.parent / "tmp" / "project_schema.json"
    if not schema_path.exists():
        return {}

    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    mapping = {}
    for q_id, info in schema.get("questions", {}).items():
        if info.get("answers"):
            mapping[q_id] = info["answers"]

    return mapping


def _load_excel(path: Path) -> Tuple[pd.DataFrame, str]:
    """Load XLSX/XLS. If multiple sheets, load the largest one."""
    try:
        with pd.ExcelFile(path, engine="openpyxl") as xl:
            if len(xl.sheet_names) == 1:
                sheet = xl.sheet_names[0]
                df = xl.parse(sheet)
            else:
                sheets = {name: xl.parse(name) for name in xl.sheet_names}
                sheet = max(sheets, key=lambda s: len(sheets[s]))
                df = sheets[sheet]
                print(f"  Multiple sheets found - using largest: '{sheet}' ({len(df)} rows)")
    except Exception:
        # Fallback for legacy .xls when openpyxl is not applicable.
        df = pd.read_excel(path)
        sheet = "Sheet1"

    print(f"  Loaded Excel: {path.name} | sheet='{sheet}' | {len(df)} rows x {len(df.columns)} cols")
    return df, "xlsx"


def _load_csv(path: Path) -> Tuple[pd.DataFrame, str]:
    """Load CSV with auto-delimiter detection."""
    import csv as csv_module

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)

    try:
        dialect = csv_module.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter
    except csv_module.Error:
        sep = ","

    df = pd.read_csv(path, sep=sep, encoding="utf-8", encoding_errors="replace")
    print(f"  Loaded CSV: {path.name} | delimiter='{sep}' | {len(df)} rows x {len(df.columns)} cols")
    return df, "csv"


def _load_txt(path: Path) -> Tuple[pd.DataFrame, str]:
    """Load TXT/TSV using delimiter sniffing with tab default."""
    import csv as csv_module

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)

    try:
        dialect = csv_module.Sniffer().sniff(sample, delimiters="\t,;|")
        sep = dialect.delimiter
    except csv_module.Error:
        sep = "\t"

    df = pd.read_csv(path, sep=sep, encoding="utf-8", encoding_errors="replace")
    print(f"  Loaded TXT: {path.name} | delimiter={repr(sep)} | {len(df)} rows x {len(df.columns)} cols")
    return df, "txt"


def _load_zip(path: Path) -> Tuple[pd.DataFrame, str]:
    """Extract ZIP and read the first CSV/XLSX/XLS found inside."""
    import tempfile

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        data_files = [
            n
            for n in names
            if n.lower().endswith((".csv", ".xlsx", ".xls"))
            and not n.startswith("__")
            and not any(skip in n.lower() for skip in ("schema", "datamap", "responseid"))
        ]
        if not data_files:
            # Fallback: accept any CSV/XLSX/XLS if no obvious data file pattern.
            data_files = [
                n for n in names if n.lower().endswith((".csv", ".xlsx", ".xls")) and not n.startswith("__")
            ]
        if not data_files:
            raise FileNotFoundError(f"No CSV/XLSX found inside {path.name}")

        target = sorted(data_files)[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extract(target, tmpdir)
            extracted = Path(tmpdir) / target
            if extracted.suffix.lower() == ".csv":
                df, fmt = _load_csv(extracted)
            else:
                df, fmt = _load_excel(extracted)

    print(f"  Loaded from ZIP: {target} ({len(df)} rows)")
    return df, f"zip/{fmt}"


def load_source(source_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Universal file loader.
    Accepts:
    - Directory: scans for first supported file inside
    - .zip: extracts and reads CSV/XLSX inside
    - .xlsx/.xls: reads directly
    - .csv/.txt/.tsv: reads with delimiter sniffing
    Returns: (DataFrame, detected_format_label)
    """
    path = Path(source_path)

    if path.is_dir():
        candidates: List[Path] = []
        for ext in sorted(SUPPORTED_EXTENSIONS):
            candidates.extend(path.glob(f"*{ext}"))
        # Prefer likely data files over schema/datamap files.
        candidates = sorted(
            candidates,
            key=lambda p: (
                any(tok in p.name.lower() for tok in ("schema", "datamap", "responseid")),
                p.name.lower(),
            ),
        )
        if not candidates:
            raise FileNotFoundError(
                f"No supported files found in {source_path}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        path = candidates[0]
        print(f"  Auto-selected file: {path.name}")

    if not path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    ext = path.suffix.lower()
    if ext == ".zip":
        return _load_zip(path)
    if ext in (".xlsx", ".xls"):
        return _load_excel(path)
    if ext == ".csv":
        return _load_csv(path)
    if ext in (".txt", ".tsv"):
        return _load_txt(path)

    raise ValueError(f"Unsupported file type: {ext}")


def find_loop_data_file(source_path: str, loop_name: str) -> Optional[Path]:
    """
    Find the data file corresponding to a loop schema (e.g. lpPatients).
    Matches files containing loop_name and ending in .xlsx/.csv/.txt.
    """
    path = Path(source_path)
    search_dirs: List[Path] = []
    extensions = (".xlsx", ".xls", ".csv", ".txt", ".tsv")

    if path.is_file() and path.suffix.lower() == ".zip":
        tmp_dir = Path(__file__).parent.parent / "tmp" / "raw_data"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    lower_name = Path(name).name.lower()
                    if loop_name.lower() in lower_name and lower_name.endswith(extensions):
                        dest = tmp_dir / Path(name).name
                        with open(dest, "wb") as f:
                            f.write(zf.read(name))
                        print(f"  Extracted loop data: {dest.name}")
                        return dest
        except Exception as exc:
            print(f"  Warning: could not inspect zip for loop data ({exc})")
        search_dirs = [tmp_dir]
    elif path.is_dir():
        search_dirs = [path]
    else:
        search_dirs = [path.parent]

    for directory in search_dirs:
        if not directory.exists():
            continue
        for candidate in directory.iterdir():
            if not candidate.is_file():
                continue
            lower_name = candidate.name.lower()
            if loop_name.lower() in lower_name and candidate.suffix.lower() in extensions:
                print(f"  Found loop data file: {candidate.name}")
                return candidate

    return None


def find_schema_file(source_dir: Path) -> Optional[Path]:
    """Look for schema/datamap ZIP or CSV alongside the data file."""
    patterns = ["schema*", "datamap*", "*schema*", "*datamap*", "*map*"]
    for pattern in patterns:
        candidates = list(source_dir.glob(pattern + ".zip")) + list(source_dir.glob(pattern + ".csv"))
        if candidates:
            print(f"  Schema file detected: {candidates[0].name}")
            return candidates[0]
    return None


def load_local_fallback() -> Tuple[pd.DataFrame, dict]:
    """
    Load from tmp/raw_data/ in this priority order:
    1. Labels zip if DATA_MODE = "labels"  - OR -  codes zip otherwise
    2. Loose xlsx files
    Exits with actionable error if nothing found.
    """
    raw_dir = Path(__file__).parent.parent / "tmp" / "raw_data"
    if not raw_dir.exists():
        _no_data_error()

    try:
        df, _fmt = load_source(str(raw_dir))
        return df, _load_mapping_from_schema()
    except Exception:
        _no_data_error()


def download_from_sharepoint(config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Download raw Forsta data from SharePoint via Microsoft Graph API.

    Behavior:
    1) If SHAREPOINT_CLIENT_ID and SHAREPOINT_SECRET are set, attempt Graph API download.
    2) If keys are missing or Graph download fails, fall back to local files in tmp/raw_data/.
    3) If no local data exists, exit with explicit error.
    """
    print("Downloading data source...")

    client_id = os.getenv("SHAREPOINT_CLIENT_ID", "").strip()
    client_secret = os.getenv("SHAREPOINT_SECRET", "").strip()

    if client_id and client_secret:
        tenant_id = os.getenv("SHAREPOINT_TENANT_ID", "").strip()
        drive_id = os.getenv("SHAREPOINT_DRIVE_ID", "").strip()

        if not tenant_id or not drive_id:
            print(
                "SharePoint credentials found but SHAREPOINT_TENANT_ID and SHAREPOINT_DRIVE_ID are required for Graph download. "
                "Falling back to local file mode."
            )
            return load_local_fallback()

        remote_path = resolve_sharepoint_file_path(config)
        graph_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:{remote_path}:/content"

        try:
            token = get_graph_token(client_id, client_secret, tenant_id)
            headers = {"Authorization": f"Bearer {token}"}
            resp = requests.get(graph_url, headers=headers, timeout=60)
            resp.raise_for_status()

            fname = remote_path.lower()
            if fname.endswith(".zip"):
                tmp_zip = Path(__file__).parent.parent / "tmp" / "_sp_download.zip"
                tmp_zip.write_bytes(resp.content)
                df = load_xlsx_from_zip(tmp_zip)
                if df is None:
                    raise ValueError("No usable xlsx found in downloaded zip")
            else:
                df = pd.read_excel(io.BytesIO(resp.content))

            print(f"SharePoint download success: {len(df)} rows, {len(df.columns)} columns")
            return df, _load_mapping_from_schema()
        except Exception as exc:
            print(f"SharePoint download failed ({exc}). Falling back to local file mode.")
            return load_local_fallback()

    print(
        "Warning: SHAREPOINT_CLIENT_ID and SHAREPOINT_SECRET are missing in .env. "
        "Falling back to local file mode."
    )
    return load_local_fallback()


def apply_column_filter(df: pd.DataFrame, config: dict = None) -> Tuple[pd.DataFrame, dict]:
    """
    Filter columns using schema-derived variable list (preferred) or manual config.
    In DATA_MODE=labels: keeps text columns too.
    In DATA_MODE=codes: drops all text/open-text columns (RGPD-safe default).
    """
    config = config or {}
    schema_filter = config.get("schema_column_filter", {})
    keep_vars = schema_filter.get("keep_variables", [])

    logs = {
        "kept": [],
        "dropped_not_in_schema": [],
        "dropped_pii": [],
        "dropped_sparse": [],
        "mode": DATA_MODE,
    }

    if keep_vars:
        cols_in_data = [c for c in keep_vars if c in df.columns]
        missing_from_data = [c for c in keep_vars if c not in df.columns]
        if missing_from_data:
            print(f"  Note: {len(missing_from_data)} schema vars not in data (may be loop/conditional)")

        if cols_in_data:
            logs["dropped_not_in_schema"] = [c for c in df.columns if c not in cols_in_data]
            df_filtered = df[cols_in_data].copy()
            print(f"  Schema-driven: kept {len(cols_in_data)} / {len(df.columns)} columns")
        else:
            print("  Schema keep list has no overlap with this file. Falling back to manual filter.")
            keep_vars = []

    if not keep_vars:
        keep_prefixes = tuple(p.lower() for p in KEEP_PREFIXES)
        drop_prefixes = tuple(p.lower() for p in DROP_PREFIXES)
        drop_suffixes = tuple(s.lower() for s in DROP_SUFFIXES)

        candidate_cols = []
        for col in df.columns:
            cl = col.lower()
            if keep_prefixes and not cl.startswith(keep_prefixes):
                logs["dropped_not_in_schema"].append(col)
                continue
            if drop_prefixes and cl.startswith(drop_prefixes):
                logs["dropped_not_in_schema"].append(col)
                continue
            if DATA_MODE == "codes" and drop_suffixes and cl.endswith(drop_suffixes):
                logs["dropped_not_in_schema"].append(col)
                continue
            candidate_cols.append(col)

        df_filtered = df[candidate_cols].copy()
        print(f"  Manual filter: kept {len(candidate_cols)} / {len(df.columns)} columns")

    pii_cols = [c for c in df_filtered.columns if any(p in c.lower() for p in PII_PATTERNS)]
    if pii_cols:
        df_filtered = df_filtered.drop(columns=pii_cols)
        logs["dropped_pii"] = pii_cols
        print(f"  Dropped {len(pii_cols)} PII columns")

    if not df_filtered.empty:
        missing_ratio = df_filtered.isna().mean()
        sparse_cols = missing_ratio[missing_ratio > SPARSITY_THRESHOLD].index.tolist()
        if sparse_cols:
            df_filtered = df_filtered.drop(columns=sparse_cols, errors="ignore").copy()
            logs["dropped_sparse"] = sparse_cols
            print(f"  Dropped {len(sparse_cols)} sparse columns (>{SPARSITY_THRESHOLD * 100:.0f}% empty)")

    logs["kept"] = df_filtered.columns.tolist()
    return df_filtered, logs


def normalize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric-looking columns to numeric dtype. Avoids FutureWarning."""
    for col in df.columns:
        if df[col].dtype == object:
            try:
                converted = pd.to_numeric(df[col], errors="raise")
            except Exception:
                continue
            df = df.copy()
            df[col] = converted
    return df


def update_mapping_dict(mapping_dict: dict, column_filter_logs: dict) -> dict:
    """Add metadata for traceability."""
    enhanced_mapping = mapping_dict.copy() if mapping_dict else {}
    enhanced_mapping["_metadata"] = {
        "created_at": datetime.now().isoformat(),
        "data_mode": DATA_MODE,
        "column_filter": {
            "keep_prefixes": KEEP_PREFIXES,
            "drop_prefixes": DROP_PREFIXES,
            "drop_suffixes": DROP_SUFFIXES,
            "sparsity_threshold": SPARSITY_THRESHOLD,
            "pii_patterns": PII_PATTERNS,
        },
        "filter_logs": column_filter_logs,
    }
    return enhanced_mapping


def apply_ingest_skills(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply enabled skills registered for ingest_clean step."""
    root_dir = Path(__file__).parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    try:
        from skills.skill_loader import apply_skills

        return apply_skills("ingest_clean", df, config)
    except Exception as exc:
        print(f"Skill loader unavailable or failed ({exc}). Continuing without ingest skills.")
        return df


def save_outputs(df_codes: pd.DataFrame, mapping_dict: dict):
    """Save cleaned data and mapping to tmp/."""
    tmp_dir = Path(__file__).parent.parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    codes_path = tmp_dir / "cleaned_codes.csv"
    df_codes.to_csv(codes_path, index=False)
    print(f"Saved: {codes_path}")

    mapping_path = tmp_dir / "mapping_dict.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved: {mapping_path}")


def main():
    """Main ingestion and cleaning workflow."""
    print("=" * 70)
    print("MODULE 1: INGEST & CLEAN")
    print("=" * 70)

    load_env()
    config = load_config()

    print(f"DATA_MODE={DATA_MODE}")

    config["run_info"]["status"] = "ingesting"
    config["run_info"]["ingest_started_at"] = datetime.now().isoformat()
    save_config(config)

    source_path = config["project_metadata"].get("source_path", "tmp/raw_data")
    selected_schema = config.get("project_metadata", {}).get("selected_schema", "responseid")
    print(f"Loading from: {source_path}")
    print(f"Selected schema: {selected_schema}")

    detected_format = "unknown"
    try:
        if selected_schema != "responseid" and str(selected_schema).startswith("lp"):
            loop_file = find_loop_data_file(source_path, selected_schema)
            if loop_file:
                print(f"Using loop data file: {loop_file.name}")
                df_raw, detected_format = load_source(str(loop_file))
                config["run_info"]["loop_data_file"] = loop_file.name
            else:
                print(
                    f"WARNING: Loop data file for '{selected_schema}' not found; using main source."
                )
                df_raw, detected_format = load_source(source_path)
        else:
            df_raw, detected_format = load_source(source_path)

        mapping_dict = _load_mapping_from_schema()
    except Exception as exc:
        print(f"Local source load failed ({exc}). Trying SharePoint/local fallback...")
        df_raw, mapping_dict = download_from_sharepoint(config)
        detected_format = "sharepoint_or_fallback"

    config["run_info"]["detected_format"] = detected_format

    source_obj = Path(source_path)
    schema_candidate: Optional[Path] = None
    if source_obj.exists() and source_obj.is_dir():
        schema_candidate = find_schema_file(source_obj)
    elif source_obj.exists() and source_obj.is_file():
        schema_candidate = find_schema_file(source_obj.parent)
    else:
        default_raw = Path(__file__).parent.parent / "tmp" / "raw_data"
        if default_raw.exists():
            schema_candidate = find_schema_file(default_raw)

    if schema_candidate is not None:
        config.setdefault("project_metadata", {})["schema_file_path"] = str(schema_candidate)
        config["run_info"]["schema_file_detected"] = schema_candidate.name

    save_config(config)
    print(f"Raw dataset: {len(df_raw)} rows, {len(df_raw.columns)} columns")

    df_filtered, filter_logs = apply_column_filter(df_raw, config)
    df_filtered = normalize_data_types(df_filtered)

    df_skill_applied = apply_ingest_skills(df_filtered, config)
    print(f"Final dataset: {len(df_skill_applied)} rows, {len(df_skill_applied.columns)} columns")

    mapping_dict = update_mapping_dict(mapping_dict, filter_logs)
    save_outputs(df_skill_applied, mapping_dict)

    config["run_info"]["status"] = "cleaned"
    config["run_info"]["rows_processed"] = len(df_skill_applied)
    config["run_info"]["columns_processed"] = len(df_skill_applied.columns)
    config["run_info"]["ingest_completed_at"] = datetime.now().isoformat()
    save_config(config)

    print("\n" + "=" * 70)
    print("INGEST & CLEAN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
