"""
Antigravity — FastAPI Bridge
Port: 8502  (Streamlit stays on 8501)

Bridges the React frontend (localhost:5173) to the Python pipeline.

Endpoints
─────────
POST  /api/upload           Receive a survey file, save to tmp/raw_data/
POST  /api/detect-schema    Scan the uploaded file for schema files
POST  /api/run              Write run_config.json, launch pipeline, stream logs (SSE)
POST  /api/feedback         Submit analyst feedback → SharePoint / local fallback
GET   /api/download/{name}  Serve a file from tmp/ as download
GET   /api/status           Return current pipeline run status

Start
─────
.venv\\Scripts\\uvicorn.exe api_server:app --port 8502 --reload
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ── Bootstrap ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
TMP = ROOT / "tmp"
TMP.mkdir(exist_ok=True)
(TMP / "raw_data").mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")

# Add tools/ to path so we can import schema parser directly
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Antigravity API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # React dev server + any deployed origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared run state ────────────────────────────────────────────────────────
_run_state: dict = {
    "running": False,
    "complete": False,
    "success": False,
    "step": "idle",
    "log": [],
    "started_at": None,
    "finished_at": None,
}


# ═══════════════════════════════════════════════════════════════════════════
# 1.  POST /api/upload
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Receive a survey ZIP/XLSX/CSV/TXT and save it to tmp/raw_data/.
    Returns the server-side path so subsequent calls can reference it.
    """
    dest = TMP / "raw_data" / file.filename
    content = await file.read()
    dest.write_bytes(content)
    size_kb = len(content) / 1024
    return {
        "filename": file.filename,
        "path": str(dest),
        "size_kb": round(size_kb, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2.  POST /api/detect-schema
# ═══════════════════════════════════════════════════════════════════════════
class DetectSchemaRequest(BaseModel):
    path: str


@app.post("/api/detect-schema")
async def detect_schema(req: DetectSchemaRequest):
    """
    Scan the uploaded file for Forsta schema files (responseid.txt + lp*.txt/xlsx).
    Returns schema names, parsed question lists, and data preview rows.
    """
    try:
        from tools.schema_parser import find_all_schema_files, parse_schema

        config = {"project_metadata": {"source_path": req.path}}
        all_schemas = find_all_schema_files(config)

        available: dict[str, str] = {}
        if all_schemas.get("responseid"):
            available["responseid"] = all_schemas["responseid"]
        available.update(all_schemas.get("loops", {}))

        if not available:
            return {"schemas": [], "questions": {}, "preview": []}

        # Parse all schemas → question lists
        questions_by_schema: dict[str, list] = {}
        for name, content in available.items():
            parsed = parse_schema(content)
            questions_by_schema[name] = [
                {
                    "variable_id": var_id,
                    "question_id": var_id,   # outer key IS the question id
                    "label": (info["label"] if isinstance(info, dict) else str(info))[:120],
                    "type": info["type"] if isinstance(info, dict) else "unknown",
                    "is_content": info.get("is_content", True) if isinstance(info, dict) else True,
                }
                for var_id, info in parsed.items()
            ]

        # Data preview from first loop schema (or responseid)
        preview_rows = _load_preview(req.path, next(iter(available.keys())))

        return {
            "schemas": list(available.keys()),
            "default_schema": next(
                (k for k in available if k != "responseid"), next(iter(available))
            ),
            "questions": questions_by_schema,
            "preview": preview_rows,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _load_preview(source_path: str, schema_name: str, max_rows: int = 5) -> list[dict]:
    """Load a small preview of the data file matching the schema."""
    import io as io_mod

    path = Path(source_path)
    loop_pat = re.compile(rf"{re.escape(schema_name)}\.(xlsx|csv|txt)$", re.IGNORECASE)

    def _read(raw_bytes_or_path, ext: str) -> list[dict]:
        import pandas as _pd
        try:
            if isinstance(raw_bytes_or_path, bytes):
                buf = io_mod.BytesIO(raw_bytes_or_path)
                df = _pd.read_excel(buf, nrows=max_rows) if ext in (".xlsx", ".xls") \
                    else _pd.read_csv(buf, nrows=max_rows)
            else:
                df = _pd.read_excel(raw_bytes_or_path, nrows=max_rows) if ext in (".xlsx", ".xls") \
                    else _pd.read_csv(raw_bytes_or_path, nrows=max_rows)
            return df.fillna("").astype(str).to_dict(orient="records")
        except Exception:
            return []

    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if loop_pat.search(Path(name).name):
                        ext = Path(name).suffix.lower()
                        return _read(zf.read(name), ext)
        except Exception:
            pass
    elif path.is_dir():
        for f in path.iterdir():
            if loop_pat.search(f.name):
                return _read(str(f), f.suffix.lower())

    return []


# ═══════════════════════════════════════════════════════════════════════════
# 3.  POST /api/run  (SSE log stream)
# ═══════════════════════════════════════════════════════════════════════════
class RunRequest(BaseModel):
    source_path: str
    selected_schema: str = "responseid"
    selected_columns: list[str] = []
    selected_modules: list[str] = ["descriptive"]
    goal: str = "descriptive"
    k_segments: int = 4
    target_variable: Optional[str] = None
    llm_provider: str = "openai"
    project_name: str = "Survey"
    wave: str = ""
    generate_notebook: bool = True
    generate_powerbi: bool = True


@app.post("/api/run")
async def run_pipeline(req: RunRequest):
    """
    Write run_config.json, launch 0_launcher.py as subprocess, stream stdout via SSE.
    Call GET /api/status to poll without SSE, or consume this stream directly.
    """
    global _run_state

    if _run_state["running"]:
        raise HTTPException(status_code=409, detail="Pipeline already running")

    # Build config
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    config = {
        "project_metadata": {
            "project_name": req.project_name,
            "wave": req.wave or "Unknown",
            "source_path": req.source_path,
            "selected_schema": req.selected_schema,
            "selected_columns": req.selected_columns,
            "selected_modules": req.selected_modules,
            "goal": req.goal,
            "k_segments": req.k_segments,
            "target_variable": req.target_variable,
            "llm_provider": req.llm_provider,
            "delivery_format": "json",
            "workflow_type": "full",
            "generate_notebook": req.generate_notebook,
            "generate_powerbi": req.generate_powerbi,
            "modules_source": "api_request",
        },
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "status": "starting",
        },
    }

    (TMP / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Build subprocess command
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "0_launcher.py"),
        "--non-interactive",
        "--provider", req.llm_provider,
        "--source", req.source_path,
        "--modules", *req.selected_modules,
        "--goal", req.goal,
        "--k", str(req.k_segments),
    ]
    if req.target_variable:
        cmd += ["--target", req.target_variable]

    return StreamingResponse(
        _stream_pipeline(cmd),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_pipeline(cmd: list[str]) -> AsyncGenerator[str, None]:
    """
    Run pipeline subprocess and yield SSE-formatted log lines.
    Uses a thread + queue to avoid asyncio subprocess issues on Windows.
    """
    global _run_state

    _run_state.update({
        "running": True,
        "complete": False,
        "success": False,
        "step": "starting",
        "log": [],
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
    })

    STEP_MARKERS = {
        "schema_parser": "schema",
        "ingest_clean": "ingest",
        "statistical_engine": "statistics",
        "ai_commentator": "commentary",
        "export_deliver": "export",
        "notebook_generator": "notebook",
        "powerbi_export": "powerbi",
    }

    import queue
    import threading

    line_queue: queue.Queue = queue.Queue()
    _DONE = object()  # sentinel

    def _run_in_thread():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
                encoding="utf-8",
                errors="replace",
            )
            for line in proc.stdout:
                line_queue.put(line.rstrip())
            proc.wait()
            line_queue.put((_DONE, proc.returncode))
        except Exception as exc:
            line_queue.put((_DONE, -1, str(exc)))

    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()

    try:
        while True:
            try:
                item = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: line_queue.get(timeout=0.1)
                )
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue

            # Sentinel = subprocess finished
            if isinstance(item, tuple) and item[0] is _DONE:
                returncode = item[1]
                error_msg = item[2] if len(item) > 2 else None
                break

            line = item
            _run_state["log"].append(line)

            for marker, step_name in STEP_MARKERS.items():
                if marker.replace("_", "") in line.lower().replace("_", ""):
                    _run_state["step"] = step_name

            payload = json.dumps({"type": "log", "line": line, "step": _run_state["step"]})
            yield f"data: {payload}\n\n"

        if error_msg:
            raise RuntimeError(error_msg)

        success = returncode == 0
        _run_state.update({
            "running": False, "complete": True, "success": success,
            "step": "done", "finished_at": datetime.now().isoformat(),
        })

        outputs = _collect_outputs()
        final_payload = json.dumps({
            "type": "complete",
            "success": success,
            "returncode": returncode,
            "outputs": outputs,
        })
        yield f"data: {final_payload}\n\n"

    except Exception as exc:
        _run_state.update({
            "running": False, "complete": True, "success": False,
            "step": "error", "finished_at": datetime.now().isoformat(),
        })
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"


def _collect_outputs() -> dict:
    """Return names of generated output files so the frontend can build download links."""
    outputs = {}
    payload = TMP / "final_delivery_payload.json"
    if payload.exists():
        outputs["json_report"] = payload.name

    notebooks = sorted((TMP / "notebooks").glob("*.ipynb")) if (TMP / "notebooks").exists() else []
    if notebooks:
        outputs["notebook"] = notebooks[-1].name

    pbi_dir = TMP / "powerbi"
    if pbi_dir.exists() and any(pbi_dir.iterdir()):
        outputs["powerbi_dir"] = "powerbi"

    return outputs


# ═══════════════════════════════════════════════════════════════════════════
# 4.  POST /api/feedback
# ═══════════════════════════════════════════════════════════════════════════
class FeedbackRequest(BaseModel):
    analyst: str = "anonymous"
    comment: str


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Submit analyst feedback. Tries SharePoint via sharepoint_feedback module,
    falls back to local JSON file in feedback/.
    """
    if not req.comment.strip():
        raise HTTPException(status_code=422, detail="Comment cannot be empty")

    try:
        from tools.sharepoint_feedback import submit_feedback as sp_feedback

        result = sp_feedback(analyst=req.analyst, comment=req.comment)
        return {"success": True, "method": result.get("method", "sharepoint"), "detail": result}

    except Exception as sp_exc:
        # Graceful local fallback
        try:
            fb_dir = ROOT / "feedback"
            fb_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            entry = {
                "timestamp": ts,
                "analyst": req.analyst,
                "comment": req.comment.strip(),
            }
            (fb_dir / f"feedback_{ts}.json").write_text(
                json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return {"success": True, "method": "local", "note": str(sp_exc)}
        except Exception as local_exc:
            raise HTTPException(
                status_code=500,
                detail=f"SharePoint failed: {sp_exc}; local fallback also failed: {local_exc}",
            )


# ═══════════════════════════════════════════════════════════════════════════
# 5.  GET /api/download/{name}
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/download/{name:path}")
async def download_file(name: str):
    """
    Serve a file from tmp/ or tmp/notebooks/ or tmp/powerbi/.
    name can include a subpath, e.g.  notebooks/survey_20260222.ipynb
    """
    # Security: prevent directory traversal
    candidate = (TMP / name).resolve()
    if not str(candidate).startswith(str(TMP.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    if not candidate.exists() or not candidate.is_file():
        # Try notebooks/ subfolder as shortcut for bare .ipynb names
        alt = TMP / "notebooks" / name
        if alt.exists() and alt.is_file():
            candidate = alt
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {name}")

    return FileResponse(
        path=str(candidate),
        filename=candidate.name,
        media_type="application/octet-stream",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6.  GET /api/status
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/status")
async def get_status():
    """Poll current pipeline run state (alternative to SSE stream)."""
    return {
        "running": _run_state["running"],
        "complete": _run_state["complete"],
        "success": _run_state["success"],
        "step": _run_state["step"],
        "log_lines": len(_run_state["log"]),
        "last_lines": _run_state["log"][-20:] if _run_state["log"] else [],
        "started_at": _run_state["started_at"],
        "finished_at": _run_state["finished_at"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7.  GET /  (health check)
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/")
async def health():
    return {
        "service": "Antigravity API",
        "version": "1.0.0",
        "status": "ok",
        "docs": "/docs",
    }


# ── Launch ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8502, reload=True)
