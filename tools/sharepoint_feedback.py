"""
SharePoint Feedback Writeback
Appends analyst suggestions to a Markdown table in a SharePoint file.

Table format in analyst_feedback.md:
| Date | Analyst | Comment | Status |
|------|---------|---------|--------|
| 2026-02-22 14:30 | Marie D. | Add NPS question support | todo |

Requires in .env:
  SHAREPOINT_CLIENT_ID
  SHAREPOINT_TENANT_ID
  SHAREPOINT_SECRET
  SHAREPOINT_FEEDBACK_SITE_ID
  SHAREPOINT_FEEDBACK_FOLDER   (e.g. "Shared Documents/Antigravity/feedback")
  SHAREPOINT_FEEDBACK_FILE     (e.g. "analyst_feedback.md")
"""

import json
import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
CLIENT_ID   = os.getenv("SHAREPOINT_CLIENT_ID", "")
TENANT_ID   = os.getenv("SHAREPOINT_TENANT_ID", "")
SECRET      = os.getenv("SHAREPOINT_SECRET", "")
SITE_ID     = os.getenv("SHAREPOINT_FEEDBACK_SITE_ID", "")
FOLDER      = os.getenv("SHAREPOINT_FEEDBACK_FOLDER", "Shared Documents/Antigravity/feedback")
FILENAME    = os.getenv("SHAREPOINT_FEEDBACK_FILE", "analyst_feedback.md")

TABLE_HEADER = "| Date | Analyst | Comment | Status |\n|------|---------|---------|--------|\n"
FALLBACK_DIR = Path(__file__).parent.parent / "feedback"

# ── Auth ──────────────────────────────────────────────────────────────────────

def _get_token() -> str | None:
    """Get Microsoft Graph access token via client credentials flow."""
    if not all([CLIENT_ID, TENANT_ID, SECRET]):
        return None
    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    data = {
        "grant_type":    "client_credentials",
        "client_id":     CLIENT_ID,
        "client_secret": SECRET,
        "scope":         "https://graph.microsoft.com/.default",
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        print(f"  [feedback] Auth error: {e}")
        return None


# ── SharePoint file read/write ────────────────────────────────────────────────

def _graph_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "text/plain; charset=utf-8"}


def _file_url(token: str) -> str:
    return (
        f"https://graph.microsoft.com/v1.0/sites/{SITE_ID}"
        f"/drive/root:/{FOLDER}/{FILENAME}:/content"
    )


def _read_current_content(token: str) -> str:
    """Download current content of feedback.md from SharePoint."""
    try:
        r = requests.get(_file_url(token), headers=_graph_headers(token), timeout=10)
        if r.status_code == 200:
            return r.text
        elif r.status_code == 404:
            return ""  # file doesn't exist yet
        else:
            print(f"  [feedback] Read error {r.status_code}: {r.text[:200]}")
            return ""
    except Exception as e:
        print(f"  [feedback] Read error: {e}")
        return ""


def _write_content(token: str, content: str) -> bool:
    """Upload updated content to SharePoint."""
    try:
        r = requests.put(
            _file_url(token),
            headers=_graph_headers(token),
            data=content.encode("utf-8"),
            timeout=15,
        )
        return r.status_code in (200, 201)
    except Exception as e:
        print(f"  [feedback] Write error: {e}")
        return False


# ── Table manipulation ────────────────────────────────────────────────────────

def _escape_md(text: str) -> str:
    """Escape pipe characters so they don't break the Markdown table."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _build_new_row(analyst: str, comment: str, status: str = "todo") -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"| {ts} | {_escape_md(analyst)} | {_escape_md(comment)} | {status} |\n"


def _ensure_table_exists(content: str) -> str:
    """
    If the file is empty or has no table header, prepend the standard header.
    Otherwise return as-is.
    """
    if TABLE_HEADER.strip() in content:
        return content
    # Prepend header + existing content
    prefix = (
        "# Antigravity — Analyst Feedback\n\n"
        f"{TABLE_HEADER}"
    )
    return prefix + content


def _append_row(content: str, new_row: str) -> str:
    """Append a new table row after the last existing table row."""
    lines = content.splitlines(keepends=True)
    # Find last table row (starts with |)
    last_table_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("|"):
            last_table_idx = i
    if last_table_idx >= 0:
        lines.insert(last_table_idx + 1, new_row)
    else:
        lines.append("\n" + new_row)
    return "".join(lines)


# ── Fallback: local file ──────────────────────────────────────────────────────

def _save_local_fallback(analyst: str, comment: str, status: str = "todo"):
    """Save to local feedback/ folder as JSON when SharePoint is unavailable."""
    FALLBACK_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    entry = {
        "timestamp": ts,
        "analyst": analyst,
        "comment": comment,
        "status": status,
    }
    path = FALLBACK_DIR / f"suggestion_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
    print(f"  [feedback] Saved locally: {path.name}")


# ── Public API ────────────────────────────────────────────────────────────────

def submit_feedback(analyst: str, comment: str, status: str = "todo") -> dict:
    """
    Main entry point. Submit analyst feedback.
    Tries SharePoint first, falls back to local JSON.

    Returns:
        {"success": True, "method": "sharepoint"|"local", "message": "..."}
    """
    if not comment.strip():
        return {"success": False, "method": None, "message": "Empty comment — nothing saved"}

    analyst = analyst.strip() or "Anonymous"

    # Try SharePoint
    if all([CLIENT_ID, TENANT_ID, SECRET, SITE_ID]):
        token = _get_token()
        if token:
            current = _read_current_content(token)
            current = _ensure_table_exists(current)
            new_row = _build_new_row(analyst, comment, status)
            updated = _append_row(current, new_row)
            ok = _write_content(token, updated)
            if ok:
                print(f"  [feedback] ✅ Saved to SharePoint: {FOLDER}/{FILENAME}")
                return {
                    "success": True,
                    "method": "sharepoint",
                    "message": f"Saved to SharePoint — {FOLDER}/{FILENAME}",
                }
            else:
                print("  [feedback] SharePoint write failed — falling back to local")
        else:
            print("  [feedback] No token — falling back to local")
    else:
        print("  [feedback] SharePoint not configured — saving locally")

    # Fallback
    _save_local_fallback(analyst, comment, status)
    return {
        "success": True,
        "method": "local",
        "message": "Saved locally in feedback/ folder (SharePoint not configured)",
    }


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = submit_feedback(
        analyst="Test Analyst",
        comment="This is a test feedback entry from CLI",
        status="todo",
    )
    print(result)
