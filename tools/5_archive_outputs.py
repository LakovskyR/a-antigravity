"""
Archive Outputs - pipeline step.
Uploads final payload to SharePoint only (no git operations).
"""

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "tmp" / "run_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict):
    config_path = Path(__file__).parent.parent / "tmp" / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_graph_token(client_id: str, client_secret: str, tenant_id: str) -> str:
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


def upload_final_payload(config: dict) -> bool:
    payload_path = Path(__file__).parent.parent / "tmp" / "final_delivery_payload.json"
    if not payload_path.exists():
        print("No final payload found to archive.")
        return False

    client_id = os.getenv("SHAREPOINT_CLIENT_ID", "").strip()
    client_secret = os.getenv("SHAREPOINT_SECRET", "").strip()
    tenant_id = os.getenv("SHAREPOINT_TENANT_ID", "").strip()
    drive_id = os.getenv("SHAREPOINT_DRIVE_ID", "").strip()

    if not all([client_id, client_secret, tenant_id, drive_id]):
        print("SharePoint archive skipped (missing SHAREPOINT_CLIENT_ID/SHAREPOINT_SECRET/SHAREPOINT_TENANT_ID/SHAREPOINT_DRIVE_ID).")
        return False

    meta = config.get("project_metadata", {})
    project = meta.get("project_name", "Unknown")
    wave = meta.get("wave", "Unknown")

    default_remote_path = f"/Analytics/{project}/{wave}/final_delivery_payload.json"
    remote_path = os.getenv("SHAREPOINT_ARCHIVE_PATH", default_remote_path).strip()
    if not remote_path.startswith("/"):
        remote_path = f"/{remote_path}"

    try:
        token = get_graph_token(client_id, client_secret, tenant_id)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        upload_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:{remote_path}:/content"
        with open(payload_path, "rb") as f:
            response = requests.put(upload_url, headers=headers, data=f.read(), timeout=60)
        response.raise_for_status()
        print(f"Archived payload to SharePoint path: {remote_path}")
        return True
    except Exception as exc:
        print(f"SharePoint archive failed: {exc}")
        return False


def main():
    print("=" * 70)
    print("MODULE 5A: ARCHIVE OUTPUTS")
    print("=" * 70)

    load_env()
    config = load_config()

    config["run_info"]["status"] = "archiving_outputs"
    save_config(config)

    ok = upload_final_payload(config)

    config["run_info"]["status"] = "outputs_archived" if ok else "outputs_not_archived"
    save_config(config)

    print("=" * 70)
    print("ARCHIVE OUTPUTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
