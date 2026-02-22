"""
Skill Commit - pipeline step.
Stages and commits changes in /skills only.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "tmp" / "run_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict):
    config_path = Path(__file__).parent.parent / "tmp" / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def run_git(args):
    root = Path(__file__).parent.parent
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=root)


def in_git_repo() -> bool:
    return run_git(["rev-parse", "--git-dir"]).returncode == 0


def list_skill_changes() -> str:
    result = run_git(["status", "--porcelain", "skills"])
    return result.stdout.strip()


def infer_skill_name_description() -> tuple:
    changed = run_git(["diff", "--name-only", "--", "skills"]).stdout.strip().splitlines()
    if not changed:
        changed = run_git(["diff", "--cached", "--name-only", "--", "skills"]).stdout.strip().splitlines()

    if not changed:
        return "skills_update", "Registry or skill configuration update"

    name = Path(changed[0]).stem.replace("_", " ").title()
    return name, "Skill registry/capability update"


def commit_skills() -> str:
    add_result = run_git(["add", "skills"])
    if add_result.returncode != 0:
        raise RuntimeError(add_result.stderr.strip() or "git add failed")

    diff_check = run_git(["diff", "--cached", "--quiet", "--", "skills"])
    if diff_check.returncode == 0:
        return ""

    skill_name, description = infer_skill_name_description()
    commit_msg = f"[SKILL] Added: {skill_name} - {description}"

    commit_result = run_git(["commit", "-m", commit_msg])
    if commit_result.returncode != 0:
        raise RuntimeError(commit_result.stderr.strip() or "git commit failed")

    sha_result = run_git(["rev-parse", "HEAD"])
    if sha_result.returncode != 0:
        raise RuntimeError(sha_result.stderr.strip() or "git rev-parse failed")

    return sha_result.stdout.strip()


def main():
    print("=" * 70)
    print("MODULE 5B: SKILL COMMIT")
    print("=" * 70)

    config = load_config()
    config.setdefault("run_info", {})["status"] = "committing_skills"
    save_config(config)

    if not in_git_repo():
        print("Not a git repository. Skipping skill commit.")
        config["run_info"]["status"] = "skills_not_committed"
        save_config(config)
        return

    if not list_skill_changes():
        print("No changes under /skills to commit.")
        config["run_info"]["status"] = "skills_no_changes"
        save_config(config)
        return

    try:
        commit_sha = commit_skills()
        if commit_sha:
            print(f"Skills committed: {commit_sha[:8]}")
            config["run_info"]["skill_commit_sha"] = commit_sha
            config["run_info"]["status"] = "skills_committed"
        else:
            print("No staged /skills changes to commit.")
            config["run_info"]["status"] = "skills_no_changes"
    except Exception as exc:
        print(f"Skill commit failed: {exc}")
        config["run_info"]["status"] = "skills_commit_failed"

    config["run_info"]["skill_commit_at"] = datetime.now().isoformat()
    save_config(config)

    print("=" * 70)
    print("SKILL COMMIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
