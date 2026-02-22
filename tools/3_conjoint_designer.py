"""
Conjoint Designer - standalone design utility.

Generates a CBC/conjoint design matrix from analyst-defined attributes in:
  tmp/conjoint_attributes.json

Outputs:
  tmp/conjoint_design.csv
  tmp/conjoint_choice_sets.csv
"""

import json
import math
from itertools import product as iterproduct
from pathlib import Path

import numpy as np
import pandas as pd


def load_or_create_attributes(attr_path: Path) -> dict:
    """Load analyst-provided attributes or write a default template."""
    if attr_path.exists():
        with open(attr_path, encoding="utf-8") as f:
            return json.load(f)

    defaults = {
        "efficacy": ["high", "medium", "low"],
        "safety": ["strong", "moderate"],
        "administration": ["oral", "iv", "sc"],
        "price": ["low", "high"],
        "evidence": ["real_world", "no_real_world"],
    }
    with open(attr_path, "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=2, ensure_ascii=False)
    return defaults


def build_fractional_design(attributes: dict, n_profiles: int = None, seed: int = 42) -> pd.DataFrame:
    """Build a simple fractional factorial profile table using random sample fallback."""
    levels = [vals for vals in attributes.values()]
    full = list(iterproduct(*levels))
    full_n = len(full)

    if n_profiles is None:
        n_profiles = max(12, min(64, int(math.ceil(math.sqrt(full_n) * 2))))

    if full_n <= n_profiles:
        selected = full
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(full_n, size=n_profiles, replace=False)
        selected = [full[i] for i in idx]

    design = pd.DataFrame(selected, columns=list(attributes.keys()))
    design.insert(0, "profile_id", range(1, len(design) + 1))
    return design


def to_choice_sets(design_df: pd.DataFrame, alternatives_per_set: int = 3) -> pd.DataFrame:
    """Group profiles into choice sets for survey programming."""
    rows = []
    profiles = design_df.to_dict(orient="records")
    set_id = 1
    alt_id = 1
    for i, profile in enumerate(profiles):
        if i % alternatives_per_set == 0 and i > 0:
            set_id += 1
            alt_id = 1
        row = {"choice_set_id": set_id, "alternative_id": alt_id, **profile}
        rows.append(row)
        alt_id += 1
    return pd.DataFrame(rows)


def main():
    root = Path(__file__).parent.parent
    tmp_dir = root / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    attr_path = tmp_dir / "conjoint_attributes.json"
    design_path = tmp_dir / "conjoint_design.csv"
    choice_path = tmp_dir / "conjoint_choice_sets.csv"

    print("=" * 70)
    print("MODULE 3X: CONJOINT DESIGNER")
    print("=" * 70)

    attributes = load_or_create_attributes(attr_path)
    design_df = build_fractional_design(attributes)
    choice_df = to_choice_sets(design_df, alternatives_per_set=3)

    design_df.to_csv(design_path, index=False, encoding="utf-8")
    choice_df.to_csv(choice_path, index=False, encoding="utf-8")

    print(f"Attributes source: {attr_path.name}")
    print(f"Profiles generated: {len(design_df)}")
    print(f"Choice sets generated: {choice_df['choice_set_id'].nunique()}")
    print(f"Saved design: {design_path.name}")
    print(f"Saved choice sets: {choice_path.name}")


if __name__ == "__main__":
    main()
