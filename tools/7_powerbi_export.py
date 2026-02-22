"""
Power BI Export - Module 7.
Generates Power BI-ready files: flat CSV, summary CSV, DAX measures, and Power Query M.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def _best_key_metric(result: dict) -> str:
    """Choose a compact key metric string from a module result object."""
    if not isinstance(result, dict):
        return ""
    for key in ("optimal_k", "r_squared", "silhouette_score", "accuracy", "f1"):
        if key in result:
            return str(result.get(key, ""))[:50]
    return ""


def main():
    tmp = Path(__file__).parent.parent / "tmp"
    pbi_dir = tmp / "powerbi"
    pbi_dir.mkdir(parents=True, exist_ok=True)

    data_path = tmp / "cleaned_codes.csv"
    if not data_path.exists():
        print("No cleaned_codes.csv found - skipping Power BI export")
        return

    df = pd.read_csv(data_path)

    # 1) data_model.csv - flat denormalized dataset
    df.to_csv(pbi_dir / "data_model.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved: data_model.csv ({len(df)} rows)")

    # 2) statistical_summary.csv
    results_path = tmp / "statistical_results.json"
    summary_rows = []
    if results_path.exists():
        try:
            with open(results_path, encoding="utf-8") as f:
                results = json.load(f)
            for module, result in results.get("results", {}).items():
                summary_rows.append(
                    {
                        "module": module,
                        "status": result.get("status", "ok") if isinstance(result, dict) else "ok",
                        "n_observations": result.get("n_observations", "") if isinstance(result, dict) else "",
                        "key_metric": _best_key_metric(result),
                    }
                )
        except Exception as exc:
            print(f"  Warning: could not parse statistical_results.json ({exc})")

    pd.DataFrame(summary_rows).to_csv(
        pbi_dir / "statistical_summary.csv", index=False, encoding="utf-8-sig"
    )
    print("  Saved: statistical_summary.csv")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    target_col = numeric_cols[-1] if numeric_cols else "metric"
    group_col = next(
        (c for c in df.columns if "country" in c.lower() or "segment" in c.lower()),
        "group",
    )

    # 3) measures.dax template
    dax = f"""// Antigravity - Auto-generated DAX Measures
// Generated: {datetime.now().strftime('%Y-%m-%d')}
// Paste each block into Power BI Desktop > New Measure

Market Share % =
DIVIDE(
    COUNTROWS(FILTER(Survey, Survey[brand_prescribed] = SELECTEDVALUE(Brands[brand]))),
    COUNTROWS(Survey),
    0
) * 100

Avg {target_col} =
AVERAGE(Survey[{target_col}])

Top2Box % =
VAR MaxVal = MAXX(Survey, Survey[{target_col}])
RETURN
DIVIDE(
    COUNTROWS(FILTER(Survey, Survey[{target_col}] >= MaxVal - 1)),
    COUNTROWS(Survey),
    0
) * 100

Segment Size % =
DIVIDE(
    COUNTROWS(FILTER(Survey, Survey[segment] = SELECTEDVALUE(Segments[segment]))),
    COUNTROWS(Survey),
    0
) * 100

Avg by Group =
CALCULATE(
    AVERAGE(Survey[{target_col}]),
    ALLEXCEPT(Survey, Survey[{group_col}])
)
"""
    (pbi_dir / "measures.dax").write_text(dax, encoding="utf-8")
    print("  Saved: measures.dax")

    # 4) queries.pq template
    sample_cols = ", ".join([f'{{"{c}", type text}}' for c in df.columns[:5]])
    pq = f"""// Antigravity - Power Query M Scripts
// Generated: {datetime.now().strftime('%Y-%m-%d')}
// In Power BI: Home > Transform Data > Advanced Editor > paste this script

let
    Source = Csv.Document(
        File.Contents("[REPLACE WITH FULL PATH TO data_model.csv]"),
        [Delimiter=",", Columns={len(df.columns)}, Encoding=65001, QuoteStyle=QuoteStyle.None]
    ),
    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    #"Changed Types" = Table.TransformColumnTypes(
        #"Promoted Headers",
        {{{sample_cols}}}
    )
in
    #"Changed Types"
"""
    (pbi_dir / "queries.pq").write_text(pq, encoding="utf-8")
    print("  Saved: queries.pq")

    # 5) README
    top_cols = "\n".join(f"- {c}" for c in df.columns[:30])
    readme = f"""# Power BI Package - Antigravity
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Files
- data_model.csv: Main survey data ({len(df)} rows, {len(df.columns)} columns)
- statistical_summary.csv: Module results summary
- measures.dax: DAX measures (paste into Power BI > New Measure)
- queries.pq: Power Query M (paste into Advanced Editor)

## Quick start
1. Open Power BI Desktop
2. Get Data > Text/CSV > select data_model.csv
3. Load > Transform Data > Advanced Editor > paste queries.pq
4. Close & Apply
5. New Measure > paste blocks from measures.dax

## Columns in data_model.csv
{top_cols}
{"..." if len(df.columns) > 30 else ""}
"""
    (pbi_dir / "README.md").write_text(readme, encoding="utf-8")
    try:
        print(f"  Power BI package complete: {pbi_dir}")
    except UnicodeEncodeError:
        print("  Power BI package complete: tmp/powerbi")


if __name__ == "__main__":
    main()
