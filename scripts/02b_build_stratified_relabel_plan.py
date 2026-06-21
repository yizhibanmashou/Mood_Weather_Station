"""
Script 02b: Build Stratified Relabel Plan
Reads mini_dataset.csv and labeled_dataset.csv, computes per (date_month x province)
gaps, and outputs a plan CSV of posts to label + a summary JSON.

Supports dry-run only — no DeepSeek API calls.
"""
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
TMP_DIR = ROOT / "tmp"

from _config import VALID_PROVINCES

# Province values to exclude (noise, overseas, unknown)
EXCLUDE_PROVINCES = {"", "未知", "其他", "海外", "null", "none", "nan"}


def load_data():
    """Load mini and labeled datasets."""
    mini_path = PROCESSED_DIR / "mini_dataset.csv"
    labeled_path = PROCESSED_DIR / "labeled_dataset.csv"

    if not mini_path.exists():
        print(f"[ERROR] {mini_path} not found.")
        sys.exit(1)
    if not labeled_path.exists():
        print(f"[ERROR] {labeled_path} not found. Run Script 02 first.")
        sys.exit(1)

    mini = pd.read_csv(mini_path, encoding="utf-8-sig")
    labeled = pd.read_csv(labeled_path, encoding="utf-8-sig")
    return mini, labeled


def filter_valid_provinces(df):
    """Keep only rows with valid province values."""
    df = df.copy()
    df["province"] = df["province"].astype(str).str.strip()
    # Exclude noise values
    mask = ~df["province"].str.lower().isin(EXCLUDE_PROVINCES)
    # Keep only 34 standard provinces
    mask = mask & df["province"].isin(VALID_PROVINCES)
    return df[mask]


def build_plan(mini, labeled, cap, seed):
    """Build the stratified relabel plan."""
    # Identify already-labeled post_ids
    labeled_ids = set(labeled["post_id"].astype(str).unique())
    print(f"  Already labeled post_ids: {len(labeled_ids)}")

    # Filter mini to valid provinces
    mini_valid = filter_valid_provinces(mini)
    print(f"  Mini dataset (valid provinces): {len(mini_valid)} / {len(mini)}")

    # Split into labeled and unlabeled
    mini_valid["post_id"] = mini_valid["post_id"].astype(str)
    labeled_in_mini = mini_valid[mini_valid["post_id"].isin(labeled_ids)]
    unlabeled = mini_valid[~mini_valid["post_id"].isin(labeled_ids)].copy()
    print(f"  Labeled (in mini, valid province): {len(labeled_in_mini)}")
    print(f"  Unlabeled candidates: {len(unlabeled)}")

    # Count existing labeled per (date_month, province)
    existing_counts = (
        labeled_in_mini.groupby(["date_month", "province"])
        .size()
        .reset_index(name="existing_count")
    )

    # Build the full grid of date_month x province
    all_months = sorted(mini_valid["date_month"].unique())
    all_provinces = sorted(VALID_PROVINCES)

    # Compute gap for each cell
    plan_rows = []
    detail_rows = []
    rng = np.random.RandomState(seed)

    for month in all_months:
        for province in all_provinces:
            # Current labeled count
            row = existing_counts[
                (existing_counts["date_month"] == month)
                & (existing_counts["province"] == province)
            ]
            current = int(row["existing_count"].values[0]) if len(row) > 0 else 0
            gap = max(0, cap - current)

            # Available unlabeled candidates for this cell
            candidates = unlabeled[
                (unlabeled["date_month"] == month)
                & (unlabeled["province"] == province)
            ]
            n_candidates = len(candidates)
            n_to_sample = min(gap, n_candidates)
            shortage = gap - n_to_sample

            if n_to_sample > 0:
                sampled = candidates.sample(n=n_to_sample, random_state=rng)
                plan_rows.append(sampled)

            detail_rows.append({
                "date_month": str(month),
                "province": province,
                "existing_labeled": current,
                "cap": cap,
                "gap": gap,
                "available_candidates": n_candidates,
                "planned_new": n_to_sample,
                "shortage": shortage,
                "expected_total": current + n_to_sample,
                "reliable_after": (current + n_to_sample) >= 30,
            })

    # Combine plan rows
    if plan_rows:
        plan_df = pd.concat(plan_rows, ignore_index=True)
    else:
        plan_df = pd.DataFrame()

    detail_df = pd.DataFrame(detail_rows)
    return plan_df, detail_df, labeled_in_mini, unlabeled


def compute_summary(plan_df, detail_df, labeled_in_mini, unlabeled, cap, seed):
    """Build the summary JSON."""
    before_reliable = detail_df[detail_df["existing_labeled"] >= 30]
    after_reliable = detail_df[detail_df["reliable_after"] == True]

    # Coverage = cells with at least 1 post
    before_coverage = detail_df[detail_df["existing_labeled"] > 0]
    after_coverage = detail_df[detail_df["expected_total"] > 0]

    # Cells with shortage
    shortage_cells = detail_df[detail_df["shortage"] > 0]

    summary = {
        "cap": cap,
        "seed": seed,
        "existing_labeled_rows": len(labeled_in_mini),
        "candidate_unlabeled_rows": len(unlabeled),
        "planned_new_rows": len(plan_df),
        "expected_total_after_labeling": len(labeled_in_mini) + len(plan_df),
        "before_month_province_coverage": len(before_coverage),
        "after_month_province_coverage": len(after_coverage),
        "total_month_province_cells": len(detail_df),
        "before_reliable_cells_count": len(before_reliable),
        "after_reliable_cells_count": len(after_reliable),
        "before_reliable_pct": round(len(before_reliable) / len(detail_df) * 100, 1) if len(detail_df) > 0 else 0,
        "after_reliable_pct": round(len(after_reliable) / len(detail_df) * 100, 1) if len(detail_df) > 0 else 0,
        "shortage_cells_count": len(shortage_cells),
        "shortage_total_missing": int(shortage_cells["shortage"].sum()) if len(shortage_cells) > 0 else 0,
        "by_month_province_detail": detail_df.to_dict(orient="records"),
        "timestamp": datetime.now().isoformat(),
    }
    return summary


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse

    parser = argparse.ArgumentParser(description="Build stratified relabel plan (dry-run only)")
    parser.add_argument("--cap", type=int, default=30, help="Target labeled count per (month, province)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--summary", type=str, default=None, help="Output summary JSON path")
    args = parser.parse_args()

    if args.output is None:
        args.output = str(PROCESSED_DIR / f"stratified_label_plan_cap{args.cap}.csv")
    if args.summary is None:
        args.summary = str(TMP_DIR / f"02b_stratified_label_plan_cap{args.cap}_summary.json")

    print("=" * 60)
    print(f"Script 02b: Build Stratified Relabel Plan (cap={args.cap}, seed={args.seed})")
    print("DRY-RUN ONLY — no DeepSeek API calls")
    print("=" * 60)

    # Ensure output dirs
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/4] Loading data...")
    mini, labeled = load_data()

    # Build plan
    print("\n[2/4] Building stratified plan...")
    plan_df, detail_df, labeled_in_mini, unlabeled = build_plan(
        mini, labeled, args.cap, args.seed
    )

    # Compute summary
    print("\n[3/4] Computing summary...")
    summary = compute_summary(plan_df, detail_df, labeled_in_mini, unlabeled, args.cap, args.seed)

    # Save outputs
    print("\n[4/4] Saving outputs...")
    if len(plan_df) > 0:
        plan_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"  Plan CSV: {args.output} ({len(plan_df)} rows)")
    else:
        # Write empty CSV with header
        pd.DataFrame(columns=["post_id"]).to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"  Plan CSV: {args.output} (0 rows — no gaps to fill)")

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary JSON: {args.summary}")

    # Print key stats
    print(f"\n{'=' * 60}")
    print(f"RESULTS (cap={args.cap})")
    print(f"{'=' * 60}")
    print(f"  Existing labeled rows:           {summary['existing_labeled_rows']}")
    print(f"  Candidate unlabeled rows:        {summary['candidate_unlabeled_rows']}")
    print(f"  Planned new rows:                {summary['planned_new_rows']}")
    print(f"  Expected total after labeling:   {summary['expected_total_after_labeling']}")
    print(f"  Before reliable cells (>=30):    {summary['before_reliable_cells_count']} ({summary['before_reliable_pct']}%)")
    print(f"  After reliable cells (>=30):     {summary['after_reliable_cells_count']} ({summary['after_reliable_pct']}%)")
    print(f"  Cells with shortage:             {summary['shortage_cells_count']}")
    print(f"  Total missing due to shortage:   {summary['shortage_total_missing']}")
    print(f"\nDone. No API calls made.")


if __name__ == "__main__":
    main()
