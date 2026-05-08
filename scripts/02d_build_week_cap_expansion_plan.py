"""
Script 02d: Build Week-Province Cap Expansion Plan
Generates stratified sampling plan for date_week x province cells
to fill each cell up to a configurable cap.
Supports dry-run mode to compare multiple cap options.
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

STANDARD_PROVINCES = {
    "北京", "天津", "上海", "重庆",
    "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "海南",
    "四川", "贵州", "云南", "陕西", "甘肃", "青海", "台湾",
    "内蒙古", "广西", "西藏", "宁夏", "新疆",
    "香港", "澳门",
}

PLAN_COLS = [
    "post_id", "user_id", "created_at", "date_week", "date_month",
    "province", "city", "gender", "content_clean", "content_raw", "word_count"
]

STRATA_KEY = ["date_week", "province"]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Build week-province cap expansion plan")
    parser.add_argument("--mini", type=str, default=None,
                        help="Mini dataset CSV path")
    parser.add_argument("--labeled", type=str, default=None,
                        help="Existing labeled dataset CSV path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output expansion plan CSV path")
    parser.add_argument("--cap", type=int, default=35,
                        help="Target samples per week-province cell")
    parser.add_argument("--strata", type=str, default="week_province",
                        choices=["week_province"], help="Stratification key (fixed)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compare multiple caps and output comparison table")
    parser.add_argument("--standard-only", action="store_true", default=True,
                        help="Only include standard 34 provinces")
    return parser.parse_args()


def load_data(mini_path, labeled_path, standard_only=True):
    """Load and validate input datasets."""
    mini = pd.read_csv(mini_path, encoding="utf-8-sig")
    mini["post_id"] = mini["post_id"].astype(str)

    if standard_only:
        before = len(mini)
        mini = mini[mini["province"].isin(STANDARD_PROVINCES)].copy()
        print(f"  Standard provinces filter: {before} -> {len(mini)} rows")

    labeled = pd.read_csv(labeled_path, encoding="utf-8-sig")
    labeled["post_id"] = labeled["post_id"].astype(str)

    if standard_only:
        labeled = labeled[labeled["province"].isin(STANDARD_PROVINCES)].copy()

    return mini, labeled


def compute_cell_status(mini, labeled, cap):
    """Compute current labeled counts and candidate availability per cell."""
    labeled_ids = set(labeled["post_id"].unique())
    candidates = mini[~mini["post_id"].isin(labeled_ids)].copy()

    # Current labeled counts per cell
    current_counts = labeled.groupby(STRATA_KEY).size()

    # Candidate counts per cell
    cand_counts = candidates.groupby(STRATA_KEY).size()

    # Compute gaps
    results = []
    all_cells = mini.groupby(STRATA_KEY).size().index

    for key in all_cells:
        cur = current_counts.get(key, 0)
        avail = cand_counts.get(key, 0)
        gap = max(cap - cur, 0)
        actual = min(gap, avail)
        shortage = gap - actual
        results.append({
            "date_week": key[0],
            "province": key[1],
            "current_labeled": int(cur),
            "candidates_available": int(avail),
            "gap": int(gap),
            "to_sample": int(actual),
            "shortage": int(shortage),
        })

    cell_df = pd.DataFrame(results)
    total_new = cell_df["to_sample"].sum()
    total_short = cell_df["shortage"].sum()
    short_cells = (cell_df["shortage"] > 0).sum()
    return cell_df, candidates, total_new, total_short, short_cells


def sample_plan(cell_df, candidates, cap, seed=42):
    """Sample the required number of posts per cell."""
    rng = np.random.default_rng(seed)
    plan_rows = []
    cell_results = []
    total_planned = 0
    total_shortage = 0
    short_cells = 0

    for _, row in cell_df.iterrows():
        wk = row["date_week"]
        prov = row["province"]
        n_sample = row["to_sample"]

        if n_sample <= 0:
            cell_results.append({**row.to_dict(), "actual_sampled": 0})
            continue

        pool = candidates[
            (candidates["date_week"] == wk) &
            (candidates["province"] == prov)
        ]
        if len(pool) == 0:
            cell_results.append({**row.to_dict(), "actual_sampled": 0, "note": "no_candidates"})
            continue

        # Ensure reproducibility
        pool_sorted = pool.sort_values("post_id").reset_index(drop=True)
        sampled = pool_sorted.iloc[rng.permutation(len(pool_sorted))[:n_sample]]

        for _, s in sampled.iterrows():
            row_out = {c: s.get(c, "") for c in PLAN_COLS if c in s}
            # Fill missing columns with empty string
            for c in PLAN_COLS:
                if c not in row_out:
                    row_out[c] = ""
            plan_rows.append(row_out)

        shortage = max(0, n_sample - len(sampled))
        total_shortage += shortage
        if shortage > 0:
            short_cells += 1
        total_planned += len(sampled)
        cell_results.append({**row.to_dict(), "actual_sampled": len(sampled), "shortage": shortage})

    plan_df = pd.DataFrame(plan_rows)
    cell_result_df = pd.DataFrame(cell_results)
    return plan_df, cell_result_df, total_planned, total_shortage, short_cells


def compute_expansion_stats(mini, labeled, cap):
    """Quick dry-run computation for a single cap value."""
    cell_df, candidates, total_new, total_short, short_cells = compute_cell_status(mini, labeled, cap)

    # Compute what stats would look like after filling
    current_counts = labeled.groupby(STRATA_KEY).size()
    target_cell_stat = []
    for _, row in cell_df.iterrows():
        key = (row["date_week"], row["province"])
        cur = current_counts.get(key, 0)
        target = cur + row["to_sample"]
        target_cell_stat.append(target)

    arr = np.array(target_cell_stat)
    stats = {
        "cap": int(cap),
        "new_samples": int(total_new),
        "total_after_merge": int(len(labeled) + total_new),
        "shortage_cells": int(short_cells),
        "total_shortage": int(total_short),
        "cell_min": int(arr.min()) if len(arr) > 0 else 0,
        "cell_max": int(arr.max()) if len(arr) > 0 else 0,
        "cell_median": round(float(np.median(arr)), 1) if len(arr) > 0 else 0,
        "cells_gte_15": int((arr >= 15).sum()),
        "cells_gte_30": int((arr >= 30).sum()),
        "total_cells": int(len(arr)),
        "in_target_range_30k_40k": "YES" if 30000 <= len(labeled) + total_new <= 40000 else "NO",
        "within_40000": bool(len(labeled) + total_new <= 40000),
    }
    return stats


def format_stats_table(all_stats, current_total):
    """Format comparison table as markdown."""
    lines = []
    lines.append("## 省-周 Cap 扩充方案对比")
    lines.append("")
    lines.append(f"- 当前已标注: **{current_total:,}** 条")
    lines.append(f"- 标准省份: **34**, 周数: **58**, 理论格子: **1,972**")
    lines.append("")
    lines.append("| Cap | 新增样本 | 合并后总量 | 候选不足格子 | 格子Min | 格子Max | 格子中位数 | >=15格子 | >=30格子 | 30-40k目标 |")
    lines.append("|-----|----------|------------|--------------|---------|---------|------------|----------|----------|------------|")
    for s in all_stats:
        lines.append(
            f"| {s['cap']} | {s['new_samples']:,} | {s['total_after_merge']:,} "
            f"| {s['shortage_cells']} | {s['cell_min']} | {s['cell_max']} "
            f"| {s['cell_median']} | {s['cells_gte_15']}/{s['total_cells']} "
            f"| {s['cells_gte_30']}/{s['total_cells']} "
            f"| {s['in_target_range_30k_40k']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    # Resolve paths
    mini_path = Path(args.mini) if args.mini else PROCESSED_DIR / "mini_dataset.csv"
    labeled_path = Path(args.labeled) if args.labeled else PROCESSED_DIR / "labeled_dataset_merged_cap30.csv"
    output_path = Path(args.output) if args.output else PROCESSED_DIR / "week_cap60_expansion_plan.csv"

    # Convert relative to absolute if needed
    for p in [mini_path, labeled_path, output_path]:
        if not p.is_absolute():
            p = ROOT / p

    print("=" * 60)
    print("Script 02d: Build Week-Province Cap Expansion Plan")
    print("=" * 60)
    print(f"Mini dataset:     {mini_path}")
    print(f"Labeled dataset:  {labeled_path}")
    print(f"Output plan:      {output_path}")
    print(f"Cap:              {args.cap}")
    print(f"Seed:             {args.seed}")
    print(f"Dry run:          {args.dry_run}")
    print()

    # Load data
    print("[1/4] Loading datasets...")
    mini, labeled = load_data(mini_path, labeled_path, standard_only=args.standard_only)
    print(f"  Mini:       {len(mini):,} rows, {mini['province'].nunique()} provinces, {mini['date_week'].nunique()} weeks")
    print(f"  Labeled:    {len(labeled):,} rows, {labeled['province'].nunique()} provinces, {labeled['date_week'].nunique()} weeks")

    # Check for overlapping post_ids
    labeled_ids = set(labeled["post_id"].unique())
    mini_ids = set(mini["post_id"].unique())
    overlap = labeled_ids & mini_ids
    print(f"  Labeled in mini: {len(overlap):,} ({len(overlap)/len(labeled)*100:.1f}%)")

    if args.dry_run:
        print("\n[2/4] Dry-run: comparing cap options...")
        all_stats = []
        for cap in [25, 30, 35, 40, 45, 50, 60, 70]:
            stats = compute_expansion_stats(mini, labeled, cap)
            all_stats.append(stats)
            flag = " <<<" if 30000 <= stats["total_after_merge"] <= 40000 else ""
            print(f"  cap={cap:2d}: +{stats['new_samples']:5d} new = {stats['total_after_merge']:5d} total  "
                  f"short={stats['shortage_cells']:4d} cells  min={stats['cell_min']:2d}  "
                  f"gte15={stats['cells_gte_15']}/{stats['total_cells']}{flag}")

        # Markdown table
        md = format_stats_table(all_stats, len(labeled))
        md_path = TMP_DIR / "week_cap_expansion_options.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\n  Comparison table saved: {md_path}")

        # CSV table
        csv_path = TMP_DIR / "week_cap_expansion_options.csv"
        pd.DataFrame(all_stats).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  Comparison CSV saved:   {csv_path}")

        # Determine best cap in 30-40k range
        in_range = [s for s in all_stats if 30000 <= s["total_after_merge"] <= 40000]
        if in_range:
            best = max(in_range, key=lambda s: s["new_samples"])
            print(f"\n  [RECOMMENDATION] cap={best['cap']} (total={best['total_after_merge']:,}) fits 30-40k target.")
        else:
            closest = min(all_stats, key=lambda s: abs(s["total_after_merge"] - 35000))
            print(f"\n  [NOTE] No cap fits exactly in 30-40k range. Closest: cap={closest['cap']} (total={closest['total_after_merge']:,})")
        return

    # Full plan generation
    print(f"\n[2/4] Computing cell status for cap={args.cap}...")
    cell_df, candidates, total_new, total_short, short_cells = compute_cell_status(mini, labeled, args.cap)
    print(f"  Total new to sample: {total_new:,}")
    print(f"  Cells with shortage: {short_cells}")
    print(f"  Total shortage items: {total_short}")

    # Validate candidate availability
    if total_new == 0:
        print("\n[WARN] No new samples needed. All cells are at or above cap.")
        sys.exit(0)

    print(f"\n[3/4] Sampling {total_new:,} posts...")
    plan_df, cell_result_df, total_planned, total_shortage, short_cells = sample_plan(
        cell_df, candidates, args.cap, seed=args.seed
    )
    print(f"  Sampled: {total_planned:,} posts")
    print(f"  Cells with shortage after sampling: {short_cells}")
    print(f"  Total shortage: {total_shortage}")

    if len(plan_df) == 0:
        print("[ERROR] No samples could be selected. Aborting.")
        sys.exit(1)

    # Deduplication check
    n_dupes = plan_df["post_id"].duplicated().sum()
    if n_dupes > 0:
        print(f"  [WARN] {n_dupes} duplicate post_ids found (should be 0)! Deduplicating...")
        plan_df = plan_df.drop_duplicates(subset=["post_id"], keep="first")

    # Verify no overlap with labeled
    labeled_ids = set(labeled["post_id"].unique())
    plan_ids = set(plan_df["post_id"].unique())
    overlap_new = labeled_ids & plan_ids
    if len(overlap_new) > 0:
        print(f"  [ERROR] {len(overlap_new)} plan post_ids already labeled! Removing...")
        plan_df = plan_df[~plan_df["post_id"].isin(labeled_ids)]
        print(f"  After removal: {len(plan_df)} rows")

    # Save plan CSV
    print(f"\n[4/4] Saving output...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  Plan CSV: {output_path} ({len(plan_df):,} rows)")

    # Save summary
    summary = {
        "cap": args.cap,
        "total_mini_rows": int(len(mini)),
        "total_labeled_rows": int(len(labeled)),
        "total_plan_rows": int(len(plan_df)),
        "province_count": int(mini["province"].nunique()),
        "week_count": int(mini["date_week"].nunique()),
        "total_new_samples": int(len(plan_df)),
        "total_after_merge": int(len(labeled) + len(plan_df)),
        "cells_with_shortage": int(short_cells),
        "total_shortage_items": int(total_shortage),
        "duplicates_in_plan": int(n_dupes),
        "overlap_with_labeled": int(len(overlap_new)),
        "seed": args.seed,
        "standard_provinces_only": bool(args.standard_only),
        "timestamp": datetime.now().isoformat(),
    }

    # Summary MD
    md_lines = [
        "# 省-周 Cap 扩充计划摘要",
        "",
        f"- Cap: **{args.cap}** 条/省-周",
        f"- 候选池: mini_dataset.csv (**{len(mini):,}** 条)",
        f"- 现有标注: labeled_dataset_merged_cap30.csv (**{len(labeled):,}** 条)",
        f"- 计划新增: **{len(plan_df):,}** 条",
        f"- 合并后预计: **{len(labeled) + len(plan_df):,}** 条",
        f"- 候选不足格子: **{short_cells}** 个",
        f"- 总短缺量: **{total_shortage}** 条",
        f"- 省份数: **{mini['province'].nunique()}**",
        f"- 周数: **{mini['date_week'].nunique()}**",
        f"- 随机种子: **{args.seed}**",
        f"- 生成时间: **{summary['timestamp']}**",
        "",
        "## 各省样本量",
    ]
    prov_counts = plan_df["province"].value_counts()
    for prov, cnt in prov_counts.items():
        md_lines.append(f"- {prov}: {cnt}")

    md_path = TMP_DIR / "week_cap60_expansion_plan_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  Summary MD: {md_path}")

    summary_json_path = TMP_DIR / "week_cap60_expansion_plan_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary JSON: {summary_json_path}")

    # Cell-level result CSV
    cell_csv_path = TMP_DIR / "week_cap60_expansion_cell_results.csv"
    cell_result_df.to_csv(cell_csv_path, index=False, encoding="utf-8-sig")
    print(f"  Cell results: {cell_csv_path}")

    print(f"\n{'=' * 60}")
    print(f"PLAN SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Cap:               {args.cap}/省-周")
    print(f"  New samples:       {len(plan_df):,}")
    print(f"  After merge:       {len(labeled) + len(plan_df):,}")
    print(f"  Shortage cells:    {short_cells}")
    print(f"  Total shortage:    {total_shortage}")
    print(f"  Post-ID overlap:   {len(overlap_new)}")
    print(f"  Plan duplicates:   {n_dupes}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
