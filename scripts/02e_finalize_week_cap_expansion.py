"""
Script 02e: Finalize Week Cap60 Expansion
After labeling completes:
1. Merge old merged + new labeled data
2. Quality check
3. Run 04-08 downstream pipeline
4. Generate quality report
5. Optionally copy to frontend default files
"""
import pandas as pd
import numpy as np
import json
import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
TMP_DIR = ROOT / "tmp"
ANALYSIS_DIR = ROOT / "analysis"

EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
VALID_PROVINCES = {
    "北京", "天津", "上海", "重庆",
    "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "广西", "海南",
    "四川", "贵州", "云南", "西藏",
    "陕西", "甘肃", "青海", "宁夏", "新疆",
    "内蒙古", "香港", "澳门", "台湾",
}

PYTHON = sys.executable


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def run_script(script_name, args_list, step_name):
    """Run a Python script with args, return success status."""
    script_path = ROOT / "scripts" / script_name
    cmd = [PYTHON, str(script_path)] + args_list
    log(f"Running {step_name}: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=False, timeout=7200)
        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        if result.returncode != 0:
            log(f"[ERROR] {step_name} failed with code {result.returncode}")
            log(f"stderr: {stderr[-2000:]}")
            return False
        # Print stdout
        for line in stdout.split("\n"):
            if line.strip():
                log(f"  {line.strip()}")
        return True
    except subprocess.TimeoutExpired:
        log(f"[ERROR] {step_name} timed out")
        return False
    except Exception as e:
        log(f"[ERROR] {step_name} exception: {e}")
        return False


def compute_emotion_sum_valid(df):
    if not all(k in df.columns for k in EMOTION_KEYS):
        return 0.0
    sums = df[EMOTION_KEYS].sum(axis=1)
    return round(float((abs(sums - 1.0) < 0.05).mean()), 4)


def quality_check(merged_df, old_df, new_df, plan_df):
    """Run quality checks and return report dict."""
    report = {
        "old_merged_rows": int(len(old_df)),
        "plan_rows": int(len(plan_df)) if plan_df is not None else 0,
        "new_labeled_rows": int(len(new_df)) if new_df is not None else 0,
        "merged_total_rows": int(len(merged_df)),
        "duplicate_post_ids_removed": 0,
        "invalid_labels_removed": 0,
        "timestamp": datetime.now().isoformat(),
    }

    # Duplicate check
    dup_count = merged_df["post_id"].duplicated().sum()
    report["duplicate_post_ids_remaining"] = int(dup_count)

    if old_df is not None and new_df is not None:
        old_ids = set(old_df["post_id"].astype(str))
        new_ids = set(new_df["post_id"].astype(str))
        report["duplicate_post_ids_removed"] = int(len(old_ids & new_ids))

    # Emotion validity
    report["emotion_sum_valid_rate"] = compute_emotion_sum_valid(merged_df)

    # Invalid labels (emotions out of range) — row-level check
    if all(k in merged_df.columns for k in EMOTION_KEYS):
        out_of_range = ((merged_df[EMOTION_KEYS] < 0) | (merged_df[EMOTION_KEYS] > 1)).any(axis=1)
        report["invalid_labels_removed"] = int(out_of_range.sum())
    else:
        report["invalid_labels_removed"] = 0

    # Province distribution
    if "province" in merged_df.columns:
        prov_counts = merged_df[merged_df["province"].isin(VALID_PROVINCES)]["province"].value_counts()
        report["province_sample_min"] = int(prov_counts.min()) if len(prov_counts) > 0 else 0
        report["province_sample_max"] = int(prov_counts.max()) if len(prov_counts) > 0 else 0
        report["province_sample_median"] = round(float(prov_counts.median()), 1) if len(prov_counts) > 0 else 0
        report["province_count"] = int(prov_counts.count())

    # Week distribution
    if "date_week" in merged_df.columns:
        week_counts = merged_df["date_week"].value_counts()
        report["week_count"] = int(week_counts.count())
        report["week_sample_min"] = int(week_counts.min()) if len(week_counts) > 0 else 0
        report["week_sample_max"] = int(week_counts.max()) if len(week_counts) > 0 else 0

    # Week-province grid
    if "province" in merged_df.columns and "date_week" in merged_df.columns:
        grid = merged_df[merged_df["province"].isin(VALID_PROVINCES)].groupby(
            ["date_week", "province"]
        ).size()
        report["grid_non_empty_cells"] = int(len(grid))
        report["grid_total_cells_theoretical"] = int(
            merged_df[merged_df["province"].isin(VALID_PROVINCES)]["date_week"].nunique()
            * merged_df[merged_df["province"].isin(VALID_PROVINCES)]["province"].nunique()
        )
        report["grid_cells_gte_15"] = int((grid >= 15).sum())
        report["grid_cells_gte_30"] = int((grid >= 30).sum())
        report["grid_cell_min"] = int(grid.min()) if len(grid) > 0 else 0
        report["grid_cell_max"] = int(grid.max()) if len(grid) > 0 else 0
        report["grid_cell_median"] = round(float(grid.median()), 1) if len(grid) > 0 else 0

    # Emotion distribution
    if all(k in merged_df.columns for k in EMOTION_KEYS):
        emotion_means = merged_df[EMOTION_KEYS].mean().to_dict()
        report["emotion_distribution"] = {k: round(float(v), 4) for k, v in emotion_means.items()}

    # Compare with cap30
    if old_df is not None and all(k in old_df.columns for k in EMOTION_KEYS):
        old_emotion_means = old_df[EMOTION_KEYS].mean().to_dict()
        report["old_emotion_distribution"] = {k: round(float(v), 4) for k, v in old_emotion_means.items()}
        # Drift
        drifts = {}
        for k in EMOTION_KEYS:
            cur = emotion_means.get(k, 0)
            old = old_emotion_means.get(k, 0)
            drifts[k] = round(float(cur - old), 4)
        report["emotion_drift_from_cap30"] = drifts

    # Recommend frontend replace?
    drift_vals = report.get("emotion_drift_from_cap30", {})
    max_drift = max(abs(v) for v in drift_vals.values()) if drift_vals else 0
    report["max_emotion_drift"] = round(float(max_drift), 4)

    issues = []
    if report.get("duplicate_post_ids_remaining", 0) > 0:
        issues.append(f"Remaining duplicate post_ids: {report['duplicate_post_ids_remaining']}")
    if report.get("invalid_labels_removed", 0) > 0:
        issues.append(f"Invalid labels removed: {report['invalid_labels_removed']}")
    if report.get("emotion_sum_valid_rate", 1) < 0.95:
        issues.append(f"Low emotion sum validity: {report['emotion_sum_valid_rate']}")
    if max_drift > 0.05:
        issues.append(f"Large emotion drift from cap30: {max_drift:.4f}")

    report["issues"] = issues
    report["recommend_frontend_replace"] = len(issues) == 0
    report["recommendation"] = (
        "OK - Replace frontend data" if len(issues) == 0
        else f"CAUTION - Issues found: {'; '.join(issues)}"
    )

    return report


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse

    parser = argparse.ArgumentParser(description="Finalize week cap expansion")
    parser.add_argument("--old-merged", type=str, default=None,
                        help="Old merged dataset (default: labeled_dataset_merged_cap30.csv)")
    parser.add_argument("--new-labeled", type=str, default=None,
                        help="Newly labeled dataset")
    parser.add_argument("--plan", type=str, default=None,
                        help="Expansion plan CSV (for stats)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output merged CSV path")
    parser.add_argument("--skip-downstream", action="store_true",
                        help="Skip running 04-08 downstream scripts")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if output exists")
    parser.add_argument("--frontend-copy", action="store_true",
                        help="Copy result to frontend default paths")
    parser.add_argument("--log", type=str, default=None,
                        help="Run log path")
    args = parser.parse_args()

    # Paths
    old_path = Path(args.old_merged) if args.old_merged else PROCESSED_DIR / "labeled_dataset_merged_cap30.csv"
    new_path = Path(args.new_labeled) if args.new_labeled else PROCESSED_DIR / "labeled_dataset_week_cap60_expansion.csv"
    plan_path = Path(args.plan) if args.plan else PROCESSED_DIR / "week_cap60_expansion_plan.csv"
    output_path = Path(args.output) if args.output else PROCESSED_DIR / "labeled_dataset_merged_week_cap60.csv"
    run_log_path = Path(args.log) if args.log else TMP_DIR / "week_cap60_finalize_run.log"

    for p in [old_path, new_path, plan_path, output_path]:
        if not p.is_absolute():
            p = ROOT / p

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Redirect log
    log_file = open(run_log_path, "w", encoding="utf-8")

    def tlog(msg):
        log(msg)
        log_file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        log_file.flush()

    tlog("=" * 60)
    tlog("Script 02e: Finalize Week Cap60 Expansion")
    tlog("=" * 60)

    # Check if output already exists
    if output_path.exists() and not args.force:
        tlog(f"[INFO] Output already exists: {output_path}")
        tlog("[INFO] Use --force to re-run. Skipping.")
        log_file.close()
        return

    # =========================================================
    # STEP 1: Load and merge datasets
    # =========================================================
    tlog("\n[Step 1/6] Loading and merging datasets...")

    if not old_path.exists():
        tlog(f"[ERROR] Old merged not found: {old_path}")
        log_file.close()
        sys.exit(1)
    old_df = pd.read_csv(old_path, encoding="utf-8-sig")
    old_df["post_id"] = old_df["post_id"].astype(str)
    tlog(f"  Old merged: {len(old_df):,} rows")

    if not new_path.exists():
        tlog(f"[ERROR] New labeled dataset not found: {new_path}")
        tlog("  Has the labeling run completed?")
        log_file.close()
        sys.exit(1)
    new_df = pd.read_csv(new_path, encoding="utf-8-sig")
    new_df["post_id"] = new_df["post_id"].astype(str)
    tlog(f"  New labeled: {len(new_df):,} rows")

    # Load plan for reference (optional)
    plan_df = None
    if plan_path.exists():
        plan_df = pd.read_csv(plan_path, encoding="utf-8-sig")
        plan_df["post_id"] = plan_df["post_id"].astype(str)
        tlog(f"  Plan: {len(plan_df):,} rows")

    # Duplicate check
    old_ids = set(old_df["post_id"])
    new_ids = set(new_df["post_id"])
    overlap = old_ids & new_ids
    tlog(f"  Overlapping post_ids: {len(overlap)}")

    # Merge: dedup by post_id, keep latest labeled if conflict
    combined = pd.concat([old_df, new_df], ignore_index=True)
    if "label_status" in combined.columns:
        combined["_priority"] = (combined["label_status"] != "ok").astype(int)
        combined = combined.sort_values(["post_id", "_priority"])
        combined = combined.drop_duplicates(subset=["post_id"], keep="first")
        combined = combined.drop(columns=["_priority"])
    else:
        combined = combined.drop_duplicates(subset=["post_id"], keep="first")

    combined = combined.sort_values("post_id").reset_index(drop=True)
    tlog(f"  Merged output: {len(combined):,} rows")

    # Validate emotion columns
    missing_emotions = [k for k in EMOTION_KEYS if k not in combined.columns]
    if missing_emotions:
        tlog(f"[ERROR] Missing emotion columns in merged data: {missing_emotions}")
        tlog("  Cannot proceed with downstream aggregation.")
        log_file.close()
        sys.exit(1)

    # Quality check
    tlog("\n[Step 2/6] Quality check...")
    qc_report = quality_check(combined, old_df, new_df, plan_df)
    tlog(f"  Emotion sum valid rate: {qc_report['emotion_sum_valid_rate']}")
    tlog(f"  Grid cells >=15: {qc_report.get('grid_cells_gte_15', 'N/A')}")
    tlog(f"  Max emotion drift: {qc_report.get('max_emotion_drift', 'N/A')}")
    tlog(f"  Recommend frontend replace: {qc_report['recommend_frontend_replace']}")
    if qc_report.get("issues"):
        for issue in qc_report["issues"]:
            tlog(f"  [ISSUE] {issue}")

    # Save merged output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    tlog(f"\n  Saved merged: {output_path}")

    # Save quality report
    qc_md_path = TMP_DIR / "week_cap60_quality_report.md"
    qc_json_path = TMP_DIR / "week_cap60_quality_report.json"

    with open(qc_json_path, "w", encoding="utf-8") as f:
        json.dump(qc_report, f, indent=2, ensure_ascii=False)

    qc_md_lines = [
        "# Quality Report - Week Cap60 Expansion",
        "",
        f"- Old merged rows: **{qc_report['old_merged_rows']:,}**",
        f"- Plan rows: **{qc_report['plan_rows']:,}**",
        f"- New labeled rows: **{qc_report['new_labeled_rows']:,}**",
        f"- Merged total: **{qc_report['merged_total_rows']:,}**",
        f"- Duplicates removed: **{qc_report['duplicate_post_ids_removed']}**",
        f"- Invalid labels removed: **{qc_report['invalid_labels_removed']}**",
        f"- Emotion sum valid rate: **{qc_report['emotion_sum_valid_rate']:.2%}**",
        "",
        "### Grid Coverage",
        f"- Non-empty cells: **{qc_report.get('grid_non_empty_cells', 'N/A')}**",
        f"- Cells >= 15: **{qc_report.get('grid_cells_gte_15', 'N/A')}**",
        f"- Cells >= 30: **{qc_report.get('grid_cells_gte_30', 'N/A')}**",
        f"- Cell min: **{qc_report.get('grid_cell_min', 'N/A')}**",
        f"- Cell median: **{qc_report.get('grid_cell_median', 'N/A')}**",
        "",
        "### Emotion Distribution",
    ]
    if "emotion_distribution" in qc_report:
        for k, v in qc_report["emotion_distribution"].items():
            drift = qc_report.get("emotion_drift_from_cap30", {}).get(k, 0)
            drift_str = f" ({drift:+.4f} vs cap30)" if drift else ""
            qc_md_lines.append(f"- {k}: **{v:.4f}**{drift_str}")

    qc_md_lines.append("")
    qc_md_lines.append("### Issues")
    if qc_report.get("issues"):
        for issue in qc_report["issues"]:
            qc_md_lines.append(f"- ⚠ {issue}")
    else:
        qc_md_lines.append("- No issues found.")

    qc_md_lines.append("")
    qc_md_lines.append(f"### Recommendation")
    qc_md_lines.append(f"**{qc_report['recommendation']}**")
    qc_md_lines.append(f"")
    qc_md_lines.append(f"- Generated: {qc_report['timestamp']}")

    with open(qc_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(qc_md_lines))
    tlog(f"  Quality report: {qc_md_path}")

    # Save merge report
    merge_report = {
        "old_rows": int(len(old_df)),
        "new_rows": int(len(new_df)),
        "merged_rows": int(len(combined)),
        "duplicate_post_ids_removed": int(len(overlap)),
        "ok_rate": round(float((combined["label_status"] == "ok").mean()), 4) if "label_status" in combined.columns else 1.0,
        "emotion_sum_valid_rate": qc_report["emotion_sum_valid_rate"],
        "timestamp": datetime.now().isoformat(),
    }
    merge_json_path = TMP_DIR / "week_cap60_merge_report.json"
    with open(merge_json_path, "w", encoding="utf-8") as f:
        json.dump(merge_report, f, indent=2, ensure_ascii=False)

    merge_md_lines = [
        "# Merge Report - Week Cap60 Expansion",
        "",
        f"- Old rows: **{merge_report['old_rows']:,}**",
        f"- New rows: **{merge_report['new_rows']:,}**",
        f"- Merged rows: **{merge_report['merged_rows']:,}**",
        f"- Duplicates removed: **{merge_report['duplicate_post_ids_removed']}**",
        f"- OK rate: **{merge_report['ok_rate']:.2%}**",
        f"- Emotion validity: **{merge_report['emotion_sum_valid_rate']:.2%}**",
        f"- Timestamp: **{merge_report['timestamp']}**",
    ]
    merge_md_path = TMP_DIR / "week_cap60_merge_report.md"
    with open(merge_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(merge_md_lines))

    # =========================================================
    # STEP 3: Run downstream scripts (04-08)
    # =========================================================
    if args.skip_downstream:
        tlog("\n[Step 3/6] Skipping downstream (--skip-downstream).")
    else:
        tlog("\n[Step 3/6] Running 04_aggregate_emotions.py...")
        ok = run_script("04_aggregate_emotions.py", [
            "--input", str(output_path),
            "--output-dir", str(PROCESSED_DIR),
            "--summary", str(TMP_DIR / "04_aggregation_summary_cap60.txt"),
        ], "04_aggregation")
        if not ok:
            tlog("[ERROR] 04 aggregation failed. Stopping downstream.")
            log_file.close()
            sys.exit(1)

        # Run 04b NLP keywords if it exists
        nlp_script = ROOT / "scripts" / "04b_nlp_keywords.py"
        if nlp_script.exists():
            tlog("\n[Step 4/6] Running 04b_nlp_keywords.py...")
            ok = run_script("04b_nlp_keywords.py", [
                "--input", str(output_path),
            ], "04b_nlp_keywords")
            if not ok:
                tlog("[WARN] 04b NLP keywords failed (non-fatal). Continuing.")
        else:
            tlog("\n[Step 4/6] NLP keywords script not found, skipping.")

        tlog("\n[Step 5/6] Running 05_detect_anomalies.py...")
        ok = run_script("05_detect_anomalies.py", [
            "--national", str(PROCESSED_DIR / "emotion_national_timeline.csv"),
            "--weekly", str(PROCESSED_DIR / "emotion_panel_weekly.csv"),
            "--output", str(PROCESSED_DIR / "anomaly_detection.json"),
        ], "05_anomalies")
        if not ok:
            tlog("[WARN] 05 anomaly detection failed (non-fatal). Continuing.")

        tlog("\n[Step 6/6] Running 06_cluster_provinces.py...")
        ok = run_script("06_cluster_provinces.py", [
            "--vectors", str(PROCESSED_DIR / "province_emotion_vectors.csv"),
            "--output-dir", str(PROCESSED_DIR),
            "--data-dir", str(PROCESSED_DIR),
        ], "06_clustering")
        if not ok:
            tlog("[WARN] 06 clustering failed (non-fatal). Continuing.")

        tlog("\n[Step 6b] Running 07_cluster_evolution.py...")
        ok = run_script("07_cluster_evolution.py", [
            "--monthly", str(PROCESSED_DIR / "emotion_panel_monthly.csv"),
            "--output-dir", str(ANALYSIS_DIR),
            "--tmp-dir", str(TMP_DIR),
        ], "07_evolution")
        if not ok:
            tlog("[WARN] 07 evolution failed (non-fatal). Continuing.")

        tlog("\n[Step 6c] Running 08_prepare_frontend_assets.py...")
        ok = run_script("08_prepare_frontend_assets.py", [
            "--data-dir", str(PROCESSED_DIR),
            "--analysis-dir", str(ANALYSIS_DIR),
            "--output-dir", str(ROOT / "app" / "public"),
        ], "08_frontend")
        if not ok:
            tlog("[WARN] 08 frontend assets failed (non-fatal). Continuing.")

    # =========================================================
    # STEP 4: Frontend copy (if requested and quality passes)
    # =========================================================
    if args.frontend_copy:
        tlog("\n[Extra] Frontend copy requested...")
        if qc_report["recommend_frontend_replace"]:
            frontend_dir = ROOT / "app" / "public" / "data"
            frontend_dir.mkdir(parents=True, exist_ok=True)
            frontend_default = frontend_dir / "labeled_dataset.csv"
            shutil.copy2(output_path, frontend_default)
            tlog(f"  Copied merged output to frontend default: {frontend_default}")

            # Update manifest if it exists
            manifest_path = frontend_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
                except Exception:
                    manifest = {}
                manifest["week_cap60_expansion"] = {
                    "merged_file": str(output_path.name),
                    "total_rows": int(len(combined)),
                    "quality_status": qc_report["recommendation"],
                    "timestamp": datetime.now().isoformat(),
                }
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)
                tlog(f"  Updated manifest: {manifest_path}")
        else:
            tlog("  [SKIP] Quality check not passed. Frontend copy skipped.")
            tlog(f"  Reason: {'; '.join(qc_report.get('issues', ['Unknown']))}")

    # Final summary
    tlog("\n" + "=" * 60)
    tlog("FINALIZE COMPLETE")
    tlog("=" * 60)
    tlog(f"  Old merged:  {merge_report['old_rows']:,}")
    tlog(f"  New labeled: {merge_report['new_rows']:,}")
    tlog(f"  Merged:      {merge_report['merged_rows']:,}")
    tlog(f"  Quality:     {qc_report['recommendation']}")
    tlog(f"  Log:         {run_log_path}")
    tlog(f"  Quality:     {qc_md_path}")
    tlog(f"  Merge:       {merge_md_path}")

    log_file.close()
    print(f"\n[OK] Finalize complete. Log: {run_log_path}")


if __name__ == "__main__":
    main()
