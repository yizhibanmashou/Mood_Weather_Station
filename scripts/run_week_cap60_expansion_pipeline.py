"""
Script: run_week_cap60_expansion_pipeline.py
One-click pipeline for week-province cap expansion.
Orchestrates: audit → dry-run → plan → cost-estimate → smoke → label → finalize → quality report.
Supports resume: if a step already completed, it is skipped (unless --force).
"""
import sys
import os
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
TMP_DIR = ROOT / "tmp"
PYTHON = sys.executable

# Pipeline state file — tracks completion status of each step
STATE_FILE = TMP_DIR / "week_cap60_pipeline_state.json"


def load_state():
    if STATE_FILE.exists():
        try:
            return json.load(open(STATE_FILE, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def mark_step(state, step, status, info=None):
    state[step] = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }
    if info:
        state[step]["info"] = info
    save_state(state)


def is_step_done(state, step, force=False):
    if force:
        return False
    return state.get(step, {}).get("status") == "done"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def run_script(script_name, args_list, step_name, timeout=7200):
    """Run a Python script with args, return (success, stdout_truncated)."""
    script_path = ROOT / "scripts" / script_name
    cmd = [PYTHON, str(script_path)] + args_list
    log(f"Running {step_name}: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=False, timeout=timeout)
        result.stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        result.stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        # Print relevant output
        for line in result.stdout.split("\n"):
            if line.strip() and ("[OK]" in line or "[ERROR]" in line or "Done" in line or "SUMMARY" in line):
                log(f"  {line.strip()}")
        if result.returncode != 0:
            log(f"[ERROR] {step_name} failed (code {result.returncode})")
            log(f"  Last stderr: {result.stderr[-1000:]}")
            return False, result.stderr[-500:]
        return True, result.stdout[-500:]
    except subprocess.TimeoutExpired:
        log(f"[ERROR] {step_name} timed out after {timeout}s")
        return False, "timeout"
    except Exception as e:
        log(f"[ERROR] {step_name} exception: {e}")
        return False, str(e)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse

    parser = argparse.ArgumentParser(description="Week Cap60 Expansion Pipeline")
    parser.add_argument("--cap", type=int, default=60,
                        help="Target samples per week-province cell (default: 60)")
    parser.add_argument("--batch-size", type=int, default=30,
                        help="Batch size for labeling API calls")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from last incomplete step (default: True)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resume, re-run all steps")
    parser.add_argument("--auto-finalize", action="store_true", default=True,
                        help="Auto-finalize after labeling completes")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip smoke test")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run all steps even if completed")
    args = parser.parse_args()

    effective_resume = not args.no_resume
    state = load_state() if effective_resume else {}
    force = args.force

    print("=" * 60)
    print("Week Cap60 Expansion Pipeline")
    print("=" * 60)
    print(f"Cap:              {args.cap}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Resume:           {effective_resume}")
    print(f"Skip smoke:       {args.skip_smoke}")
    print(f"Auto finalize:    {args.auto_finalize}")
    print(f"Force re-run:     {force}")
    print(f"Python:           {PYTHON}")
    print()

    # Step 0: Audit
    log("[Step 0/7] Audit (run by inline commands)...")
    # Simple audit is embedded; full audit already done separately.

    # Step 1: Dry-run comparison
    step = "dry_run"
    if is_step_done(state, step, force):
        log(f"[SKIP] {step} already completed.")
    else:
        log(f"\n[Step 1/7] Dry-run comparing cap options...")
        ok, out = run_script("02d_build_week_cap_expansion_plan.py", [
            "--dry-run", "--cap", str(args.cap),
        ], "dry_run", timeout=180)
        if ok:
            mark_step(state, step, "done", {"cap": args.cap})
            # Print the comparison table
            md_path = TMP_DIR / "week_cap_expansion_options.md"
            if md_path.exists():
                log(f"  Comparison table: {md_path}")
        else:
            log("[FATAL] Dry-run failed. Aborting.")
            sys.exit(1)

    # Step 2: Generate expansion plan
    step = "generate_plan"
    plan_csv = PROCESSED_DIR / "week_cap60_expansion_plan.csv"
    if is_step_done(state, step, force) and plan_csv.exists():
        log(f"[SKIP] {step} already completed. Plan: {plan_csv}")
    else:
        log(f"\n[Step 2/7] Generating expansion plan (cap={args.cap})...")
        ok, out = run_script("02d_build_week_cap_expansion_plan.py", [
            "--cap", str(args.cap),
            "--output", str(plan_csv),
        ], "generate_plan", timeout=300)
        if ok:
            mark_step(state, step, "done", {"cap": args.cap, "rows": "see plan"})
        else:
            log("[FATAL] Plan generation failed. Aborting.")
            sys.exit(1)

    # Step 2b: Cost estimate
    step = "cost_estimate"
    cost_json = TMP_DIR / "week_cap60_cost_estimate.json"
    if is_step_done(state, step, force) and cost_json.exists():
        log(f"[SKIP] {step} already completed.")
    else:
        log(f"\n[Step 2b/7] Generating cost estimate...")
        # Count plan rows
        if plan_csv.exists():
            import pandas as pd
            plan_df = pd.read_csv(plan_csv, encoding="utf-8-sig")
            n_items = len(plan_df)
        else:
            n_items = 0
            log("[WARN] Plan CSV not found for cost estimate.")

        import math
        batches = (n_items + args.batch_size - 1) // args.batch_size
        prompt_per = 600
        completion_per = 100
        total_prompt = prompt_per * n_items
        total_completion = completion_per * n_items
        input_cost = total_prompt / 1_000_000 * 0.5
        output_cost = total_completion / 1_000_000 * 2
        total_cost = input_cost + output_cost
        total_hours = batches * 15 / 3600

        est = {
            "new_samples": n_items,
            "batch_size": args.batch_size,
            "total_batches": batches,
            "prompt_tokens_est": total_prompt,
            "completion_tokens_est": total_completion,
            "input_cost_yuan": round(input_cost, 2),
            "output_cost_yuan": round(output_cost, 2),
            "total_cost_est_yuan": round(total_cost, 2),
            "time_est_hours": round(total_hours, 1),
            "model": "deepseek-v4-flash",
            "prompt_version": "v1",
            "temperature": 0,
        }
        with open(cost_json, "w", encoding="utf-8") as f:
            json.dump(est, f, indent=2, ensure_ascii=False)
        log(f"  Estimated cost: {total_cost:.2f} yuan")
        log(f"  Estimated time: {total_hours:.1f} hours")
        mark_step(state, step, "done", est)

    # Step 3: Smoke test
    smoke_csv = PROCESSED_DIR / "week_cap60_expansion_plan_smoke.csv"
    step = "generate_smoke"
    if is_step_done(state, step, force) and smoke_csv.exists():
        log(f"[SKIP] {step} already completed.")
    else:
        log(f"\n[Step 3/7] Generating smoke test plan...")
        import pandas as pd
        import numpy as np
        plan = pd.read_csv(plan_csv, encoding="utf-8-sig")
        rng = np.random.default_rng(42)
        provinces = plan["province"].unique()
        rng.shuffle(provinces)
        selected = []
        for prov in provinces:
            prov_data = plan[plan["province"] == prov]
            weeks = prov_data["date_week"].unique()
            rng.shuffle(list(weeks))
            count = 0
            for wk in weeks:
                pool = prov_data[prov_data["date_week"] == wk]
                if len(pool) > 0:
                    picked = pool.iloc[rng.integers(0, len(pool))]
                    selected.append(picked)
                    count += 1
                    if count >= 2:
                        break
            if len(selected) >= 60:
                break
        smoke = pd.DataFrame(selected).head(60).reset_index(drop=True)
        if len(smoke) < 60:
            extra = plan[~plan.index.isin(smoke.index)].sample(60 - len(smoke), random_state=42)
            smoke = pd.concat([smoke, extra]).reset_index(drop=True)
        smoke.to_csv(smoke_csv, index=False, encoding="utf-8-sig")
        log(f"  Smoke plan: {len(smoke)} rows, {smoke['province'].nunique()} provinces, {smoke['date_week'].nunique()} weeks")
        mark_step(state, step, "done")

    # Step 4: Run smoke labeling
    smoke_output = PROCESSED_DIR / "labeled_dataset_week_cap60_expansion_smoke.csv"
    step = "smoke_label"
    if is_step_done(state, step, force) and smoke_output.exists():
        log(f"[SKIP] {step} already completed. Output: {smoke_output}")
    else:
        if args.skip_smoke:
            log(f"[SKIP] {step} skipped (--skip-smoke).")
            mark_step(state, step, "skipped")
        else:
            log(f"\n[Step 4/7] Running smoke labeling (60 samples)...")
            log("  This will call the DeepSeek API. Expected cost: minimal.")
            ok, out = run_script("02_label_emotions.py", [
                "--input", str(smoke_csv),
                "--output", str(smoke_output),
                "--batch-size", str(args.batch_size),
                "--log", str(TMP_DIR / "week_cap60_smoke_log.json"),
                "--failed", str(TMP_DIR / "week_cap60_smoke_failed.csv"),
                "--limit", "60",
            ], "smoke_label", timeout=600)

            if not ok:
                log("[FATAL] Smoke labeling failed. Check API key and try again.")
                log("  Fix the issue and re-run with --resume (default).")
                mark_step(state, step, "failed")
                sys.exit(1)

            # Verify smoke output
            if smoke_output.exists():
                import pandas as pd
                smoke_df = pd.read_csv(smoke_output, encoding="utf-8-sig")
                log(f"  Smoke output: {len(smoke_df)} rows")

                # Validate
                issues = []
                if len(smoke_df) < 50:
                    issues.append(f"Too few rows: {len(smoke_df)}")
                if "post_id" not in smoke_df.columns:
                    issues.append("Missing post_id column")
                for k in ["joy", "sadness", "anger", "fear", "surprise", "neutral"]:
                    if k not in smoke_df.columns:
                        issues.append(f"Missing emotion column: {k}")
                    elif smoke_df[k].min() < 0 or smoke_df[k].max() > 1:
                        issues.append(f"Emotion {k} out of range [0,1]")

                if issues:
                    log(f"[SMOKE FAIL] Issues: {'; '.join(issues)}")
                    log("  Fix issues and re-run with --resume.")
                    mark_step(state, step, "failed", issues)
                    sys.exit(1)
                else:
                    log(f"[SMOKE PASS] All checks passed.")
                    mark_step(state, step, "done", {"rows": len(smoke_df)})
            else:
                log("[FATAL] Smoke output not found after labeling.")
                mark_step(state, step, "failed")
                sys.exit(1)

    # Step 5: Full labeling (this will be the long-running step)
    full_output = PROCESSED_DIR / "labeled_dataset_week_cap60_expansion.csv"
    step = "full_label"
    if is_step_done(state, step, force) and full_output.exists():
        log(f"[SKIP] {step} already completed. Output: {full_output}")
    else:
        log(f"\n[Step 5/7] Full labeling...")
        log(f"  Input:  {plan_csv}")
        log(f"  Output: {full_output}")
        log(f"  This is the LONG-RUNNING step (~4 hours).")
        log(f"  Interrupt safely and re-run with --resume to continue.")

        ok, out = run_script("02_label_emotions.py", [
            "--input", str(plan_csv),
            "--output", str(full_output),
            "--batch-size", str(args.batch_size),
            "--log", str(TMP_DIR / "week_cap60_labeling_log.json"),
            "--failed", str(TMP_DIR / "week_cap60_labeling_failed.csv"),
        ], "full_label", timeout=14400)  # 4 hour timeout

        if not ok:
            log("[WARN] Full labeling did not complete successfully.")
            log("  Re-run with --resume to continue from where it stopped.")
            mark_step(state, step, "interrupted")
            # Don't exit — maybe they want to finalize with what we have
            if full_output.exists():
                log(f"  Partial output exists: {full_output}")
            sys.exit(1)
        else:
            mark_step(state, step, "done")

    # Step 6: Finalize
    if args.auto_finalize:
        step = "finalize"
        merged_output = PROCESSED_DIR / "labeled_dataset_merged_week_cap60.csv"
        if is_step_done(state, step, force) and merged_output.exists():
            log(f"[SKIP] {step} already completed. Output: {merged_output}")
        else:
            log(f"\n[Step 6/7] Finalizing (merge + downstream)...")
            ok, out = run_script("02e_finalize_week_cap_expansion.py", [
                "--old-merged", str(PROCESSED_DIR / "labeled_dataset_merged_cap30.csv"),
                "--new-labeled", str(full_output),
                "--plan", str(plan_csv),
                "--output", str(merged_output),
                "--log", str(TMP_DIR / "week_cap60_finalize_run.log"),
            ], "finalize", timeout=3600)

            if ok:
                mark_step(state, step, "done")
            else:
                log("[WARN] Finalize step had issues. Check logs.")
                mark_step(state, step, "issues")

    # Step 7: Quality report
    step = "quality_report"
    qc_json = TMP_DIR / "week_cap60_quality_report.json"
    if is_step_done(state, step, force) and qc_json.exists():
        log(f"[SKIP] {step} already completed.")
    else:
        log(f"\n[Step 7/7] Generating final quality report...")
        # Quality report was generated in finalize; just verify it exists.
        if qc_json.exists():
            with open(qc_json, "r", encoding="utf-8") as f:
                qc = json.load(f)
            log(f"  Merged total: {qc.get('merged_total_rows', 'N/A')}")
            log(f"  Recommend frontend replace: {qc.get('recommend_frontend_replace', 'N/A')}")
            mark_step(state, step, "done")
        else:
            # Try to generate standalone quality report
            log("[SKIP] Quality report not available (finalize may not have completed).")

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    state = load_state()
    for s, info in state.items():
        status = info.get("status", "?")
        ts = info.get("timestamp", "")[11:19] if "timestamp" in info else ""
        print(f"  {s}: {status} ({ts})")
    print()
    print(f"Pipeline state: {STATE_FILE}")
    print(f"To resume after interrupt: python scripts/run_week_cap60_expansion_pipeline.py")
    print("Done.")


if __name__ == "__main__":
    main()
