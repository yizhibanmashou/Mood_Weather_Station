"""
Script 02: DeepSeek Emotion Labeling
Labels mini_dataset posts with 6-dimension emotion scores.
Modes: smoke (300), pilot (3000), full (all)
Supports checkpoint resume, dry-run cost estimation.
"""
import pandas as pd
import numpy as np
import json
import os
import re
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
TMP_DIR = ROOT / "tmp"

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip() or "https://api.deepseek.com/v1"
TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "2000"))
BATCH_SIZE = int(os.getenv("DEEPSEEK_BATCH_SIZE", "30"))
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """你是中文社交媒体情绪分析专家。为每条微博输出6维情绪分数(0-1):
喜悦(joy)、悲伤(sadness)、愤怒(anger)、恐惧(fear)、惊讶(surprise)、中性(neutral)
6维之和=1.0，输出严格JSON数组。

标注要点:
- 反讽/阴阳怪气可能隐含愤怒
- emoji可作情绪信号 (😭=悲伤 😡=愤怒 😊=喜悦)
- "封城"≠恐惧,看上下文("封城也要加油"=正面)
- 纯事实报道=中性

Few-shot示例:
[1] "太好了武汉解封了热干面我来了！！😭"
→ {"joy":0.75,"sadness":0.05,"anger":0.0,"fear":0.0,"surprise":0.15,"neutral":0.05}
[2] "不是凭什么啊封了两个月现在跟我说数据有问题"
→ {"joy":0.0,"sadness":0.3,"anger":0.65,"fear":0.0,"surprise":0.05,"neutral":0.0}
[3] "今天新增确诊673例"
→ {"joy":0.0,"sadness":0.0,"anger":0.0,"fear":0.0,"surprise":0.0,"neutral":1.0}
[4] "害怕是真的害怕但相信国家一定能控制住"
→ {"joy":0.35,"sadness":0.0,"anger":0.0,"fear":0.6,"surprise":0.0,"neutral":0.05}

返回格式:
[{"id":1,"joy":...,"sadness":...,"anger":...,"fear":...,"surprise":...,"neutral":...}, ...]"""

EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
SOURCE_REQUIRED_COLS = ["post_id", "date_week", "date_month", "province", "content_clean", "word_count"]
OUTPUT_REQUIRED_COLS = SOURCE_REQUIRED_COLS + EMOTION_KEYS
LOG_PATH = TMP_DIR / "02_labeling_log.json"
FAILED_PATH = TMP_DIR / "02_labeling_failed.csv"
client = None


def get_client():
    """Create the API client only when a real API call is needed."""
    global client
    if client is not None:
        return client
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Create .env with DEEPSEEK_API_KEY=sk-... "
            "or set the environment variable directly."
        )
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL
    )
    return client


def parse_batch_response(raw, expected_count):
    """Parse DeepSeek JSON response. Returns list of dicts or None."""
    if not raw:
        return None
    # Strip markdown code fences
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    # Find JSON array
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        if not isinstance(data, list):
            return None
        if len(data) != expected_count:
            return None
        return data
    except json.JSONDecodeError:
        return None


def validate_scores(item):
    """Check 6 dims in [0,1] and sum in 1.0 +/- 0.03"""
    for k in EMOTION_KEYS:
        v = item.get(k)
        if v is None or not (0 <= v <= 1):
            return False, f"score_out_of_range:{k}={v}"
    total = sum(item.get(k, 0) for k in EMOTION_KEYS)
    if abs(total - 1.0) > 0.03:
        return False, f"sum_error:{total:.3f}"
    return True, "ok"


def label_batch(texts, batch_idx):
    """Label a batch of up to BATCH_SIZE texts. Returns (results, usage_info)."""
    numbered = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(texts)])
    response = get_client().chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": numbered}
        ]
    )
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    raw = response.choices[0].message.content
    parsed = parse_batch_response(raw, len(texts))
    return parsed, usage, raw


def run_dry_run(df):
    """Estimate cost with first 2 batches (60 items)"""
    print("\n[Dry-run] Estimating token usage with 60 samples...")
    sample_texts = df["content_clean"].head(60).tolist()
    total_prompt = 0
    total_completion = 0

    for i in range(0, len(sample_texts), BATCH_SIZE):
        batch = sample_texts[i:i + BATCH_SIZE]
        parsed, usage, _ = label_batch(batch, i // BATCH_SIZE)
        total_prompt += usage["prompt_tokens"]
        total_completion += usage["completion_tokens"]
        time.sleep(0.3)

    avg_prompt_per_item = total_prompt / len(sample_texts)
    avg_completion_per_item = total_completion / len(sample_texts)
    print(f"  Avg prompt tokens/item:     {avg_prompt_per_item:.0f}")
    print(f"  Avg completion tokens/item: {avg_completion_per_item:.0f}")
    print(f"  Note: check DeepSeek pricing page for current rates")

    # Estimate for full dataset
    n_total = len(df)
    est_prompt = int(avg_prompt_per_item * n_total)
    est_completion = int(avg_completion_per_item * n_total)
    print(f"  Estimated full ({n_total:,} items):")
    print(f"    Prompt tokens:     {est_prompt:,}")
    print(f"    Completion tokens: {est_completion:,}")

    est = {
        "avg_prompt_per_item": round(avg_prompt_per_item, 1),
        "avg_completion_per_item": round(avg_completion_per_item, 1),
        "est_total_prompt": est_prompt,
        "est_total_completion": est_completion,
        "total_items": n_total,
        "model": MODEL,
        "base_url": BASE_URL,
        "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE,
        "prompt_version": PROMPT_VERSION,
    }
    json.dump(est, open(TMP_DIR / "02_cost_estimate.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"  [OK] Cost estimate saved to tmp/02_cost_estimate.json")
    return est


def select_rows_for_mode(df, mode):
    """Stable shuffled prefix so smoke is a representative subset of pilot."""
    if mode == "smoke":
        n = min(300, len(df))
    elif mode == "pilot":
        n = min(3000, len(df))
    else:
        return df.copy()
    return df.sample(frac=1, random_state=42).head(n).copy()


def label_dataset(df, mode, output_path):
    """Label the dataset with checkpoint resume."""
    df_to_label = select_rows_for_mode(df, mode)
    n = len(df_to_label)
    df_to_label["post_id"] = df_to_label["post_id"].astype(str)
    texts = df_to_label["content_clean"].tolist()
    post_ids = df_to_label["post_id"].tolist()
    row_lookup = {str(row["post_id"]): row.to_dict() for _, row in df_to_label.iterrows()}

    # Check checkpoint
    existing = set()
    labeled_path = output_path
    tmp_path = TMP_DIR / f"02_labeling_{mode}.csv.tmp"
    tmp_existing_df = pd.DataFrame()
    if tmp_path.exists() and tmp_path.stat().st_size > 0:
        tmp_existing_df = pd.read_csv(tmp_path, encoding="utf-8-sig")
        if "post_id" in tmp_existing_df.columns:
            tmp_existing_df["post_id"] = tmp_existing_df["post_id"].astype(str)
            existing.update(tmp_existing_df["post_id"].tolist())
            print(f"  [Resume] {len(tmp_existing_df)} rows already present in tmp checkpoint")

    if labeled_path.exists():
        existing_df = pd.read_csv(labeled_path, encoding="utf-8-sig")
        existing_df["post_id"] = existing_df["post_id"].astype(str)
        if all(c in existing_df.columns for c in OUTPUT_REQUIRED_COLS):
            existing = set(existing_df["post_id"].tolist())
            if not tmp_existing_df.empty and "post_id" in tmp_existing_df.columns:
                existing.update(tmp_existing_df["post_id"].tolist())
            print(f"  [Resume] {len(existing)} already labeled, skipping...")
        else:
            backup = TMP_DIR / f"02_labeled_dataset_legacy_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            shutil.copy2(labeled_path, backup)
            print(f"  [WARN] Existing labeled_dataset.csv lacks source columns; backup saved to {backup}")
            existing = set()
            if not tmp_existing_df.empty and "post_id" in tmp_existing_df.columns:
                existing.update(tmp_existing_df["post_id"].tolist())

    total_batches = (len(post_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    log_entries = []
    failed_entries = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    new_count = 0

    for batch_idx in range(0, len(post_ids), BATCH_SIZE):
        batch_num = batch_idx // BATCH_SIZE
        batch_ids = post_ids[batch_idx:batch_idx + BATCH_SIZE]
        batch_texts = texts[batch_idx:batch_idx + BATCH_SIZE]

        # Skip if all already done
        if all(pid in existing for pid in batch_ids):
            continue

        # Need to label those not yet done
        to_label_ids = []
        to_label_texts = []
        for pid, txt in zip(batch_ids, batch_texts):
            if pid not in existing:
                to_label_ids.append(pid)
                to_label_texts.append(txt)

        if not to_label_texts:
            continue

        # Label
        success = False
        for attempt in range(3):
            try:
                parsed, usage, raw = label_batch(to_label_texts, batch_num)
                if parsed is None:
                    log_entries.append({
                        "batch": batch_num, "error": "parse_failed",
                        "raw": raw[:200] if raw else ""
                    })
                    time.sleep(1)
                    continue
                # Validate each
                all_ok = True
                for item in parsed:
                    ok, reason = validate_scores(item)
                    if not ok:
                        all_ok = False
                        log_entries.append({
                            "batch": batch_num, "id": item.get("id"),
                            "error": reason
                        })
                if all_ok:
                    success = True
                    break
                time.sleep(0.5)
            except Exception as e:
                log_entries.append({"batch": batch_num, "error": str(e)[:200]})
                time.sleep(2)

        if not success:
            for pid in to_label_ids:
                row = row_lookup.get(str(pid), {})
                failed_entries.append({
                    "post_id": pid,
                    "content": str(row.get("content_clean", ""))[:200]
                })
            continue

        # Map parsed results back to post_ids
        for i, item in enumerate(parsed):
            pid = str(to_label_ids[i])
            mode_label = row_lookup[pid].copy()
            for k in EMOTION_KEYS:
                mode_label[k] = item.get(k, 0)
            mode_label["label_status"] = "ok"
            mode_label["label_model"] = MODEL
            mode_label["prompt_version"] = PROMPT_VERSION

            # Append to temp file (create header if first write)
            write_header = not tmp_path.exists() or tmp_path.stat().st_size == 0
            if write_header:
                pd.DataFrame([mode_label]).to_csv(tmp_path, index=False, encoding="utf-8-sig", mode='w')
            else:
                pd.DataFrame([mode_label]).to_csv(tmp_path, index=False, encoding="utf-8-sig", mode='a', header=False)
            new_count += 1
            existing.add(pid)

        total_usage["prompt_tokens"] += usage["prompt_tokens"]
        total_usage["completion_tokens"] += usage["completion_tokens"]
        total_usage["total_tokens"] += usage["total_tokens"]

        progress = len(existing) / len(post_ids) * 100
        print(f"  [{progress:.0f}%] Batch {batch_num}/{total_batches} "
              f"labeled={len(existing)} tokens={total_usage['total_tokens']}")

        time.sleep(0.3)  # rate limiting

    # Merge checkpoint + temp into final output. If the final output already
    # covers the selected mode, leave a locked temp checkpoint alone.
    final_output_complete = False
    if labeled_path.exists() and new_count == 0:
        try:
            final_ids = pd.read_csv(labeled_path, usecols=["post_id"], encoding="utf-8-sig")
            final_ids["post_id"] = final_ids["post_id"].astype(str)
            final_output_complete = set(post_ids).issubset(set(final_ids["post_id"].tolist()))
        except Exception:
            final_output_complete = False

    if final_output_complete:
        print("  [Resume] Final output already complete; checkpoint cleanup skipped.")
    elif tmp_path.exists() and tmp_path.stat().st_size > 0:
        new_df = pd.read_csv(tmp_path, encoding="utf-8-sig")
        if labeled_path.exists():
            old_df = pd.read_csv(labeled_path, encoding="utf-8-sig")
            if all(c in old_df.columns for c in OUTPUT_REQUIRED_COLS):
                old_df["post_id"] = old_df["post_id"].astype(str)
                merged = pd.concat([old_df, new_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=["post_id"], keep="last")
            else:
                merged = new_df
        else:
            merged = new_df
        merged.to_csv(labeled_path, index=False, encoding="utf-8-sig")
        try:
            tmp_path.unlink()  # clean up temp
        except OSError as exc:
            print(f"  [WARN] Could not remove temp checkpoint {tmp_path}: {exc}")

    # Save logs
    log_path = LOG_PATH
    json.dump({
        "mode": mode,
        "total_labeled": len(existing),
        "total_batches": total_batches,
        "usage": total_usage,
        "errors": log_entries,
        "model": MODEL,
        "base_url": BASE_URL,
        "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE,
        "prompt_version": PROMPT_VERSION,
        "timestamp": datetime.now().isoformat(),
    }, open(log_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    if failed_entries:
        failed_path = FAILED_PATH
        pd.DataFrame(failed_entries).to_csv(failed_path, index=False, encoding="utf-8-sig")
        print(f"  [WARN] {len(failed_entries)} failed entries saved to {failed_path}")

    print(f"\n[OK] Labeling complete: {len(existing)}/{n} items")
    return total_usage


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke",
                        help="Labeling mode: smoke (300), pilot (3000), full (all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run cost estimation only")
    parser.add_argument("--input", type=str, default=None,
                        help="Custom input CSV path (overrides default mini_dataset.csv). "
                             "When set, --mode is ignored; all rows in the input are labeled.")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output CSV path (overrides default labeled_dataset.csv)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of rows to label (safety cap). "
                             "When set, only the first N rows will be labeled.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for API calls (overrides env DEEPSEEK_BATCH_SIZE)")
    parser.add_argument("--log", type=str, default=None,
                        help="Path for labeling log JSON (default: tmp/02_labeling_log.json)")
    parser.add_argument("--failed", type=str, default=None,
                        help="Path for failed entries CSV (default: tmp/02_labeling_failed.csv)")
    args = parser.parse_args()

    global BATCH_SIZE
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.log:
        global LOG_PATH
        LOG_PATH = Path(args.log)
    if args.failed:
        global FAILED_PATH
        FAILED_PATH = Path(args.failed)

    print("=" * 60)
    print(f"Script 02: DeepSeek Emotion Labeling (mode={args.mode})")
    print(f"Model: {MODEL}, Base URL: {BASE_URL}, Temperature: {TEMPERATURE}, Batch: {BATCH_SIZE}")
    print("=" * 60)

    # Determine input path
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = ROOT / input_path
    else:
        input_path = PROCESSED_DIR / "mini_dataset.csv"

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        sys.exit(1)
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"Loaded {len(df):,} posts from {input_path}")

    # Apply --limit
    if args.limit is not None and args.limit > 0:
        if len(df) > args.limit:
            print(f"[INFO] --limit {args.limit} specified, truncating from {len(df):,} to {args.limit:,} rows")
            df = df.head(args.limit).copy()

    # Safety check: if filename contains test/test500 and rows > 500, abort
    input_name = str(input_path).lower()
    if ("test" in input_name or "test500" in input_name) and len(df) > 500:
        print(f"[SAFETY] Input file looks like a test file but has {len(df):,} rows (>500).")
        print(f"  This would call the API {len(df):,} times. Aborting to prevent accidental full run.")
        sys.exit(1)
    missing_cols = [c for c in SOURCE_REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Input CSV missing required columns: {missing_cols}")
        sys.exit(1)

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Dry run
    if args.dry_run:
        try:
            run_dry_run(df)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        return

    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
    else:
        output_path = PROCESSED_DIR / "labeled_dataset.csv"

    # When using custom --input, label all rows (ignore --mode)
    if args.input:
        effective_mode = "full"
        print(f"\n[INFO] Custom input specified; labeling all {len(df):,} rows.")
    else:
        effective_mode = args.mode

    # Count existing output rows for resume info
    existing_output_count = 0
    if output_path.exists():
        try:
            existing_output_df = pd.read_csv(output_path, usecols=["post_id"], encoding="utf-8-sig")
            existing_output_count = len(existing_output_df)
        except Exception:
            existing_output_count = 0

    planned_rows = len(df)
    remaining_rows = max(0, planned_rows - existing_output_count)

    # Startup info
    print(f"\n{'=' * 60}")
    print(f"STARTUP INFO")
    print(f"{'=' * 60}")
    print(f"  Input:           {input_path}")
    print(f"  Output:          {output_path}")
    print(f"  Planned rows:    {planned_rows:,}")
    print(f"  Existing output: {existing_output_count:,}")
    print(f"  Remaining rows:  {remaining_rows:,}")
    print(f"  Model:           {MODEL}")
    print(f"  Prompt version:  {PROMPT_VERSION}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Limit:           {args.limit}")
    print(f"{'=' * 60}")

    # Confirm mode
    if effective_mode == "full" and not args.input:
        print(f"\n[WARNING] Full mode will label ALL {len(df):,} posts.")
        print("This WILL cost money. Run --dry-run first to estimate.")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    # Run labeling
    try:
        label_dataset(df, effective_mode, output_path)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
