"""
Script 01: Build Mini Dataset
Reservoir sampling, text cleaning, province joining, quality filtering.
Outputs: data/processed/mini_dataset.csv + data/indexes/users.db
"""
import pandas as pd
import numpy as np
import csv
import sqlite3
import re
import sys
import random
import os
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "COV-Weibo2.0"
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "indexes"
TMP_DIR = ROOT / "tmp"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Sampling targets per month
SAMPLE_TARGETS = {
    "2020-01": 10000, "2020-02": 10000, "2020-03": 10000, "2020-04": 10000,
    "2020-05": 7000, "2020-06": 7000, "2020-07": 7000, "2020-08": 7000,
    "2020-09": 7000, "2020-10": 7000, "2020-11": 7000, "2020-12": 7000,
    "2019-12": 2000,
}

BATCH_SIZE = 100_000  # batch insert size for SQLite

# --- Text cleaning patterns ---
URL_PATTERN = re.compile(r'http\S+')
REPLY_PATTERN = re.compile(r'回复@\S+:')
FORWARD_PATTERN = re.compile(r'//@\S+:')
BRACKET_EMOJI = re.compile(r'\[[^\]]+\]')
HASHTAG_PATTERN = re.compile(r'#([^#]+)#')
ZH_CHAR = re.compile(r'[一-鿿]')

from _config import VALID_PROVINCES


def clean_text(text):
    if not text or text == "nan":
        return ""
    text = URL_PATTERN.sub('', str(text))
    text = REPLY_PATTERN.sub('', text)
    text = FORWARD_PATTERN.sub('', text)
    text = BRACKET_EMOJI.sub('', text)
    text = HASHTAG_PATTERN.sub(r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def count_zh(text):
    return len(ZH_CHAR.findall(text))


def stable_month_seed(month):
    """Deterministic seed; Python's built-in hash() is randomized per process."""
    return RANDOM_SEED + sum((idx + 1) * ord(ch) for idx, ch in enumerate(month))


def should_keep_unknown_for_national():
    """Keep unknown-province posts when the feasibility gate downgraded V1 to national only."""
    report_path = TMP_DIR / "00_feasibility_report.json"
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return report.get("recommended_geo_granularity") == "national_timeline_only"


def build_user_index():
    """Build SQLite user index with valid provinces only"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    db_path = INDEX_DIR / "users.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP TABLE IF EXISTS users")
    conn.execute(
        """CREATE TABLE users (
            user_id TEXT PRIMARY KEY,
            province TEXT,
            city TEXT,
            gender TEXT
        )"""
    )

    user_file = DATA_DIR / "user.csv"
    total = 0
    kept = 0
    batch = []
    with open(user_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            province = str(row.get("province", "")).strip()
            if province not in VALID_PROVINCES:
                continue
            kept += 1
            batch.append((
                str(row.get("user_id", "")).strip(),
                province,
                str(row.get("city", "")).strip(),
                str(row.get("gender", "")).strip(),
            ))
            if len(batch) >= BATCH_SIZE:
                conn.executemany("INSERT OR IGNORE INTO users VALUES (?,?,?,?)", batch)
                batch = []
        if batch:
            conn.executemany("INSERT OR IGNORE INTO users VALUES (?,?,?,?)", batch)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_id ON users(user_id)")
    conn.commit()
    print(f"  User index: {kept:,} valid / {total:,} total")
    return conn


def reservoir_sample(filepath, k, random_seed):
    """Reservoir sampling: read file once, keep exactly k random rows"""
    rng = random.Random(random_seed)
    reservoir = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < k:
                reservoir.append(row)
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = row
    return reservoir


def parse_date(raw):
    """Parse created_at, return (datetime, date_week, date_month) or None"""
    if not raw or raw == "nan":
        return None
    try:
        dt = pd.to_datetime(str(raw).strip())
        return (dt, dt.strftime("%G-W%V"), dt.strftime("%Y-%m"))
    except Exception:
        return None


def process_month(month, filepath, target, conn, keep_unknown_province=False):
    """Sample and process one month"""
    print(f"\n  [{month}] Reservoir sampling target={target}...")
    raw_rows = reservoir_sample(filepath, target, stable_month_seed(month))

    log_entries = []
    results = []
    join_hit = 0
    join_miss = 0

    for row in raw_rows:
        uid = str(row.get("user_id", "")).strip()

        # Parse date
        date_info = parse_date(row.get("created_at", ""))
        if date_info is None:
            log_entries.append(f"bad_date,{uid}")
            continue
        dt, date_week, date_month = date_info

        # Clean text
        content_raw = str(row.get("content", "")).strip()
        content_clean = clean_text(content_raw)

        # Quality filter: Chinese chars >= 10
        zh_count = count_zh(content_clean)
        if zh_count < 10:
            log_entries.append(f"too_short,{uid},{zh_count}")
            continue

        if not content_clean:
            log_entries.append(f"empty_content,{uid}")
            continue

        # Join province
        province = "未知"
        city = ""
        gender = ""
        if uid and uid != "nan":
            cur = conn.execute(
                "SELECT province, city, gender FROM users WHERE user_id = ?", (uid,)
            )
            res = cur.fetchone()
            if res:
                province = res[0]
                city = res[1] or ""
                gender = res[2] or ""
                join_hit += 1
            else:
                join_miss += 1

        # For province-level panels, unknown is unusable; for national-only V1, keep it.
        if province == "未知" and not keep_unknown_province:
            log_entries.append(f"unknown_province,{uid}")
            continue

        results.append({
            "post_id": str(row.get("_id", "")).strip(),
            "user_id": uid,
            "created_at": str(row.get("created_at", "")).strip(),
            "date_week": date_week,
            "date_month": date_month,
            "province": province,
            "city": city,
            "gender": gender,
            "content_clean": content_clean,
            "content_raw": content_raw[:500],  # keep raw but truncate for file size
            "word_count": zh_count,
        })

    print(f"    sampled={len(raw_rows)} kept={len(results)} join_hit={join_hit} join_miss={join_miss}")
    return results, log_entries


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("Script 01: Build Mini Dataset")
    print("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build user index
    print("\n[1/3] Building user index...")
    conn = build_user_index()

    # 2. Process each month
    print("\n[2/3] Sampling and processing months...")
    keep_unknown_province = should_keep_unknown_for_national()
    if keep_unknown_province:
        print("  Feasibility gate recommends national timeline only; keeping unknown-province posts.")
    all_rows = []
    all_logs = []
    month_counts = {}

    for month in sorted(SAMPLE_TARGETS.keys()):
        filepath = DATA_DIR / f"{month}.csv"
        if not filepath.exists():
            print(f"  [{month}] SKIP: file not found")
            continue
        target = SAMPLE_TARGETS[month]
        results, logs = process_month(month, filepath, target, conn, keep_unknown_province)
        all_rows.extend(results)
        all_logs.extend(logs)
        month_counts[month] = len(results)

    conn.close()

    # 3. Build output DataFrame
    print(f"\n[3/3] Writing outputs...")
    df = pd.DataFrame(all_rows)
    print(f"  Total kept: {len(df)} rows")
    print(f"  Months: {df['date_month'].nunique()}, Weeks: {df['date_week'].nunique()}, Provinces: {df['province'].nunique()}")

    # Save mini dataset
    output_path = PROCESSED_DIR / "mini_dataset.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"  [OK] Mini dataset: {output_path} ({size_mb:.1f} MB)")

    # Save sampling log
    log_path = TMP_DIR / "01_sampling_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Script 01 Sampling Log -- {datetime.now()}\n")
        f.write(f"# Total skipped: {len(all_logs)}\n\n")
        for log in all_logs[:1000]:  # cap log lines
            f.write(log + "\n")
    print(f"  [OK] Sampling log: {log_path}")

    # Save month x province distribution
    dist = df.groupby(["date_month", "province"]).size().reset_index(name="count")
    dist_path = TMP_DIR / "01_sample_distribution.csv"
    dist.to_csv(dist_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] Distribution: {dist_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total posts:        {len(df):,}")
    print(f"Date range:         {df['date_week'].min()} to {df['date_week'].max()}")
    print(f"Provinces:          {df['province'].nunique()}")
    print(f"Months with data:   {df['date_month'].nunique()}")
    print(f"Avg word count:     {df['word_count'].mean():.0f}")
    print(f"Median word count:  {df['word_count'].median():.0f}")

    # Monthly sample counts
    print("\nMonthly sample counts:")
    for m in sorted(month_counts.keys()):
        bar = "#" * (month_counts[m] // 200)
        print(f"  {m}: {month_counts[m]:>5} {bar}")

    print("\nDone.")
    return df


if __name__ == "__main__":
    main()
