"""
Script 00: Data Feasibility Gate
Checks whether province-level V1 is viable before spending money on labeling.
Outputs: tmp/00_feasibility_report.md + tmp/00_feasibility_report.json
"""
import pandas as pd
import numpy as np
import json
import os
import re
import sys
import csv
import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "COV-Weibo2.0"
SMP2020_DIR = ROOT / "data" / "raw" / "SMP2020_EWECT"
TMP_DIR = ROOT / "tmp"
RANDOM_SEED = 42
SAMPLE_SIZE = 5000
POST_SAMPLE_ROWS = 50_000  # larger sample for geo_info rate check

POST_MONTHS = [f"2020-{m:02d}" for m in range(1, 13)] + ["2019-12"]
REQUIRED_POST_COLS = ["_id", "user_id", "created_at", "content"]
REQUIRED_USER_COLS = ["user_id", "province"]
REQUIRED_SMP_COLS = ["id", "content", "label"]
from _config import VALID_PROVINCES


def check_files():
    """1. File integrity check"""
    issues = []
    post_files = {}
    for m in POST_MONTHS:
        p = DATA_DIR / f"{m}.csv"
        if p.exists():
            post_files[m] = p
        else:
            issues.append(f"MISSING: {m}.csv")
    user_file = DATA_DIR / "user.csv"
    if not user_file.exists():
        issues.append("MISSING: user.csv")
    return post_files, user_file, issues


def check_schema(post_files, user_file):
    """2. Schema validation"""
    issues = []
    # Check one post file
    first = list(post_files.values())[0]
    df = pd.read_csv(first, nrows=3, encoding="utf-8-sig")
    post_cols = df.columns.tolist()
    for c in REQUIRED_POST_COLS:
        if c not in post_cols:
            issues.append(f"Post table missing column: {c}")
    # Check user file
    df_u = pd.read_csv(user_file, nrows=3, encoding="utf-8-sig")
    user_cols = df_u.columns.tolist()
    for c in REQUIRED_USER_COLS:
        if c not in user_cols:
            issues.append(f"User table missing column: {c}")
    # Check SMP2020 (JSON format: one-line array of objects)
    for fname in ["virus_train.txt", "virus_eval_labeled.txt", "virus_test_labeled.txt"]:
        fpath = SMP2020_DIR / fname
        if fpath.exists():
            try:
                raw = fpath.read_text(encoding="utf-8-sig").strip()
                data = json.loads(raw)
                if isinstance(data, list) and len(data) > 0:
                    smp_cols = list(data[0].keys())
                    for c in REQUIRED_SMP_COLS:
                        if c not in smp_cols:
                            issues.append(f"SMP2020 {fname} missing column: {c}")
                else:
                    issues.append(f"SMP2020 {fname}: unexpected JSON structure")
            except Exception:
                # Fallback: try TSV
                try:
                    df_s = pd.read_csv(fpath, sep="\t", nrows=3, encoding="utf-8-sig")
                    smp_cols = df_s.columns.tolist()
                    for c in REQUIRED_SMP_COLS:
                        if c not in smp_cols:
                            issues.append(f"SMP2020 {fname} missing column: {c}")
                except Exception as e2:
                    issues.append(f"SMP2020 {fname}: unable to parse ({e2})")
        else:
            issues.append(f"SMP2020 {fname} not found (non-blocking)")
    return post_cols, user_cols, issues


def human_readable_size(path):
    gb = path.stat().st_size / (1024**3)
    return f"{gb:.1f} GB"


def sample_post_quality(post_files):
    """3. Sampling quality check"""
    results = {}
    for month, path in post_files.items():
        df = pd.read_csv(path, nrows=SAMPLE_SIZE, encoding="utf-8-sig")
        total = len(df)
        created_ok = 0
        content_ok = 0
        uid_ok = 0
        geo_ok = 0
        chinese_chars = []
        for _, row in df.iterrows():
            try:
                pd.to_datetime(row["created_at"])
                created_ok += 1
            except Exception:
                pass
            c = str(row.get("content", ""))
            if c and c != "nan":
                content_ok += 1
                chinese_chars.append(len(re.findall(r"[一-鿿]", c)))
            if str(row.get("user_id", "")).strip() and str(row["user_id"]) != "nan":
                uid_ok += 1
            g = str(row.get("geo_info", ""))
            if g and g != "nan" and g.strip():
                geo_ok += 1
        results[month] = {
            "sampled": total,
            "created_at_parse_rate": created_ok / total,
            "content_non_null_rate": content_ok / total,
            "user_id_non_null_rate": uid_ok / total,
            "geo_info_non_null_rate": geo_ok / total,
            "chinese_char_mean": float(np.mean(chinese_chars)) if chinese_chars else 0,
            "chinese_char_median": float(np.median(chinese_chars)) if chinese_chars else 0,
        }
    return results


def build_user_index(user_file):
    """4. Build user SQLite index for fast lookup"""
    db_path = TMP_DIR / f"00_user_index_probe_{os.getpid()}.db"
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass
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

    count = 0
    kept = 0
    with open(user_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            count += 1
            province = str(row.get("province", "")).strip()
            if province not in VALID_PROVINCES:
                continue
            kept += 1
            batch.append(
                (
                    str(row.get("user_id", "")).strip(),
                    province,
                    str(row.get("city", "")).strip(),
                    str(row.get("gender", "")).strip(),
                )
            )
            if len(batch) >= 100_000:
                conn.executemany(
                    "INSERT OR IGNORE INTO users VALUES (?,?,?,?)", batch
                )
                batch = []
        if batch:
            conn.executemany(
                "INSERT OR IGNORE INTO users VALUES (?,?,?,?)", batch
            )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON users(user_id)")
    conn.commit()
    return conn, count, kept, db_path


def check_province_association(post_files, conn):
    """4b. Province association check with larger sample"""
    join_hits = 0
    join_misses = 0
    valid_province = 0
    invalid_province = 0
    province_month = {}  # province -> {month -> count}
    province_week = {}   # province -> {week -> count}
    months_to_sample = sorted(post_files.keys())
    sample_per_month = POST_SAMPLE_ROWS // len(months_to_sample)

    for month in months_to_sample:
        path = post_files[month]
        df = pd.read_csv(path, nrows=sample_per_month, encoding="utf-8-sig")
        for _, row in df.iterrows():
            uid = str(row.get("user_id", "")).strip()
            if not uid or uid == "nan":
                continue
            cur = conn.execute(
                "SELECT province, city, gender FROM users WHERE user_id = ?", (uid,)
            )
            result = cur.fetchone()
            if result is None:
                join_misses += 1
                province = "未知"
            else:
                join_hits += 1
                province = result[0]
            if province not in VALID_PROVINCES:
                invalid_province += 1
            else:
                valid_province += 1

            # Track distribution
            if province not in province_month:
                province_month[province] = {}
            province_month[province][month] = province_month[province].get(month, 0) + 1

            # Try week
            try:
                dt = pd.to_datetime(row.get("created_at", ""))
                week = dt.strftime("%G-W%V")
                if province not in province_week:
                    province_week[province] = {}
                province_week[province][week] = province_week[province].get(week, 0) + 1
            except Exception:
                pass

    total = join_hits + join_misses
    return {
        "total_rows": total,
        "join_hits": join_hits,
        "join_rate": join_hits / total if total else 0,
        "valid_province_count": valid_province,
        "valid_province_rate": valid_province / total if total else 0,
        "province_month_dist": province_month,
        "province_week_dist": province_week,
    }


def make_decision(quality, assoc):
    """5. V1 decision"""
    warnings = []
    valid_rate = assoc["valid_province_rate"]
    join_rate = assoc["join_rate"]

    if join_rate < 0.8:
        warnings.append(f"User join rate {join_rate:.1%} < 80% -- check user.csv completeness")

    # Check weekly per-province samples
    prov_week = assoc["province_week_dist"]
    prov_week_counts = []
    for p, weeks in prov_week.items():
        if p not in VALID_PROVINCES:
            continue
        prov_week_counts.extend(weeks.values())

    avg_week_prov = float(np.mean(prov_week_counts)) if prov_week_counts else 0
    median_week_prov = float(np.median(prov_week_counts)) if prov_week_counts else 0

    # Count provinces with median weekly >= 30
    good_provinces = 0
    total_provinces = 0
    for p, weeks in prov_week.items():
        if p not in VALID_PROVINCES:
            continue
        total_provinces += 1
        vals = list(weeks.values())
        if vals:
            med = np.median(vals)
            if med >= 30:
                good_provinces += 1

    can_do_province = valid_rate >= 0.80 and good_provinces >= 20
    if valid_rate < 0.80:
        warnings.append(f"Valid province rate {valid_rate:.1%} < 80% -- province map V1 degraded")
    if good_provinces < 20:
        warnings.append(f"Only {good_provinces}/{total_provinces} provinces have median weekly >= 30")

    if can_do_province:
        if median_week_prov >= 30:
            granularity = "weekly_province"
        else:
            granularity = "monthly_province"
            warnings.append("Weekly per-province samples insufficient, using monthly for maps")
    else:
        granularity = "national_timeline_only"
        warnings.append("Province-level V1 not feasible; downgrading to national timeline + topic anomalies")

    return {
        "can_do_province_v1": can_do_province,
        "recommended_geo_granularity": granularity,
        "good_provinces_count": good_provinces,
        "total_provinces_count": total_provinces,
        "avg_weekly_per_province": round(avg_week_prov, 1),
        "median_weekly_per_province": round(median_week_prov, 1),
        "warnings": warnings,
    }


def write_report(
    quality,
    assoc,
    decision,
    post_files,
    user_file,
    schema_issues,
    file_issues,
    user_index_info,
):
    """Write markdown report"""
    lines = []
    lines.append("# Data Feasibility Report -- Script 00")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## 1. File Integrity")
    lines.append(f"- Post files found: {len(post_files)}/13")
    lines.append(f"- User file: {'OK' if user_file.exists() else 'MISSING'}")
    if file_issues:
        for i in file_issues:
            lines.append(f"  - [WARN] {i}")
    lines.append("")

    lines.append("## 2. Schema Validation")
    if schema_issues:
        for i in schema_issues:
            lines.append(f"  - [WARN] {i}")
    else:
        lines.append("- All required columns present [OK]")
    lines.append("")

    lines.append("## 3. Sampling Quality (5,000 rows/month)")
    lines.append(
        "| Month | Sampled | created_at OK | content OK | user_id OK | geo_info OK | zh chars (med) |"
    )
    lines.append(
        "|-------|---------|---------------|------------|------------|-------------|----------------|"
    )
    for month in sorted(quality.keys()):
        q = quality[month]
        lines.append(
            f"| {month} | {q['sampled']} | {q['created_at_parse_rate']:.1%} "
            f"| {q['content_non_null_rate']:.1%} | {q['user_id_non_null_rate']:.1%} "
            f"| {q['geo_info_non_null_rate']:.2%} | {q['chinese_char_median']:.0f} |"
        )
    lines.append("")

    lines.append("## 4. User Geo Association")
    lines.append(f"- Users in index (after filtering): {user_index_info[1]:,} / {user_index_info[0]:,} total")
    lines.append(f"- Join hit rate: {assoc['join_rate']:.2%}")
    lines.append(f"- Valid province rate: {assoc['valid_province_rate']:.2%}")
    lines.append("")

    lines.append("## 5. V1 Decision")
    lines.append(f"- **Can do province V1**: {decision['can_do_province_v1']}")
    lines.append(f"- **Recommended granularity**: {decision['recommended_geo_granularity']}")
    lines.append(f"- Good provinces (median weekly >= 30): {decision['good_provinces_count']}/{decision['total_provinces_count']}")
    lines.append(f"- Average weekly posts per province: {decision['avg_weekly_per_province']:.0f}")
    lines.append(f"- Median weekly posts per province: {decision['median_weekly_per_province']:.0f}")
    if decision["warnings"]:
        lines.append("")
        lines.append("### Warnings")
        for w in decision["warnings"]:
            lines.append(f"- [WARN] {w}")
    lines.append("")

    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("Script 00: Data Feasibility Gate")
    print("=" * 60)

    # 1. File integrity
    print("\n[1/5] Checking file integrity...")
    post_files, user_file, file_issues = check_files()
    print(f"  Post files: {len(post_files)}/13, User file: {'OK' if user_file.exists() else 'MISSING'}")

    # 2. Schema
    print("\n[2/5] Validating schema...")
    post_cols, user_cols, schema_issues = check_schema(post_files, user_file)
    print(f"  Post columns: {len(post_cols)}, User columns: {len(user_cols)}")
    if schema_issues:
        for i in schema_issues:
            print(f"  [WARN] {i}")

    # 3. Sampling
    print("\n[3/5] Sampling quality check (5,000 rows/month)...")
    quality = sample_post_quality(post_files)
    for month in sorted(quality.keys()):
        q = quality[month]
        print(f"  {month}: content={q['content_non_null_rate']:.1%} geo={q['geo_info_non_null_rate']:.2%} zh_med={q['chinese_char_median']:.0f}")

    # 4. User geo association
    print("\n[4/5] Building user index and checking province association...")
    conn, user_total, user_kept, db_path = build_user_index(user_file)
    user_index_info = (user_total, user_kept, db_path)
    print(f"  Users: {user_kept:,} kept / {user_total:,} total")
    assoc = check_province_association(post_files, conn)
    print(f"  Join rate: {assoc['join_rate']:.2%}, Valid province rate: {assoc['valid_province_rate']:.2%}")
    conn.close()

    # 5. Decision
    print("\n[5/5] Making V1 decision...")
    decision = make_decision(quality, assoc)
    print(f"  Province V1: {decision['can_do_province_v1']}")
    print(f"  Granularity: {decision['recommended_geo_granularity']}")
    for w in decision["warnings"]:
        print(f"  [WARN] {w}")

    # Write outputs
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    report = write_report(quality, assoc, decision, post_files, user_file, schema_issues, file_issues, user_index_info)
    report_path = TMP_DIR / "00_feasibility_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[OK] Report written to {report_path}")

    json_output = {
        "can_do_province_v1": decision["can_do_province_v1"],
        "recommended_geo_granularity": decision["recommended_geo_granularity"],
        "post_files": len(post_files),
        "user_file_exists": user_file.exists(),
        "geo_info_non_null_rate": float(np.mean([q["geo_info_non_null_rate"] for q in quality.values()])),
        "user_join_rate": assoc["join_rate"],
        "valid_province_rate": assoc["valid_province_rate"],
        "good_provinces": decision["good_provinces_count"],
        "total_provinces": decision["total_provinces_count"],
        "avg_weekly_per_province": decision["avg_weekly_per_province"],
        "median_weekly_per_province": decision["median_weekly_per_province"],
        "warnings": decision["warnings"],
        "file_issues": file_issues,
        "schema_issues": schema_issues,
    }
    json_path = TMP_DIR / "00_feasibility_report.json"
    json_path.write_text(json.dumps(json_output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] JSON written to {json_path}")

    print("\nDone.")
    return json_output


if __name__ == "__main__":
    main()
