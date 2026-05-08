"""
Script 05: Anomaly Detection
Rolling Z-score detection on fear, anger, joy national timelines.
Outputs: data/processed/anomaly_detection.json + anomaly contribution data
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

EMOTIONS_TO_CHECK = ["fear_mean", "anger_mean", "joy_mean"]
Z_THRESHOLD = 2.5
ROLLING_WINDOW = 4

SEVERITY_BINS = [
    (2.5, 3.0, "moderate"),
    (3.0, 3.5, "severe"),
    (3.5, 100, "extreme"),
]


def classify_severity(z):
    z_abs = abs(z)
    for lo, hi, label in SEVERITY_BINS:
        if lo <= z_abs < hi:
            return label
    return "moderate"


def format_deviation_pct(value, expected):
    """Percent deviation from expected baseline, not z-score scaled percent."""
    if pd.isna(expected) or abs(expected) < 1e-8:
        return "N/A"
    pct = (float(value) - float(expected)) / abs(float(expected)) * 100
    return f"{pct:+.0f}%"


def detect_anomalies(national):
    """Compute rolling z-scores and detect anomalies for each emotion dimension.
    Filters to weeks with total_posts >= 30 to avoid small-sample noise."""
    df = national.sort_values("date_week").copy()
    # Filter out weeks with insufficient national posts
    if "total_posts" in df.columns:
        before_count = len(df)
        df = df[df["total_posts"] >= 30].reset_index(drop=True)
        dropped = before_count - len(df)
        if dropped > 0:
            print(f"  [INFO] Dropped {dropped} weeks with total_posts < 30")
    anomalies = []

    for emotion_col in EMOTIONS_TO_CHECK:
        series = df[emotion_col]
        baseline = series.shift(1)
        rolling_mean = baseline.rolling(ROLLING_WINDOW, min_periods=2).mean()
        rolling_std = baseline.rolling(ROLLING_WINDOW, min_periods=2).std()
        df[f"{emotion_col}_zscore"] = (series - rolling_mean) / (rolling_std + 1e-8)
        df[f"{emotion_col}_rolling_mean"] = rolling_mean
        df[f"{emotion_col}_rolling_std"] = rolling_std

    # Find anomalies
    for emotion_col in EMOTIONS_TO_CHECK:
        z_col = f"{emotion_col}_zscore"
        anomaly_mask = abs(df[z_col]) > Z_THRESHOLD
        for idx in df[anomaly_mask].index:
            row = df.loc[idx]
            z_val = row[z_col]
            anomalies.append({
                "date_week": row["date_week"],
                "emotion": emotion_col.replace("_mean", ""),
                "z_score": round(float(z_val), 2),
                "national_value": round(float(row[emotion_col]), 4),
                "expected_value": round(float(row[f"{emotion_col}_rolling_mean"]), 4),
                "deviation_pct": format_deviation_pct(
                    row[emotion_col], row[f"{emotion_col}_rolling_mean"]
                ),
                "severity": classify_severity(z_val),
            })

    return anomalies, df


def find_top_contributors(anomalies, weekly_panel):
    """For each anomaly week, find top 5 contributing provinces.
    Provinces must have total_posts >= 30 to be considered."""
    for anom in anomalies:
        week = anom["date_week"]
        emotion_col = f"{anom['emotion']}_mean"
        week_data = weekly_panel[weekly_panel["date_week"] == week].copy()
        if len(week_data) == 0:
            anom["top_provinces"] = []
            anom["top_provinces_reason"] = "no_weekly_panel_data"
            continue

        # Filter to provinces with sufficient posts
        if "total_posts" in week_data.columns:
            week_data = week_data[week_data["total_posts"] >= 30].copy()
        elif "reliable" in week_data.columns:
            week_data = week_data[week_data["reliable"] == True].copy()

        if len(week_data) == 0:
            anom["top_provinces"] = []
            anom["top_provinces_reason"] = "no_province_with_posts_gte_30"
            continue

        # Sort by the anomaly emotion, descending if z > 0 (spike), ascending if z < 0 (dip)
        ascending = anom["z_score"] < 0
        top = week_data.sort_values(emotion_col, ascending=ascending).head(5)
        anom["top_provinces"] = [
            {
                "province": str(r["province"]),
                f"{anom['emotion']}_mean": round(float(r[emotion_col]), 4),
                "posts": int(r["total_posts"]),
            }
            for _, r in top.iterrows()
        ]
        # Remove reason field if provinces were found
        anom.pop("top_provinces_reason", None)

    return anomalies


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import argparse
    parser = argparse.ArgumentParser(description="Anomaly Detection")
    parser.add_argument("--national", type=str, default=None,
                        help="National timeline CSV (default: data/processed/emotion_national_timeline.csv)")
    parser.add_argument("--weekly", type=str, default=None,
                        help="Weekly panel CSV (default: data/processed/emotion_panel_weekly.csv)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: data/processed/anomaly_detection.json)")
    parser.add_argument("--summary", type=str, default=None,
                        help="Summary output path")
    args = parser.parse_args()

    print("=" * 60)
    print("Script 05: Anomaly Detection")
    print("=" * 60)

    timeline_path = Path(args.national) if args.national else PROCESSED_DIR / "emotion_national_timeline.csv"
    if not timeline_path.is_absolute():
        timeline_path = ROOT / timeline_path
    panel_path = Path(args.weekly) if args.weekly else PROCESSED_DIR / "emotion_panel_weekly.csv"
    if not panel_path.is_absolute():
        panel_path = ROOT / panel_path
    output_path = Path(args.output) if args.output else PROCESSED_DIR / "anomaly_detection.json"
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if not timeline_path.exists():
        print(f"[ERROR] emotion_national_timeline.csv not found. Run Script 04 first.")
        sys.exit(1)

    print(f"\nLoading {timeline_path}...")
    national = pd.read_csv(timeline_path, encoding="utf-8-sig")
    print(f"  {len(national)} weeks of national data")

    # Get weekly panel for province breakdown
    if panel_path.exists():
        weekly_panel = pd.read_csv(panel_path, encoding="utf-8-sig")
    else:
        weekly_panel = pd.DataFrame()

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Detect anomalies
    print(f"\n[1/2] Detecting anomalies (|z| > {Z_THRESHOLD}, window={ROLLING_WINDOW})...")
    anomalies, df_zscores = detect_anomalies(national)

    print(f"  Found {len(anomalies)} anomalies:")
    for a in anomalies:
        print(f"    {a['date_week']} {a['emotion']}: z={a['z_score']:.2f} "
              f"({a['severity']}) val={a['national_value']:.4f} {a['deviation_pct']}")

    # 2. Find top contributing provinces
    print("\n[2/2] Finding top contributing provinces...")
    anomalies = find_top_contributors(anomalies, weekly_panel)

    for a in anomalies:
        if a["top_provinces"]:
            provs = ", ".join(p["province"] for p in a["top_provinces"][:3])
            print(f"    {a['date_week']} {a['emotion']}: top provinces = {provs}")
        else:
            reason = a.get("top_provinces_reason", "unknown")
            print(f"    [WARN] {a['date_week']} {a['emotion']}: no qualifying provinces ({reason})")

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(anomalies, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Anomaly detection saved: {output_path}")

    # Save z-score timeseries for debugging
    zscore_cols = ["date_week"] + [f"{c}_zscore" for c in EMOTIONS_TO_CHECK] + [c for c in EMOTIONS_TO_CHECK]
    zscore_df = df_zscores[[c for c in zscore_cols if c in df_zscores.columns]]
    zscore_df.to_csv(TMP_DIR / "05_zscore_timeseries.csv", index=False, encoding="utf-8-sig")

    # Save province contribution details
    contrib_rows = []
    for a in anomalies:
        for p in a.get("top_provinces", []):
            contrib_rows.append({
                "anomaly_week": a["date_week"],
                "emotion": a["emotion"],
                "province": p["province"],
                f"{a['emotion']}_mean": p.get(f"{a['emotion']}_mean", 0),
                "posts": p["posts"],
            })
    if contrib_rows:
        pd.DataFrame(contrib_rows).to_csv(
            TMP_DIR / "05_anomaly_contributions.csv", index=False, encoding="utf-8-sig"
        )

    empty_tp = sum(1 for a in anomalies if not a.get("top_provinces"))
    if empty_tp > 0:
        print(f"  [WARN] {empty_tp}/{len(anomalies)} anomalies have empty top_provinces")
    print(f"Done. {len(anomalies)} anomalies detected across {len(EMOTIONS_TO_CHECK)} dimensions.")


if __name__ == "__main__":
    main()
