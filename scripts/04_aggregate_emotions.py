"""
Script 04: Emotion Aggregation
Aggregates labeled data into weekly/monthly/national panels + province vectors.
Outputs: emotion_panel_weekly/monthly, national timeline, province vectors, wordclouds
"""
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jieba
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
ANALYSIS_DIR = ROOT / "analysis"
TMP_DIR = ROOT / "tmp"

from _config import EMOTION_KEYS, VALID_PROVINCES

EMOTION_LABELS = ["喜悦", "悲伤", "愤怒", "恐惧", "惊讶", "中性"]
EMOTION_CN = dict(zip(EMOTION_KEYS, EMOTION_LABELS))
MIN_POSTS_RELIABLE = int(os.getenv("MIN_POSTS_RELIABLE", "30"))

# WordCloud font — try common Chinese font paths
FONT_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/SimHei.ttf",
]


def find_font():
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            return fp
    return None


def compute_aggregations(df):
    """Compute all aggregation panels"""
    required = ["post_id", "date_week", "date_month", "province", "word_count"] + EMOTION_KEYS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"labeled_dataset.csv missing required columns: {missing}")

    province_df = df[df["province"].isin(VALID_PROVINCES)].copy()

    # Build reusable aggregation dicts
    EMOTION_MEAN_AGGS = {f"{k}_mean": (k, "mean") for k in EMOTION_KEYS}
    EMOTION_STD_AGGS = {f"{k}_std": (k, "std") for k in EMOTION_KEYS}
    FULL_WEEKLY_AGGS = {"total_posts": ("post_id", "count"), "avg_word_count": ("word_count", "mean")}
    FULL_WEEKLY_AGGS.update(EMOTION_MEAN_AGGS)
    FULL_WEEKLY_AGGS.update(EMOTION_STD_AGGS)
    MONTHLY_AGGS = {"total_posts": ("post_id", "count")}
    MONTHLY_AGGS.update(EMOTION_MEAN_AGGS)
    MONTHLY_AGGS["joy_std"] = ("joy", "std")
    MONTHLY_AGGS["fear_std"] = ("fear", "std")
    MONTHLY_AGGS["emotional_intensity"] = ("neutral", lambda x: 1 - x.mean())
    NATIONAL_AGGS = FULL_WEEKLY_AGGS.copy()
    NATIONAL_AGGS["emotional_intensity"] = ("neutral", lambda x: 1 - x.mean())
    VECTOR_AGGS = {f"{k}_mean_all": (k, "mean") for k in EMOTION_KEYS}
    VECTOR_AGGS["total_posts_all"] = ("post_id", "count")
    VECTOR_AGGS["emotional_intensity_mean"] = ("neutral", lambda x: 1 - x.mean())
    VECTOR_AGGS["fear_variance"] = ("fear", "var")
    VECTOR_AGGS["joy_variance"] = ("joy", "var")

    def _add_derived_cols(df_in, emotion_col_prefix="", include_positive=True, include_intensity=True):
        """Add dominant emotion, positive_index, emotional_intensity to a panel DataFrame."""
        df_out = df_in.copy()
        cols = [f"{emotion_col_prefix}{k}_mean" for k in EMOTION_KEYS] if emotion_col_prefix else [f"{k}_mean" for k in EMOTION_KEYS]
        df_out["dominant_emotion_key"] = df_out[cols].idxmax(axis=1).str.replace("_mean", "").replace(f"{emotion_col_prefix}", "", regex=False)
        df_out["dominant_emotion"] = df_out["dominant_emotion_key"].map(EMOTION_CN)
        df_out["dominant_score"] = df_out[cols].max(axis=1)
        if include_positive and "joy_mean" in df_out.columns:
            df_out["positive_index"] = df_out["joy_mean"] / (
                df_out["sadness_mean"] + df_out["anger_mean"] + df_out["fear_mean"] + 0.01
            )
        if include_intensity and "neutral_mean" in df_out.columns:
            df_out["emotional_intensity"] = 1 - df_out["neutral_mean"]
        if "fear_mean" in df_out.columns and "joy_mean" in df_out.columns:
            df_out["fear_joy_ratio"] = df_out["fear_mean"] / (df_out["joy_mean"] + 0.01)
        return df_out

    # Weekly x province
    weekly = province_df.groupby(["date_week", "province"]).agg(**FULL_WEEKLY_AGGS).reset_index()
    weekly = _add_derived_cols(weekly)
    weekly["reliable"] = weekly["total_posts"] >= MIN_POSTS_RELIABLE

    # Monthly x province
    monthly = province_df.groupby(["date_month", "province"]).agg(**MONTHLY_AGGS).reset_index()
    monthly = _add_derived_cols(monthly)
    monthly["reliable"] = monthly["total_posts"] >= MIN_POSTS_RELIABLE

    # National timeline (weekly, no province split)
    national_weekly = df.groupby("date_week").agg(**NATIONAL_AGGS).reset_index()
    national_weekly = _add_derived_cols(national_weekly)

    # Province vectors (full period)
    province_vecs = province_df.groupby("province").agg(**VECTOR_AGGS).reset_index()

    return weekly, monthly, national_weekly, province_vecs


def generate_wordclouds(df, national_weekly):
    """Generate wordclouds for key weeks"""
    font_path = find_font()
    wc_dir = ANALYSIS_DIR / "wordclouds"
    wc_dir.mkdir(parents=True, exist_ok=True)

    # National wordclouds for key weeks: peak fear, peak joy, first week
    all_texts = df.groupby("date_week")["content_clean"].apply(" ".join)
    peak_fear_week = national_weekly.sort_values("fear_mean", ascending=False).iloc[0]["date_week"]
    peak_joy_week = national_weekly.sort_values("joy_mean", ascending=False).iloc[0]["date_week"]

    for label, week in [("peak_fear", peak_fear_week), ("peak_joy", peak_joy_week)]:
        texts = all_texts.get(week, "")
        if not texts:
            continue
        words = jieba.cut(texts)
        word_str = " ".join(words)
        wc = WordCloud(
            width=800, height=600,
            font_path=font_path,
            background_color="#1a1a2e",
            max_words=100,
            colormap="plasma",
            collocations=False,
        ).generate(word_str)
        wc_path = wc_dir / f"national_{label}.png"
        wc.to_file(str(wc_path))
        print(f"  [OK] WordCloud: {wc_path}")

    # Top province wordclouds use valid map provinces only.
    province_df = df[df["province"].isin(VALID_PROVINCES)].copy()
    top_provinces = province_df["province"].value_counts().head(6).index.tolist()
    prov_texts = province_df.groupby("province")["content_clean"].apply(" ".join)
    for prov in top_provinces:
        texts = prov_texts.get(prov, "")
        if not texts:
            continue
        words = jieba.cut(texts)
        word_str = " ".join(words)
        wc = WordCloud(
            width=800, height=600,
            font_path=font_path,
            background_color="#1a1a2e",
            max_words=80,
            colormap="viridis",
            collocations=False,
        ).generate(word_str)
        wc_path = wc_dir / f"province_{prov}.png"
        wc.to_file(str(wc_path))
        print(f"  [OK] WordCloud: {wc_path}")

    # Save keyword frequency table
    keyword_rows = []
    for week in [peak_fear_week, peak_joy_week]:
        texts = all_texts.get(week, "")
        if texts:
            words = jieba.lcut(texts)
            word_freq = pd.Series(words).value_counts().head(50)
            for word, freq in word_freq.items():
                if len(word) >= 2:
                    keyword_rows.append({"week": week, "word": word, "freq": freq})
    if keyword_rows:
        kw_df = pd.DataFrame(keyword_rows)
        kw_df.to_csv(TMP_DIR / "04_keyword_tables.csv", index=False, encoding="utf-8-sig")
        print(f"  [OK] Keyword table: {TMP_DIR / '04_keyword_tables.csv'}")


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import argparse
    parser = argparse.ArgumentParser(description="Emotion Aggregation")
    parser.add_argument("--input", type=str, default=None,
                        help="Input labeled CSV (default: data/processed/labeled_dataset.csv)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/processed/)")
    parser.add_argument("--summary", type=str, default=None,
                        help="Summary output path (default: tmp/04_aggregation_summary.txt)")
    args = parser.parse_args()

    print("=" * 60)
    print("Script 04: Emotion Aggregation")
    print("=" * 60)

    input_path = Path(args.input) if args.input else PROCESSED_DIR / "labeled_dataset.csv"
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DIR
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else TMP_DIR / "04_aggregation_summary.txt"
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path

    if not input_path.exists():
        print(f"[ERROR] labeled_dataset.csv not found at {input_path}")
        print("Run Script 02 first.")
        sys.exit(1)

    print(f"\nLoading {input_path}...")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"  {len(df):,} posts, {df['date_week'].nunique()} weeks, {df['province'].nunique()} provinces")

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Aggregation
    print("\n[1/2] Computing aggregations...")
    weekly, monthly, national, province_vecs = compute_aggregations(df)

    weekly.to_csv(output_dir / "emotion_panel_weekly.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output_dir / "emotion_panel_monthly.csv", index=False, encoding="utf-8-sig")
    national.to_csv(output_dir / "emotion_national_timeline.csv", index=False, encoding="utf-8-sig")
    province_vecs.to_csv(output_dir / "province_emotion_vectors.csv", index=False, encoding="utf-8-sig")

    print(f"  Weekly panel:     {len(weekly):,} rows")
    print(f"  Monthly panel:    {len(monthly):,} rows")
    print(f"  National timeline:{len(national):,} weeks")
    print(f"  Province vectors: {len(province_vecs)} provinces")

    # 2. Wordclouds
    print("\n[2/2] Generating wordclouds...")
    generate_wordclouds(df, national)

    # Summary
    summary = {
        "national_weeks": len(national),
        "provinces": df["province"].nunique(),
        "weekly_rows": len(weekly),
        "monthly_rows": len(monthly),
        "weekly_reliable_pct": round(weekly["reliable"].mean() * 100, 1),
        "monthly_reliable_pct": round(monthly["reliable"].mean() * 100, 1),
        "weekly_reliable_count": int(weekly["reliable"].sum()),
        "monthly_reliable_count": int(monthly["reliable"].sum()),
        "min_posts_reliable": MIN_POSTS_RELIABLE,
        "national_dominant_emotion": national["dominant_emotion"].mode().iloc[0] if len(national) > 0 else "N/A",
        "timestamp": datetime.now().isoformat(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print(f"\n[OK] All outputs saved to {output_dir}")
    print(f"  Summary: {summary_path}")
    print(f"  Summary: {summary}")

    print("\nDone.")


if __name__ == "__main__":
    main()
