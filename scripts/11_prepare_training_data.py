"""
Script 13a: Prepare Training Data
Loads labeled_dataset_merged_week_cap60.csv and splits into 80/10/10
train/val/test stratified by province x date_week.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
TRAINING_DIR = PROCESSED_DIR / "training"
RANDOM_SEED = 42

from _config import VALID_PROVINCES


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    input_path = PROCESSED_DIR / "labeled_dataset_merged_week_cap60.csv"
    if not input_path.exists():
        print(f"[ERROR] {input_path} not found")
        sys.exit(1)

    print("=" * 60)
    print("Script 13a: Prepare Training Data")
    print("=" * 60)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"\nLoaded: {len(df):,} rows")

    # Filter valid provinces and successful labels
    df = df[df["province"].isin(VALID_PROVINCES)].copy()
    if "label_status" in df.columns:
        df = df[df["label_status"] == "ok"].copy()
    print(f"After filtering (valid province + label ok): {len(df):,} rows")

    # Build stratify key: province + date_week
    df["stratify_key"] = df["province"].astype(str) + "|" + df["date_week"].astype(str)

    # Merge rare strata (< 10 samples) into "OTHER" so all 3 splits get >= 1
    key_counts = df["stratify_key"].value_counts()
    rare_keys = set(key_counts[key_counts < 10].index)
    df["stratify_key"] = df["stratify_key"].apply(
        lambda k: "OTHER" if k in rare_keys else k
    )
    kept = len(key_counts) - len(rare_keys)
    print(f"Strata: {len(key_counts)} → {kept} after merging rare (<10) strata")

    # 80/10/10 split
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=RANDOM_SEED,
        stratify=df["stratify_key"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED,
        stratify=temp_df["stratify_key"]
    )

    # Drop the helper column
    for d in (train_df, val_df, test_df):
        d.drop(columns=["stratify_key"], inplace=True)

    print(f"\nSplit: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # Verify coverage
    train_provinces = set(train_df["province"].unique())
    print(f"Provinces covered — train: {len(train_provinces)}, val: {val_df['province'].nunique()}, test: {test_df['province'].nunique()}")
    missing_in_train = VALID_PROVINCES - train_provinces
    if missing_in_train:
        print(f"⚠ Missing provinces in train: {missing_in_train}")

    # Save
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAINING_DIR / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(TRAINING_DIR / "val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(TRAINING_DIR / "test.csv", index=False, encoding="utf-8-sig")
    print(f"\n[OK] Saved to {TRAINING_DIR}/")

    # Quick stats
    print(f"\nLabel distribution (train):")
    emotion_keys = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
    for k in emotion_keys:
        print(f"  {k:10s}: mean={train_df[k].mean():.4f}  std={train_df[k].std():.4f}")
    argmax = train_df[emotion_keys].idxmax(axis=1).value_counts(normalize=True)
    print(f"\nArgmax class distribution (train):")
    for k in emotion_keys:
        print(f"  {k:10s}: {argmax.get(k, 0):.2%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
