"""Extract key data summaries for paper writing — UTF-8 clean output."""
import json, sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "app" / "public" / "data" / "processed"
sys.stdout.reconfigure(encoding='utf-8')

EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
EMOTION_CN = {"joy":"喜悦","sadness":"悲伤","anger":"愤怒","fear":"恐惧","surprise":"惊讶","neutral":"中性"}
HISTORICAL_END_WEEK = "2020-W53"

# ── 1. Cluster Analysis ─────────────────────────
print("=" * 60)
print("1. PROVINCE CLUSTERING")
print("=" * 60)
clusters = pd.read_csv(DATA / "cluster_labels.csv")
vectors = pd.read_csv(DATA / "province_emotion_vectors.csv")
df = vectors.merge(clusters[["province", "cluster_label"]], on="province", how="left")

for label in sorted(df["cluster_label"].unique()):
    group = df[df["cluster_label"] == label]
    provinces = group["province"].tolist()
    print(f"\nCluster {label} ({len(provinces)} provinces): {', '.join(provinces)}")
    # Per-cluster emotion means
    cols = [f"{k}_mean_all" for k in EMOTION_KEYS]
    means = {k: round(group[f"{k}_mean_all"].mean(), 4) for k in EMOTION_KEYS}
    dominant = max(means, key=means.get)
    print(f"  Dominant emotion: {EMOTION_CN[dominant]} ({dominant}={means[dominant]:.4f})")
    print(f"  Means: { {EMOTION_CN[k]: v for k,v in means.items()} }")
    print(f"  Avg intensity: {group['emotional_intensity_mean'].mean():.4f}")
    print(f"  Total posts: {group['total_posts_all'].sum():,}")

# ── 2. Anomaly Events ───────────────────────────
print("\n" + "=" * 60)
print("2. ANOMALY EVENTS (Top 5)")
print("=" * 60)
anomalies = json.load(open(DATA / "anomaly_detection.json", encoding="utf-8"))
anomalies = [a for a in anomalies if a.get("date_week", "") <= HISTORICAL_END_WEEK]
for a in anomalies[:12]:
    print(f"\n{a['date_week']} | {EMOTION_CN.get(a['emotion'], a['emotion'])} | "
          f"z={a['z_score']:+.1f} | {a['deviation_pct']} | {a['severity']}")
    print(f"  Top provinces: ", end="")
    for p in a.get("top_provinces", [])[:3]:
        print(f"{p['province']}({p.get(a['emotion']+'_mean', '?')})", end=" ")
    print()

# ── 3. National Timeline ────────────────────────
print("\n" + "=" * 60)
print("3. NATIONAL TIMELINE KEY POINTS")
print("=" * 60)
tl = pd.read_csv(DATA / "emotion_national_timeline.csv")
tl = tl[tl["date_week"] <= HISTORICAL_END_WEEK].copy()
# Find peaks
for k in EMOTION_KEYS:
    col = f"{k}_mean"
    if col not in tl.columns:
        continue
    peak = tl.loc[tl[col].idxmax()]
    trough = tl.loc[tl[col].idxmin()]
    print(f"{EMOTION_CN[k]:4s}: peak={peak['date_week']}({peak[col]:.4f})  "
          f"trough={trough['date_week']}({trough[col]:.4f})")

# Overall stats
print(f"\nTotal weeks: {len(tl)}")
print(f"Avg joy: {tl['joy_mean'].mean():.4f}  Avg fear: {tl['fear_mean'].mean():.4f}")
print(f"Fear range: {tl['fear_mean'].min():.4f} - {tl['fear_mean'].max():.4f}")

# ── 4. Province Anomaly Contribution ────────────
print("\n" + "=" * 60)
print("4. PROVINCE CONTRIBUTION TO ANOMALIES")
print("=" * 60)
from collections import Counter
prov_counts = Counter()
for a in anomalies:
    for p in a.get("top_provinces", []):
        prov_counts[p["province"]] += 1
for prov, cnt in prov_counts.most_common(15):
    print(f"  {prov}: {cnt} contributions")

# ── 5. Monthly cluster evolution summary ────────
print("\n" + "=" * 60)
print("5. MONTHLY CLUSTER EVOLUTION")
print("=" * 60)
monthly = pd.read_csv(DATA / "monthly_cluster_labels.csv")
print(f"Columns: {monthly.columns.tolist()}")
print(f"Total rows: {len(monthly)}")
province_col = "province" if "province" in monthly.columns else monthly.columns[0]
month_cols = [col for col in monthly.columns if col != province_col]
transitions = monthly.set_index(province_col)[month_cols].apply(list, axis=1)
changed = sum(1 for t in transitions if len(set(v for v in t if v >= 0)) > 1)
print(f"Provinces that changed clusters: {changed}/{len(transitions)}")

print("\nDone.")
