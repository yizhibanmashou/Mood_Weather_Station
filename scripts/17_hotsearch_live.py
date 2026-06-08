"""
Script 17: UAPIS Hot Search → Local Emotion Inference → Trend Snapshot

Compliant realtime pipeline: fetches ONLY hot search topic titles from UAPIS API
(public aggregated data), runs local model inference for 6-dim emotion scores,
and exports a JSON snapshot for frontend consumption.

No user post content is scraped. No s.weibo.com access. No login cookies.
"""
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parent.parent

# Load .env before resolving local runtime paths.
_env_path = ROOT / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# Ensure Intel XPU runtime DLLs are on PATH (needed for emotion_xpu env)
_xpu_base = Path(os.getenv("EMOTION_XPU_ENV", r"D:\anaconda\envs\emotion_xpu"))
for _sub in ["", "Library\\bin", "Scripts"]:
    _p = str(_xpu_base / _sub) if _sub else str(_xpu_base)
    if _p not in os.environ["PATH"]:
        os.environ["PATH"] = _p + ";" + os.environ["PATH"]

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import requests
import torch
import numpy as np

MODEL_DIR = ROOT / "models" / "emotion_model"
OUTPUT_DIR = ROOT / "app" / "public" / "data" / "realtime"

EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
EMOTION_CN = ["喜悦", "悲伤", "愤怒", "恐惧", "惊讶", "中性"]
CST = timezone(timedelta(hours=8))

# ── Config ────────────────────────────────────────────────────
UAPIS_API_KEY = os.getenv("UAPIS_API_KEY", "")
UAPIS_HOTBOARD_URL = "https://uapis.cn/api/v1/misc/hotboard"
FETCH_INTERVAL = 300  # seconds between fetches (UAPIS updates ~every 5 min)
TOP_N = 50            # top N hot search items to analyze


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model():
    """Load fine-tuned emotion model from models/emotion_model/."""
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module
    model_mod = import_module("13_emotion_model")

    model = model_mod.EmotionClassifier(model_name=str(MODEL_DIR), dropout=0.1)
    classifier_path = MODEL_DIR / "classifier.pt"
    if classifier_path.exists():
        model.classifier.load_state_dict(
            torch.load(classifier_path, map_location="cpu", weights_only=True)
        )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    return model, tokenizer


@torch.no_grad()
def infer_emotions(model, tokenizer, texts, device, batch_size=32):
    """Batch inference: list of strings → list of {emotion: score} dicts."""
    model.to(device)
    model.eval()
    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            list(batch), max_length=128, padding=True,
            truncation=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        preds = torch.softmax(model(input_ids, attention_mask), dim=-1).cpu().numpy()
        for p in preds:
            all_preds.append({k: round(float(v), 6) for k, v in zip(EMOTION_KEYS, p)})

    return all_preds


def fetch_hotsearch():
    """Fetch Weibo hot search topics from UAPIS.

    Returns list of {"title": str, "hot_value": int, "rank": int}.
    """
    if not UAPIS_API_KEY:
        print("[WARN] UAPIS_API_KEY not set. Using demo data.")
        return _demo_topics()

    try:
        resp = requests.get(
            UAPIS_HOTBOARD_URL,
            params={"key": UAPIS_API_KEY, "type": "weibo"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        topics = data.get("list", [])
        if not topics:
            print(f"  UAPIS returned empty list: {data}")
            return _demo_topics()

        results = []
        for i, t in enumerate(topics):
            title = t.get("title", "").strip()
            hot_value = t.get("hot_value", 0) or t.get("hot", 0)
            if title:
                results.append({
                    "title": title,
                    "hot_value": int(hot_value) if hot_value else 0,
                    "rank": i + 1,
                })

        update_time = data.get("update_time", "N/A")
        print(f"  Fetched {len(results)} topics from UAPIS (updated: {update_time})")
        return results

    except Exception as e:
        print(f"  UAPIS request failed: {e}")
        return _demo_topics()


def _demo_topics():
    """Fallback demo topics when API is unavailable."""
    demos = [
        ("2026年高考今日开考", 8500000),
        ("全国多地迎来高温天气", 7200000),
        ("气象台发布暴雨橙色预警", 6800000),
        ("端午节假期首日全国旅游市场火爆", 6500000),
        ("嫦娥七号探测器成功发射", 6200000),
        ("教育部发布2026年高考作文题目", 5900000),
        ("华南地区持续强降雨", 5600000),
        ("中国女排世联赛夺冠", 5300000),
        ("全国碳交易市场正式启动三周年", 5000000),
        ("南方多地启动防汛应急响应", 4800000),
    ]
    return [
        {"title": t, "hot_value": v, "rank": i + 1}
        for i, (t, v) in enumerate(demos)
    ]


def build_snapshot(topics, emotions):
    """Build the full snapshot JSON structure."""
    now = datetime.now(CST)

    # Per-topic emotion data
    items = []
    for t, e in zip(topics, emotions):
        dominant = max(e, key=e.get)
        # Emotional entropy: 0 = all mass in one emotion, higher = more distributed
        probs = np.array([e[k] for k in EMOTION_KEYS], dtype=np.float64)
        probs = np.clip(probs, 1e-12, 1.0)
        entropy = -np.sum(probs * np.log(probs)) / np.log(len(EMOTION_KEYS))
        # Emotional intensity: 1 - neutral_score (higher = more emotional)
        intensity = round(1.0 - e.get("neutral", 1.0), 4)
        items.append({
            "rank": t["rank"],
            "title": t["title"],
            "hot_value": t["hot_value"],
            "emotion": e,
            "dominant_emotion": dominant,
            "dominant_emotion_cn": EMOTION_CN[EMOTION_KEYS.index(dominant)],
            "dominant_score": e[dominant],
            "emotional_intensity": intensity,
            "emotional_entropy": round(float(entropy), 4),
        })
    # Sort by emotional intensity descending for display
    items.sort(key=lambda x: x["emotional_intensity"], reverse=True)

    # Aggregate emotion distribution (weighted by hot_value)
    total_hot = sum(t["hot_value"] for t in topics) or 1
    agg = {k: 0.0 for k in EMOTION_KEYS}
    for t, e in zip(topics, emotions):
        weight = t["hot_value"] / total_hot
        for k in EMOTION_KEYS:
            agg[k] += e[k] * weight

    # Dominant emotion in aggregate
    agg_dominant = max(agg, key=agg.get)

    snapshot = {
        "fetch_time": now.isoformat(),
        "fetch_time_str": now.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "UAPIS Weibo Hot Search",
        "total_topics": len(items),
        "aggregate_emotion": {k: round(v, 4) for k, v in agg.items()},
        "aggregate_dominant": agg_dominant,
        "aggregate_dominant_cn": EMOTION_CN[EMOTION_KEYS.index(agg_dominant)],
        "topics": items,
    }
    return snapshot


def save_snapshot(snapshot):
    """Save snapshot JSON and maintain history log."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Latest snapshot (overwritten each fetch)
    latest_path = OUTPUT_DIR / "hotsearch_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {latest_path}")

    # Append to history log (one line per snapshot)
    history_path = OUTPUT_DIR / "hotsearch_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    print(f"  Appended: {history_path}")

    return latest_path


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("Script 17: UAPIS Hot Search → Emotion Snapshot")
    print(f"  Time: {datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Load model
    print("\n[1/4] Loading emotion model...")
    device = get_device()
    print(f"  Device: {device}")
    model, tokenizer = load_model()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # 2. Fetch hot search
    print("\n[2/4] Fetching UAPIS hot search...")
    topics = fetch_hotsearch()
    topics = topics[:TOP_N]
    print(f"  Top {len(topics)} topics:")
    for t in topics[:5]:
        print(f"    #{t['rank']} {t['title']} (hot: {t['hot_value']:,})")

    # 3. Infer emotions
    print(f"\n[3/4] Inferring emotions ({len(topics)} titles)...")
    titles = [t["title"] for t in topics]
    emotions = infer_emotions(model, tokenizer, titles, device)

    # Print most emotional topics
    items_sorted = sorted(
        [{"t": t, "e": e} for t, e in zip(topics, emotions)],
        key=lambda x: 1.0 - x["e"].get("neutral", 1.0), reverse=True,
    )
    print("  Top 5 most emotional topics:")
    for item in items_sorted[:5]:
        t, e = item["t"], item["e"]
        intensity = 1.0 - e.get("neutral", 1.0)
        dominant = max(e, key=e.get)
        print(f"  #{t['rank']} {t['title']}  (intensity={intensity:.3f})")
        print(f"       → {EMOTION_CN[EMOTION_KEYS.index(dominant)]}({e[dominant]:.2f})  "
              f"joy={e['joy']:.2f} sad={e['sadness']:.2f} ang={e['anger']:.2f} "
              f"fear={e['fear']:.2f} sur={e['surprise']:.2f} neu={e['neutral']:.2f}")

    # 4. Build & save snapshot
    print("\n[4/4] Building snapshot...")
    snapshot = build_snapshot(topics, emotions)
    agg = snapshot["aggregate_emotion"]
    print(f"  Aggregate emotion: {snapshot['aggregate_dominant_cn']} (dominant)")
    print(f"    joy={agg['joy']:.3f}  sadness={agg['sadness']:.3f}  "
          f"anger={agg['anger']:.3f}  fear={agg['fear']:.3f}  "
          f"surprise={agg['surprise']:.3f}  neutral={agg['neutral']:.3f}")

    path = save_snapshot(snapshot)

    # Domain shift note
    neutral_pct = agg.get("neutral", 0) * 100
    if neutral_pct > 80:
        print(f"\n  ⚠ Note: {neutral_pct:.0f}% of aggregate emotion is 'neutral'.")
        print(f"    This is expected: hot search titles are news headlines (objective/factual),")
        print(f"    while the model was trained on personal Weibo posts (emotional/expressive).")
        print(f"    This domain shift means titles show lower emotional signal than posts.")

    print(f"\n{'='*60}")
    print(f"Done. Snapshot saved to {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
