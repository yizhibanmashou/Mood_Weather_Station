"""
Script 18: Synthetic Weibo Data Generator
Legal data acquisition via:
  1. UAPIS hot search API → trending topics
  2. DeepSeek API → generates realistic Weibo-style posts with province/time
  3. Local emotion model → scores 6-dim emotions
  4. Output matches existing labeled dataset schema

Fallback-only approach: all other legal data sources exhausted.
See docs/DATA_ACQUISITION_REPORT.md for full exploration log.
"""
import sys, os, json, time, re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Ensure Intel XPU runtime DLLs are on PATH
_xpu_base = Path(os.getenv("EMOTION_XPU_ENV", r"D:\anaconda\envs\emotion_xpu"))
for _sub in ["", "Library\\bin", "Scripts"]:
    _p = str(_xpu_base / _sub) if _sub else str(_xpu_base)
    if _p not in os.environ["PATH"]:
        os.environ["PATH"] = _p + ";" + os.environ["PATH"]

os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "synthetic"
TMP_DIR = ROOT / "tmp"

CST = timezone(timedelta(hours=8))

# ── Config ────────────────────────────────────────────────────
TOPICS_TO_USE = 10       # top N hot topics to generate from
POSTS_PER_TOPIC = 20     # posts per topic (total = TOPICS_TO_USE * POSTS_PER_TOPIC)
BATCH_SIZE = 10          # DeepSeek API batch size
DRY_RUN = False          # if True, skip API calls, use demo data

EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
VALID_PROVINCES = [
    "北京", "天津", "上海", "重庆",
    "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "广西", "海南",
    "四川", "贵州", "云南", "西藏",
    "陕西", "甘肃", "青海", "宁夏", "新疆",
    "内蒙古", "香港", "澳门", "台湾",
]

# ── UAPIS ──────────────────────────────────────────────────────
UAPIS_API_KEY = os.getenv("UAPIS_API_KEY", "")
UAPIS_HOTBOARD_URL = "https://uapis.cn/api/v1/misc/hotboard"

# ── DeepSeek ───────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()

SYSTEM_PROMPT = """你是一个微博内容模拟器。根据给定的话题标题，生成符合以下要求的微博帖子：

1. 每条帖子必须看起来像真实用户发布的微博（口语化、带个人观点或情绪）
2. 必须包含省份信息（随机从34个省级行政区中选择）
3. 时间设为当前时间附近
4. 每条字数30-150字
5. 不要使用特定用户ID，使用占位符 user_xxxxx

输出JSON数组，每个元素格式：
{
  "content_clean": "微博正文",
  "province": "省份名",
  "gender": "m/f",
  "city": "城市名（可选）"
}"""


def fetch_hot_topics(dry_run=False):
    """Fetch current Weibo hot search topics from UAPIS."""
    if not UAPIS_API_KEY or dry_run:
        print("  [DEMO] Using demo topics")
        return [
            {"title": "2026年高考今日开考", "hot_value": 8500000},
            {"title": "全国多地迎来高温天气", "hot_value": 7200000},
            {"title": "气象台发布暴雨橙色预警", "hot_value": 6800000},
            {"title": "端午节假期首日全国旅游市场火爆", "hot_value": 6500000},
            {"title": "嫦娥七号探测器成功发射", "hot_value": 6200000},
            {"title": "教育部发布2026年高考作文题目", "hot_value": 5900000},
            {"title": "华南地区持续强降雨", "hot_value": 5600000},
            {"title": "中国女排世联赛夺冠", "hot_value": 5300000},
            {"title": "全国碳交易市场正式启动三周年", "hot_value": 5000000},
            {"title": "南方多地启动防汛应急响应", "hot_value": 4800000},
        ]

    import requests
    try:
        resp = requests.get(
            UAPIS_HOTBOARD_URL,
            params={"key": UAPIS_API_KEY, "type": "weibo"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        topics = data.get("list", [])
        results = []
        for i, t in enumerate(topics):
            title = t.get("title", "").strip()
            hot_value = t.get("hot_value", 0) or t.get("hot", 0)
            if title:
                results.append({"title": title, "hot_value": int(hot_value) if hot_value else 0})
        print(f"  Fetched {len(results)} topics")
        return results[:TOPICS_TO_USE]
    except Exception as e:
        print(f"  [WARN] UAPIS failed: {e}. Using demo.")
        return fetch_hot_topics(dry_run=True)  # fallback to demo


def generate_posts_for_topic(topic_title, n=POSTS_PER_TOPIC, dry_run=False):
    """Call DeepSeek API to generate n Weibo-style posts for a topic."""
    if dry_run:
        return _demo_posts(topic_title, n)

    from openai import OpenAI

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    user_prompt = f"""话题标题：{topic_title}

请生成{n}条不同风格、不同省份用户发布的微博帖子。
每条帖子应该表达对这条热搜的个人看法、感受或评论。
省份要多样化，覆盖不同地区。
"""

    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content.strip()
        # Remove markdown code fences if present
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        # Try to extract the array from the response
        data = json.loads(text)
        if isinstance(data, dict):
            # Find the array in the response
            for key in data:
                if isinstance(data[key], list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            data = [data]
        print(f"  Generated {len(data)} posts for '{topic_title[:20]}'")
        return data[:n]
    except Exception as e:
        print(f"  [WARN] DeepSeek generation failed for '{topic_title[:20]}': {e}")
        return _demo_posts(topic_title, n)


def _demo_posts(topic_title, n):
    """Demo posts when API unavailable."""
    import random
    random.seed(hash(topic_title) % (2**31))
    demos = [
        f"看到{topic_title}的热搜，真的觉得#感慨# 这个世界变化太快了",
        f"今天刷微博看到{topic_title}，和我朋友刚聊过这个话题！",
        f"[思考] {topic_title}，大家怎么看？我觉得还需要更多信息才能判断",
        f"#daily# {topic_title} 这个热搜让我想到很多，希望后续有更多进展",
        f"朋友们都在讨论{topic_title}，我也来说两句……其实没那么简单",
    ]
    results = []
    for i in range(min(n, len(demos))):
        prov = random.choice(VALID_PROVINCES)
        results.append({
            "content_clean": demos[i],
            "province": prov,
            "gender": random.choice(["m", "f"]),
        })
    return results


def score_with_local_model(posts, model=None, tokenizer=None, device=None):
    """Run local emotion model on generated posts, returning emotion scores.
    Pass in model+tokenizer to avoid reloading (recommended for multiple calls)."""
    import torch

    # Infer device from model if not explicitly provided
    if device is None and model is not None:
        device = next(model.parameters()).device

    if model is None or tokenizer is None:
        # Load model once
        import importlib.util
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location("emotion_model",
            ROOT / "scripts" / "13_emotion_model.py")
        model_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_mod)
        from transformers import AutoTokenizer

        MODEL_DIR = ROOT / "models" / "emotion_model"

        if device is None:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                device = torch.device("xpu")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        model = model_mod.EmotionClassifier(model_name=str(MODEL_DIR), dropout=0.1)
        classifier_path = MODEL_DIR / "classifier.pt"
        if classifier_path.exists():
            model.classifier.load_state_dict(
                torch.load(classifier_path, map_location=device, weights_only=True)
            )
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    texts = [p["content_clean"] for p in posts]
    results = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                list(batch_texts), max_length=128, padding=True,
                truncation=True, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            probs = torch.softmax(model(input_ids, attention_mask), dim=-1)
            for j, p in enumerate(probs.cpu().numpy()):
                results.append({k: round(float(v), 6) for k, v in zip(EMOTION_KEYS, p)})

    return results


def build_dataset_rows(topics, posts_batch, emotions_batch, spread_days=7):
    """Build rows matching the existing labeled dataset schema.
    spread_days: distribute posts across this many days (default: 7 = 1 week)."""
    import random
    base_time = datetime.now(CST)
    rows = []
    post_idx = 0
    total_posts = sum(len(p) for p in posts_batch)

    for topic, posts, emotions in zip(topics, posts_batch, emotions_batch):
        for post, emo in zip(posts, emotions):
            post_idx += 1
            # Distribute timestamp across spread_days, weighted by post index
            offset_hours = (post_idx / max(total_posts, 1)) * spread_days * 24
            offset_hours += random.uniform(-2, 2)  # jitter
            post_time = base_time - timedelta(hours=offset_hours)
            row = {
                "post_id": f"syn_{base_time.strftime('%y%m%d')}_{post_idx:05d}",
                "user_id": f"syn_user_{post_idx % 1000:04d}",
                "created_at": post_time.strftime("%Y-%m-%d %H:%M:%S"),
                "date_week": post_time.strftime("%Y-W%V"),
                "date_month": post_time.strftime("%Y-%m"),
                "province": post.get("province", "北京"),
                "city": post.get("city", ""),
                "gender": post.get("gender", ""),
                "content_clean": post["content_clean"],
                "content_raw": post["content_clean"],  # same as clean for synthetic
                "word_count": len(post["content_clean"]),
                "joy": emo["joy"],
                "sadness": emo["sadness"],
                "anger": emo["anger"],
                "fear": emo["fear"],
                "surprise": emo["surprise"],
                "neutral": emo["neutral"],
                "label_status": "ok",
                "label_model": "chinese-roberta-wwm-ext-local",
                "prompt_version": "synthetic-v1",
                "source_topic": topic["title"],
                "source_hot_value": topic["hot_value"],
            }
            rows.append(row)

    return rows


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic Weibo Data Generator")
    parser.add_argument("--topics", type=int, default=TOPICS_TO_USE,
                        help=f"Number of hot topics to use (default: {TOPICS_TO_USE})")
    parser.add_argument("--posts-per-topic", type=int, default=POSTS_PER_TOPIC,
                        help=f"Posts per topic (default: {POSTS_PER_TOPIC})")
    parser.add_argument("--dry-run", action="store_true", help="Use demo data, skip API calls")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: data/synthetic/synthetic_YYYYMMDD_HHMM.csv)")
    args = parser.parse_args()

    topics_to_use = args.topics
    posts_per_topic = args.posts_per_topic
    dry_run = args.dry_run

    total_target = topics_to_use * posts_per_topic
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Script 18: Synthetic Weibo Data Generator")
    print(f"  Topics:       {topics_to_use}")
    print(f"  Posts/topic:  {posts_per_topic}")
    print(f"  Total target: {total_target}")
    print(f"  Mode:         {'DRY RUN (demo)' if dry_run else 'LIVE'}")
    print("=" * 60)

    # Step 1: Fetch hot topics
    print("\n[1/4] Fetching hot topics from UAPIS...")
    topics = fetch_hot_topics(dry_run=dry_run)
    topics = topics[:topics_to_use]
    print(f"  Top {len(topics)} topics:")
    for t in topics:
        print(f"    · {t['title']} (hot: {t.get('hot_value', 0):,})")

    # Step 2: Generate posts via DeepSeek
    print(f"\n[2/4] Generating {posts_per_topic} posts per topic via DeepSeek...")
    all_posts = []
    for i, topic in enumerate(topics):
        print(f"  [{i+1}/{len(topics)}] '{topic['title']}'")
        posts = generate_posts_for_topic(topic["title"], n=posts_per_topic, dry_run=dry_run)
        all_posts.append(posts)

    total_generated = sum(len(p) for p in all_posts)
    print(f"  Total generated: {total_generated} posts")

    if total_generated == 0:
        print("[ERROR] No posts generated. Aborting.")
        sys.exit(1)

    # Step 3: Score with local model (load once, score all)
    print(f"\n[3/4] Loading emotion model and scoring {total_generated} posts...")
    import torch
    import importlib.util
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / "scripts"))
    _spec = importlib.util.spec_from_file_location("emotion_model",
        ROOT / "scripts" / "13_emotion_model.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    MODEL_DIR = ROOT / "models" / "emotion_model"
    _device = torch.device("xpu") if (hasattr(torch, "xpu") and torch.xpu.is_available()) else torch.device("cpu")
    _model = _mod.EmotionClassifier(model_name=str(MODEL_DIR), dropout=0.1)
    _cp = MODEL_DIR / "classifier.pt"
    if _cp.exists():
        _model.classifier.load_state_dict(torch.load(_cp, map_location=_device, weights_only=True))
    _model.to(_device)
    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    all_emotions = []
    for posts in all_posts:
        emotions = score_with_local_model(posts, model=_model, tokenizer=_tokenizer)
        all_emotions.append(emotions)
        print(f"  Scored batch: {len(emotions)} posts")

    # Quick stats
    flat_emotions = [e for batch in all_emotions for e in batch]
    if flat_emotions:
        avg_emo = {k: sum(e[k] for e in flat_emotions) / len(flat_emotions)
                   for k in EMOTION_KEYS}
        print(f"  Avg emotion distribution:")
        for k, v in avg_emo.items():
            print(f"    {k:10s}: {v:.4f}")
        dominant = max(avg_emo, key=avg_emo.get)
        print(f"  Dominant: {dominant} ({avg_emo[dominant]:.2%})")

    # Step 4: Build dataset & export
    print(f"\n[4/4] Building dataset and exporting...")
    rows = build_dataset_rows(topics, all_posts, all_emotions, spread_days=10)
    print(f"  Built {len(rows)} rows matching existing schema")

    # Save CSV
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"synthetic_{timestamp}.csv"
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    import pandas as pd
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  CSV saved:  {output_path} ({len(df)} rows, {output_path.stat().st_size / 1024:.1f} KB)")

    # Save JSON (for frontend)
    json_path = OUTPUT_DIR / f"synthetic_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(CST).isoformat(),
            "config": {"topics": TOPICS_TO_USE, "posts_per_topic": POSTS_PER_TOPIC},
            "sources": [t["title"] for t in topics],
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved: {json_path}")

    # Print sample
    print(f"\n  Sample rows:")
    for row in rows[:3]:
        print(f"    [{row['province']}] {row['content_clean'][:60]}...")
        emo_str = "  ".join([f"{k}={row[k]:.3f}" for k in EMOTION_KEYS])
        print(f"    → {emo_str}")

    print(f"\n{'='*60}")
    print(f"Done! {len(rows)} synthetic posts generated.")
    print(f"Output: {output_path}")
    print(f"Next: merge this with existing data via Script 02e or run_pipeline")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
