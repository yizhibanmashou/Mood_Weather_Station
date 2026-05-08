"""
Script 04b: NLP Keyword Extraction for Anomaly Weeks
Extracts TF-IDF keywords from Weibo text to explain emotion anomalies.
Run after 04_aggregate_emotions.py, before 08_prepare_frontend_assets.py.
"""
import pandas as pd
import numpy as np
import json
import sys
import os
import re
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
TMP_DIR = ROOT / "tmp"

EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
EMOTION_CN = {
    "joy": "喜悦", "sadness": "悲伤", "anger": "愤怒",
    "fear": "恐惧", "surprise": "惊讶", "neutral": "中性"
}

MIN_POSTS_PER_WEEK = 50
TOP_K = 30
SURGE_THRESHOLD = 2.0

POS_KEEP = {"n", "nr", "ns", "nt", "nz", "v", "vn", "a", "an", "i", "l"}

BUILTIN_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "他", "她", "它", "们", "那", "些", "什么", "怎么", "如何", "为什么",
    "因为", "所以", "但是", "虽然", "然而", "而且", "或者", "如果", "可以", "这个",
    "那个", "哪个", "这样", "那样", "怎样", "转发", "微博", "视频", "网页", "链接",
    "展开", "全文", "分享", "图片", "评论", "点赞", "回复", "http", "https", "www", "com",
    "的了", "不是", "知道", "还是", "时候", "没有", "已经", "可能", "应该", "需要",
    "希望", "觉得", "真的", "这个", "那个", "一下", "一些", "一样", "一直", "一定",
    "其实", "然后", "现在", "今天", "昨天", "明天", "大家", "出来", "起来", "下来",
    "只是", "这么", "那么", "多么", "什么", "这里", "那里", "这些", "那些", "这种",
    "那种", "每个", "某个", "任何", "所有", "全部", "之外", "之间", "之前", "之后",
    "以上", "以下", "以来", "以及", "以前", "以后", "以来", "除了", "对于", "关于",
    "通过", "进行", "实现", "认为", "可以", "能够", "可能", "必须", "需要", "应该",
    "而是", "不是", "就是", "也是", "都是", "只是", "只有", "只要", "除非", "无论",
}

WEIBO_NOISE = {
    "转发", "微博", "视频", "网页", "链接", "展开", "全文", "分享", "图片",
    "评论", "点赞", "回复", "http", "https", "com", "cn", "html", "shtml",
    "超话", "话题", "热搜", "头条", "新闻", "资讯", "资讯", "关注", "粉丝",
    "博主", "网友", "评论区", "私信", "消息", "通知", "主页", "个人", "主页",
    "秒拍", "秒拍视频", "酷燃", "微博视频", "微博故事", "微博问答",
    "抽奖", "中奖", "红包", "福利", "活动", "免费", "赠送", "领取",
    "理由", "转发理由", "转发微博", "原图", "查看", "大图", "小图",
}


def load_stopwords():
    """Load stopwords from file or use builtin"""
    stopwords = BUILTIN_STOPWORDS.copy()
    sw_path = ROOT / "data" / "stopwords" / "chinese_stopwords.txt"
    if sw_path.exists():
        with open(sw_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w and not w.startswith("#"):
                    stopwords.add(w)
        return stopwords, "file"
    return stopwords, "builtin"


def detect_text_field(df):
    """Auto-detect text column"""
    for col in ["content_clean", "content_raw", "content"]:
        if col in df.columns:
            return col
    raise ValueError("No text column found (content_clean, content_raw, content)")


def detect_week_field(df):
    """Auto-detect week column"""
    for col in ["date_week"]:
        if col in df.columns:
            return col
    raise ValueError("No week column found (date_week)")


def detect_emotion_fields(df):
    """Detect emotion score columns"""
    return [k for k in EMOTION_KEYS if k in df.columns]


def is_valid_token(word, pos, stopwords):
    """Check if token should be kept"""
    if len(word) < 2:
        return False
    if word in stopwords:
        return False
    if word in WEIBO_NOISE:
        return False
    if re.match(r'^[\d\.]+$', word):
        return False
    if re.match(r'^[a-zA-Z]{1,4}$', word):
        return False
    if re.match(r'^[\W_]+$', word):
        return False
    if pos and pos not in POS_KEEP:
        return False
    return True


def tokenize_text(text, stopwords):
    """Tokenize text with jieba.posseg and filter"""
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = []
    for word, pos in pseg.cut(text):
        word = word.strip()
        if is_valid_token(word, pos, stopwords):
            tokens.append((word, pos))
    return tokens


def build_week_documents(df, text_col, week_col, stopwords):
    """Build per-week tokenized documents"""
    week_docs = {}
    week_tokens = {}
    week_pos = {}

    for week, group in df.groupby(week_col):
        all_tokens = []
        all_pos = {}
        for text in group[text_col].dropna():
            tokens = tokenize_text(str(text), stopwords)
            all_tokens.extend([t[0] for t in tokens])
            for w, p in tokens:
                if w not in all_pos:
                    all_pos[w] = p

        if all_tokens:
            week_docs[week] = " ".join(all_tokens)
            week_tokens[week] = Counter(all_tokens)
            week_pos[week] = all_pos

    return week_docs, week_tokens, week_pos


def compute_tfidf(week_docs, max_features=8000):
    """Compute TF-IDF across weeks"""
    weeks = sorted(week_docs.keys())
    doc_list = [week_docs[w] for w in weeks]

    min_df = 1 if len(weeks) < 20 else 2
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=0.85,
        tokenizer=lambda x: x.split(),
        token_pattern=None,
    )

    tfidf_matrix = vectorizer.fit_transform(doc_list)
    feature_names = vectorizer.get_feature_names_out()

    return weeks, tfidf_matrix, feature_names, vectorizer


def extract_keywords_for_week(week_idx, tfidf_matrix, feature_names, week_tokens, week_pos, global_avg_tfidf, max_k=80):
    """Extract keywords for a specific week (up to max_k)"""
    row = tfidf_matrix[week_idx].toarray().flatten()
    top_indices = row.argsort()[::-1][:max_k * 2]

    keywords = []
    for idx in top_indices:
        if row[idx] <= 0:
            continue
        word = feature_names[idx]
        tf = week_tokens.get(word, 0)
        tfidf = float(row[idx])
        global_tfidf = float(global_avg_tfidf.get(word, 0))
        surge_ratio = tfidf / max(global_tfidf, 1e-6)
        surge = tfidf > 0 and surge_ratio >= SURGE_THRESHOLD

        keywords.append({
            "word": word,
            "tf": int(tf),
            "tfidf": round(tfidf, 4),
            "global_tfidf": round(global_tfidf, 4),
            "surge": surge,
            "surge_ratio": round(surge_ratio, 2),
            "pos": week_pos.get(word, ""),
        })

        if len(keywords) >= max_k:
            break

    return keywords


def compute_emotion_dominant(df, week_col, emotion_fields):
    """Get dominant emotion per week with full score breakdown"""
    week_emotions = {}
    for week, group in df.groupby(week_col):
        means = {k: float(group[k].mean()) for k in emotion_fields if k in group.columns}
        if means:
            sorted_emotions = sorted(means.items(), key=lambda x: x[1], reverse=True)
            week_emotions[week] = {
                "dominant": sorted_emotions[0][0],
                "top": [e[0] for e in sorted_emotions[:2]],
                "means": means,
            }
    return week_emotions


EMOTION_ATTR_THRESHOLD = 0.05


def build_emotion_keywords(nlp_weeks, week_emotions):
    """Aggregate keywords by emotion — assign keywords to all emotions above threshold"""
    emotion_kw = defaultdict(list)
    emotion_word_tfidf = defaultdict(lambda: defaultdict(list))

    for week_key, week_data in nlp_weeks.items():
        if week_data.get("status") != "ok":
            continue
        emo_info = week_emotions.get(week_key, {})
        # Assign keywords to all emotions with mean >= threshold
        means = emo_info.get("means", {})
        active_emotions = [k for k, v in means.items() if v >= EMOTION_ATTR_THRESHOLD]
        if not active_emotions:
            active_emotions = [emo_info.get("dominant", "neutral")]

        for kw in week_data.get("keywords", []):
            word = kw["word"]
            tfidf = kw["tfidf"]
            for emotion in active_emotions:
                emotion_word_tfidf[emotion][word].append((tfidf, week_key))

    for emotion, words in emotion_word_tfidf.items():
        word_stats = []
        for word, tfidf_weeks in words.items():
            avg_tfidf = np.mean([t[0] for t in tfidf_weeks])
            peak_item = max(tfidf_weeks, key=lambda x: x[0])
            peak_week = peak_item[1]
            peak_tfidf = peak_item[0]
            word_stats.append({
                "word": word,
                "avg_tfidf": round(float(avg_tfidf), 4),
                "peak_week": peak_week,
                "peak_tfidf": round(float(peak_tfidf), 4),
            })
        word_stats.sort(key=lambda x: x["avg_tfidf"], reverse=True)
        emotion_kw[emotion] = word_stats[:30]

    # Ensure all emotion keys exist (even if empty)
    for emo in EMOTION_KEYS:
        if emo not in emotion_kw:
            emotion_kw[emo] = []

    return dict(emotion_kw)


def build_global_vocabulary(tfidf_matrix, feature_names, week_tokens_list):
    """Build global vocabulary stats"""
    global_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    vocab = []
    for idx, word in enumerate(feature_names):
        total_tf = sum(wt.get(word, 0) for wt in week_tokens_list)
        avg_tfidf = float(global_tfidf[idx])

        peak_week_idx = tfidf_matrix[:, idx].toarray().flatten().argmax()
        peak_tfidf = float(tfidf_matrix[peak_week_idx, idx])

        vocab.append({
            "word": word,
            "total_tf": int(total_tf),
            "avg_tfidf": round(avg_tfidf, 4),
            "peak_week_idx": int(peak_week_idx),
            "peak_tfidf": round(peak_tfidf, 4),
        })

    vocab.sort(key=lambda x: x["avg_tfidf"], reverse=True)
    return vocab[:5000]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import argparse

    parser = argparse.ArgumentParser(description="NLP Keyword Extraction")
    parser.add_argument("--input", type=str, default=None,
                        help="Input labeled CSV path")
    args = parser.parse_args()

    print("=" * 60)
    print("Script 04b: NLP Keyword Extraction")
    print("=" * 60)

    # Load stopwords
    stopwords, sw_source = load_stopwords()
    print(f"  Stopwords: {len(stopwords)} ({sw_source})")

    # Find input file
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = ROOT / input_path
    else:
        input_candidates = [
            PROCESSED_DIR / "labeled_dataset_merged_week_cap60.csv",
            PROCESSED_DIR / "labeled_dataset_merged_cap30.csv",
            PROCESSED_DIR / "labeled_dataset.csv",
        ]
        input_path = None
        for p in input_candidates:
            if p.exists():
                input_path = p
                break
    if not input_path or not input_path.exists():
        print("[ERROR] No labeled dataset found.")
        sys.exit(1)

    print(f"  Input: {input_path}")

    # Load data
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"  Posts: {len(df):,}")

    # Detect fields
    text_col = detect_text_field(df)
    week_col = detect_week_field(df)
    emotion_fields = detect_emotion_fields(df)
    print(f"  Text field: {text_col}")
    print(f"  Week field: {week_col}")
    print(f"  Emotion fields: {emotion_fields}")

    # Load anomaly data
    anomaly_path = PROCESSED_DIR / "anomaly_detection.json"
    anomaly_weeks = set()
    if anomaly_path.exists():
        with open(anomaly_path, "r", encoding="utf-8") as f:
            anomalies = json.load(f)
        anomaly_weeks = {a["date_week"] for a in anomalies}
        print(f"  Anomaly weeks: {len(anomaly_weeks)}")
    else:
        print("  [WARN] No anomaly_detection.json found, will process all weeks")

    # Build week documents
    print("\n[1/4] Tokenizing texts...")
    week_docs, week_tokens, week_pos_map = build_week_documents(df, text_col, week_col, stopwords)
    print(f"  Weeks with text: {len(week_docs)}")

    # Compute TF-IDF
    print("\n[2/4] Computing TF-IDF...")
    weeks_order, tfidf_matrix, feature_names, vectorizer = compute_tfidf(week_docs)
    print(f"  Vocabulary size: {len(feature_names)}")

    # Global average TF-IDF
    global_avg_tfidf = {}
    global_avg = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    for idx, word in enumerate(feature_names):
        global_avg_tfidf[word] = float(global_avg[idx])

    # Get dominant emotion per week
    week_emotions = compute_emotion_dominant(df, week_col, emotion_fields)

    # Count posts per week
    week_post_counts = df.groupby(week_col).size().to_dict()

    # Extract keywords for all weeks (anomaly weeks get priority in frontend)
    print("\n[3/4] Extracting keywords for all weeks...")
    nlp_weeks = {}
    success_count = 0
    insufficient_count = 0

    target_weeks = set(weeks_order)

    for week_key in sorted(target_weeks):
        if week_key not in weeks_order:
            nlp_weeks[week_key] = {"status": "no_data", "total_posts": 0}
            continue

        week_idx = weeks_order.index(week_key)
        total_posts = week_post_counts.get(week_key, 0)

        if total_posts < MIN_POSTS_PER_WEEK:
            nlp_weeks[week_key] = {
                "status": "insufficient_data",
                "total_posts": int(total_posts),
                "min_required": MIN_POSTS_PER_WEEK,
            }
            insufficient_count += 1
            continue

        all_keywords = extract_keywords_for_week(
            week_idx, tfidf_matrix, feature_names,
            week_tokens.get(week_key, Counter()),
            week_pos_map.get(week_key, {}),
            global_avg_tfidf,
            max_k=80,
        )

        # Group keywords by different criteria
        top_keywords = sorted(all_keywords, key=lambda x: x["tfidf"], reverse=True)[:TOP_K]
        frequent_keywords = sorted(all_keywords, key=lambda x: x["tf"], reverse=True)[:TOP_K]
        surge_keywords = sorted(
            [kw for kw in all_keywords if kw["surge"]],
            key=lambda x: x["surge_ratio"],
            reverse=True,
        )[:TOP_K]

        emo_info = week_emotions.get(week_key, {})
        means = emo_info.get("means", {})
        top_emotions = sorted(means, key=means.get, reverse=True)[:3] if means else []
        nlp_weeks[week_key] = {
            "status": "ok",
            "total_posts": int(total_posts),
            "dominant_emotion": emo_info.get("dominant", "neutral"),
            "top_emotions": top_emotions,
            "top_keywords": top_keywords,
            "frequent_keywords": frequent_keywords,
            "surge_keywords": surge_keywords,
            "keywords": all_keywords,  # full set, up to 80
        }
        success_count += 1

    print(f"  Success: {success_count}, Insufficient: {insufficient_count}")

    # Build emotion keywords
    print("\n[4/4] Building emotion keywords and global vocabulary...")
    emotion_keywords = build_emotion_keywords(nlp_weeks, week_emotions)

    # Build global vocabulary
    week_tokens_list = [week_tokens.get(w, Counter()) for w in weeks_order]
    global_vocab = build_global_vocabulary(tfidf_matrix, feature_names, week_tokens_list)

    # Export outputs
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. nlp_keywords_by_week.json
    output_weeks = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "source_file": str(input_path.name),
            "min_posts_per_week": MIN_POSTS_PER_WEEK,
            "top_k": TOP_K,
            "method": "jieba_posseg_tfidf",
            "stopwords_source": sw_source,
        },
        "weeks": nlp_weeks,
    }
    out1 = PROCESSED_DIR / "nlp_keywords_by_week.json"
    with open(out1, "w", encoding="utf-8") as f:
        json.dump(output_weeks, f, ensure_ascii=False, indent=2)
    print(f"  [OK] {out1}")

    # 2. nlp_emotion_keywords.json
    output_emotions = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "method": "emotion_weighted_keyword_tfidf",
        },
        "emotions": emotion_keywords,
    }
    out2 = PROCESSED_DIR / "nlp_emotion_keywords.json"
    with open(out2, "w", encoding="utf-8") as f:
        json.dump(output_emotions, f, ensure_ascii=False, indent=2)
    print(f"  [OK] {out2}")

    # 3. nlp_global_vocabulary.json
    output_vocab = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "total_terms": len(global_vocab),
        },
        "vocabulary": global_vocab,
    }
    out3 = PROCESSED_DIR / "nlp_global_vocabulary.json"
    with open(out3, "w", encoding="utf-8") as f:
        json.dump(output_vocab, f, ensure_ascii=False, indent=2)
    print(f"  [OK] {out3}")

    # 4. Review CSV
    review_rows = []
    for week_key, week_data in sorted(nlp_weeks.items()):
        if week_data.get("status") != "ok":
            continue
        for i, kw in enumerate(week_data.get("keywords", [])[:50]):
            review_rows.append({
                "week": week_key,
                "rank": i + 1,
                "word": kw["word"],
                "tf": kw["tf"],
                "tfidf": kw["tfidf"],
                "global_tfidf": kw["global_tfidf"],
                "surge": kw["surge"],
                "surge_ratio": kw["surge_ratio"],
                "pos": kw["pos"],
            })
    review_df = pd.DataFrame(review_rows)
    review_path = TMP_DIR / "04b_nlp_keyword_review.csv"
    review_df.to_csv(review_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] {review_path}")

    # 5. Summary report
    total_weeks = len(weeks_order)
    anomaly_count = len(anomaly_weeks)
    top_noise = []
    all_words = Counter()
    for wt in week_tokens.values():
        all_words.update(wt)
    for w, c in all_words.most_common(20):
        if w in WEIBO_NOISE or w in BUILTIN_STOPWORDS:
            top_noise.append(f"{w}({c})")

    summary_path = TMP_DIR / "04b_nlp_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# NLP Keyword Extraction Summary\n\n")
        f.write(f"- Generated: {datetime.now().isoformat()}\n")
        f.write(f"- Source: {input_path.name}\n")
        f.write(f"- Total weeks: {total_weeks}\n")
        f.write(f"- Anomaly weeks: {anomaly_count}\n")
        f.write(f"- Success (ok): {success_count}\n")
        f.write(f"- Insufficient data: {insufficient_count}\n")
        f.write(f"- Vocabulary size: {len(feature_names)}\n")
        f.write(f"- Stopwords: {len(stopwords)} ({sw_source})\n")
        f.write(f"- Top noise words filtered: {', '.join(top_noise[:10])}\n")
    print(f"  [OK] {summary_path}")

    print(f"\nDone. Check {PROCESSED_DIR} for outputs.")


if __name__ == "__main__":
    main()
