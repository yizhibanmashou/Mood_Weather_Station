"""
Script 16: Local Emotion Inference Engine
Load fine-tuned model and label new text/CSV.
"""
import sys
import json
from pathlib import Path
import importlib.util

import torch
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "emotion_model"

EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device):
    model_mod = _load_local("13_emotion_model")
    from transformers import AutoTokenizer

    model = model_mod.EmotionClassifier(model_name=str(MODEL_DIR), dropout=0.1)
    classifier_path = MODEL_DIR / "classifier.pt"
    if classifier_path.exists():
        model.classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    return model, tokenizer


def _load_local(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def label_texts(model, tokenizer, texts, device, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            list(batch), max_length=128, padding=True,
            truncation=True, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        preds = torch.softmax(model(input_ids, attention_mask), dim=-1).cpu().numpy()
        for p in preds:
            results.append({k: round(float(v), 6) for k, v in zip(EMOTION_KEYS, p)})
    return results


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import argparse

    parser = argparse.ArgumentParser(description="Local Emotion Inference")
    parser.add_argument("--text", type=str, default=None, help="Single text to label")
    parser.add_argument("--input", type=str, default=None, help="Input CSV with content_clean column")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.text and not args.input:
        parser.error("Either --text or --input is required")

    device = get_device()
    print(f"Device: {device}")
    print("Loading model...")
    model, tokenizer = load_model(device)
    print("Model loaded.\n")

    if args.text:
        results = label_texts(model, tokenizer, [args.text], device, args.batch_size)
        print(json.dumps(results[0], ensure_ascii=False, indent=2))
        return

    # Batch CSV mode
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".labeled.csv")
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    if "content_clean" not in df.columns:
        print("[ERROR] CSV missing 'content_clean' column")
        sys.exit(1)

    texts = df["content_clean"].fillna("").astype(str).tolist()
    print(f"Labeling {len(texts):,} texts (batch_size={args.batch_size})...")
    results = label_texts(model, tokenizer, texts, device, args.batch_size)

    scores_df = pd.DataFrame(results)
    for k in EMOTION_KEYS:
        df[k] = scores_df[k]

    df["label_status"] = "ok"
    df["label_model"] = "chinese-roberta-wwm-ext-local"
    df["prompt_version"] = "local-v1"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path} ({len(df):,} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
