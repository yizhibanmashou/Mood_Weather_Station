"""
Script 13e: Evaluate Emotion Model
Test set metrics + SMP2020 benchmark comparison.
"""
import sys
import os
import json
import math
from pathlib import Path
import importlib.util
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
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_DIR = ROOT / "models" / "emotion_model"
TRAINING_DIR = ROOT / "data" / "processed" / "training"
OUTPUT_DIR = ROOT / "analysis" / "evaluation"
SMP2020_DIR = ROOT / "data" / "raw" / "SMP2020_EWECT"

EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
EMOTION_LABELS = ["喜悦", "悲伤", "愤怒", "恐惧", "惊讶", "中性"]

# SMP2020 label mapping: index → our emotion key
SMP2020_MAP = {0: "neutral", 1: "joy", 2: "sadness", 3: "anger", 4: "fear", 5: "surprise"}


def get_device():
    # XPU integrated GPU has limited memory — use CPU for evaluation
    # to avoid OOM with 102M param RoBERTa model
    return torch.device("cpu")


def load_local_mod(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(device):
    model_mod = load_local_mod("13_emotion_model")
    from transformers import AutoTokenizer

    model = model_mod.EmotionClassifier(model_name=str(MODEL_DIR), dropout=0.1)
    classifier_path = MODEL_DIR / "classifier.pt"
    if classifier_path.exists():
        model.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    return model, tokenizer


@torch.no_grad()
def predict_batch(model, tokenizer, texts, device, batch_size=8):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            list(batch_texts), max_length=128, padding=True,
            truncation=True, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        preds = torch.softmax(model(input_ids, attention_mask), dim=-1).cpu()
        all_preds.append(preds)
    return torch.cat(all_preds)


def evaluate_test_set(model, tokenizer, device):
    """Evaluate on our 10% test split."""
    print("\n[1/3] Evaluating on test set...")
    ds_mod = load_local_mod("12_emotion_dataset")
    test_ds = ds_mod.EmotionDataset(
        str(TRAINING_DIR / "test.csv"),
        tokenizer_name=str(MODEL_DIR),
        max_length=128,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    if device.type == "xpu":
        torch.xpu.empty_cache()

    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        preds = torch.softmax(model(input_ids, attention_mask), dim=-1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
        if device.type == "xpu":
            torch.xpu.empty_cache()

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds.argmax(dim=-1) == all_labels.argmax(dim=-1)).float().mean().item()
    mae = (all_preds - all_labels).abs().mean(dim=0)

    print(f"  Test argmax accuracy: {acc:.4f}")
    print(f"  Test per-dim MAE:")
    for k, v in zip(EMOTION_KEYS, mae):
        print(f"    {k:10s}: {v:.4f}")

    return all_preds.numpy(), all_labels.numpy(), acc, mae.numpy()


def evaluate_smp2020(model, tokenizer, device):
    """Evaluate on SMP2020 test sets."""
    print("\n[2/3] Evaluating on SMP2020...")
    results = {}
    for prefix in ["usual", "virus"]:
        path = SMP2020_DIR / f"{prefix}_test_labeled.txt"
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue

        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    records.append({"text": parts[1], "label": int(parts[2])})

        df = pd.DataFrame(records)
        preds = predict_batch(model, tokenizer, df["text"].tolist(), device)
        pred_argmax = preds.argmax(dim=-1).numpy()

        # Map SMP2020 numeric → our key → our index
        y_true = []
        for lbl in df["label"]:
            key = SMP2020_MAP.get(lbl, "neutral")
            y_true.append(EMOTION_KEYS.index(key))
        y_true = np.array(y_true)

        acc = (pred_argmax == y_true).mean()
        print(f"  {prefix} test: {len(df):,} samples, argmax acc = {acc:.4f}")
        results[prefix] = {"samples": len(df), "accuracy": float(acc)}

    return results


def generate_plots(all_preds, all_labels):
    """Scatter plots + confusion matrix."""
    print("\n[3/3] Generating plots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-dim scatter
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, (ax, key) in enumerate(zip(axes.flat, EMOTION_KEYS)):
        ax.scatter(all_labels[:, i], all_preds[:, i], alpha=0.1, s=1)
        ax.plot([0, 1], [0, 1], "r--", linewidth=0.8)
        ax.set_xlabel(f"True {key}")
        ax.set_ylabel(f"Pred {key}")
        ax.set_title(f"{EMOTION_LABELS[i]} ({key})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    plt.tight_layout()
    sp = OUTPUT_DIR / "scatter_per_dim.png"
    fig.savefig(sp, dpi=150)
    plt.close(fig)
    print(f"  Saved: {sp}")

    # Confusion matrix
    pred_argmax = all_preds.argmax(axis=1)
    true_argmax = all_labels.argmax(axis=1)
    cm = np.zeros((6, 6), dtype=int)
    for t, p in zip(true_argmax, pred_argmax):
        cm[t, p] += 1
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", ax=ax,
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    cp = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(cp, dpi=150)
    plt.close(fig)
    print(f"  Saved: {cp}")


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    device = get_device()

    print("=" * 60)
    print("Script 13e: Evaluate Emotion Model")
    print(f"  Device: {device}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(device)
    model.eval()

    # 1. Test set evaluation
    preds, labels, test_acc, test_mae = evaluate_test_set(model, tokenizer, device)

    # 2. SMP2020 benchmark
    smp2020_results = evaluate_smp2020(model, tokenizer, device)

    # 3. Plots
    generate_plots(preds, labels)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Test argmax accuracy:    {test_acc:.4f}")
    print(f"  Test avg MAE:            {test_mae.mean():.4f}")
    for k, v in zip(EMOTION_KEYS, test_mae):
        print(f"    {k:10s} MAE: {v:.4f}")
    for prefix, r in smp2020_results.items():
        print(f"  SMP2020 {prefix}: acc={r['accuracy']:.4f} ({r['samples']:,} samples)")

    # Save summary JSON
    summary = {
        "test_argmax_accuracy": float(test_acc),
        "test_avg_mae": float(test_mae.mean()),
        "test_per_dim_mae": {k: float(v) for k, v in zip(EMOTION_KEYS, test_mae)},
        "smp2020": smp2020_results,
    }
    with open(OUTPUT_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
