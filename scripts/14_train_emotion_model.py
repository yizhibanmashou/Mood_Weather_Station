"""
Script 14: Train Emotion Model
Fine-tune chinese-roberta-wwm-ext on DeepSeek-labeled data.
KL divergence loss, BF16 AMP, XPU/CUDA/CPU auto-select.
"""
import sys
import os
import json
import math
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Ensure Intel XPU runtime DLLs are on PATH
_xpu_base = Path(os.getenv("EMOTION_XPU_ENV", r"D:\anaconda\envs\emotion_xpu"))
for _sub in ["", "Library\\bin", "Scripts"]:
    _p = str(_xpu_base / _sub) if _sub else str(_xpu_base)
    if _p not in os.environ["PATH"]:
        os.environ["PATH"] = _p + ";" + os.environ["PATH"]

# Avoid HuggingFace 403 on discussions endpoint
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import functools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Force flush stdout for XPU print buffer compatibility
print = functools.partial(print, flush=True)

TRAINING_DIR = ROOT / "data" / "processed" / "training"
MODEL_OUTPUT_DIR = ROOT / "models" / "emotion_model"
LOG_DIR = ROOT / "tmp" / "training_logs"

# ── Hyperparameters ──────────────────────────────────────────
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MAX_LENGTH = 256
BATCH_SIZE = 16  # reduced from 32: max_length=256 doubles tokens/batch
EPOCHS = 15
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
DROPOUT = 0.2
GRAD_ACCUM_STEPS = 1
EARLY_STOP_PATIENCE = 4
COSINE_RESTART_PERIOD = 3
LABEL_TEMPERATURE = 1.0
LOSS_MODE = "v1"
USE_AMP = True
LR_PATIENCE = 3
LR_FACTOR = 0.5

EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

# Dimension-level class weights for weighted KL loss
# sqrt(1/freq) normalized to mean=1.0 — up-weights rare emotions (fear, surprise)
_W = [0.4186, 0.8983, 0.6890, 1.6221, 1.9884, 0.3837]  # joy, sad, ang, fear, sur, neu
DIM_WEIGHTS = torch.tensor(_W, dtype=torch.float32)
del _W


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_imports():
    """Import local modules from scripts/"""
    import importlib.util
    scripts_dir = ROOT / "scripts"

    def _load(name):
        path = scripts_dir / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    return _load("12_emotion_dataset"), _load("13_emotion_model")


def standard_kl_loss(pred_logits, target_probs):
    """Standard KL divergence (v1) — uniform treatment across all emotion dims."""
    log_pred = nn.functional.log_softmax(pred_logits, dim=-1)
    per_dim_kl = target_probs * (torch.log(target_probs + 1e-9) - log_pred)
    return per_dim_kl.sum(dim=-1).mean()


def weighted_kl_loss(pred_logits, target_probs):
    """Weighted KL divergence — dimension-level rebalancing for rare emotions.
    Loss = sum(w_i * y_i * log(y_i / p_i)) across 6 dims.
    fear (1.62x) and surprise (1.99x) get higher gradient; neutral (0.38x) lower.
    """
    log_pred = nn.functional.log_softmax(pred_logits, dim=-1)
    w = DIM_WEIGHTS.to(pred_logits.device)
    # Per-dim KL: y_i * (log(y_i) - log(p_i))
    per_dim_kl = target_probs * (torch.log(target_probs + 1e-9) - log_pred)
    weighted = w.unsqueeze(0) * per_dim_kl  # (batch, 6) * (6,) → (batch, 6)
    return weighted.sum(dim=-1).mean()


def compute_loss(pred_logits, target_probs):
    if LOSS_MODE == "v3":
        return weighted_kl_loss(pred_logits, target_probs)
    return standard_kl_loss(pred_logits, target_probs)


def train_one_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    total_batches = len(loader)
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if scaler is not None:
            with torch.amp.autocast("xpu" if device.type == "xpu" else "cuda"):
                logits = model(input_ids, attention_mask)
                loss = compute_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask)
            loss = compute_loss(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if (step + 1) % 200 == 0:
            print(f"  batch {step+1}/{total_batches}  loss={loss.item():.4f}")

    return total_loss / (step + 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = compute_loss(logits, labels)
        total_loss += loss.item()

        preds = nn.functional.softmax(logits, dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Unweighted val_loss for cross-version comparison (V1/V2/V3)
    unweighted_kl = nn.functional.kl_div(
        torch.log(all_preds + 1e-9), all_labels,
        reduction="batchmean"
    )

    # Per-dim MAE (pure — no weights, directly comparable across versions)
    mae = (all_preds - all_labels).abs().mean(dim=0)

    # Argmax accuracy
    acc = (all_preds.argmax(dim=-1) == all_labels.argmax(dim=-1)).float().mean()

    return total_loss / (step + 1), acc.item(), mae, unweighted_kl.item()


def save_model(model, tokenizer, output_dir, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Save classifier head separately
    torch.save(model.classifier.state_dict(), output_dir / "classifier.pt")
    if metadata:
        with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Model saved to {output_dir}")


def main():
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

    # Log to file (timestamped, avoid overwriting previous runs)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_stem = datetime.now().strftime("run_%m%d_%H%M")
    sys.stdout = open(LOG_DIR / f"{log_stem}.log", "w", encoding="utf-8", buffering=1)

    device = get_device()
    amp_dtype = torch.bfloat16 if USE_AMP and device.type in ("xpu", "cuda") else None
    scaler = torch.amp.GradScaler(device.type) if amp_dtype == torch.bfloat16 else None

    print("=" * 60)
    print(f"Script 14: Train Emotion Model ({LOSS_MODE} — KL distillation)")
    print(f"  Device:     {device}")
    print(f"  AMP:        BF16" if scaler else f"  AMP:        FP32 (no AMP)")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Temp:       {LABEL_TEMPERATURE}")
    print("=" * 60)

    # Setup
    ds_mod, model_mod = setup_imports()

    # Datasets
    print("\nLoading datasets...")
    train_ds = ds_mod.EmotionDataset(
        str(TRAINING_DIR / "train.csv"), tokenizer_name=MODEL_NAME, max_length=MAX_LENGTH,
        temperature=LABEL_TEMPERATURE,
    )
    val_ds = ds_mod.EmotionDataset(
        str(TRAINING_DIR / "val.csv"), tokenizer_name=MODEL_NAME, max_length=MAX_LENGTH,
        temperature=LABEL_TEMPERATURE,
    )
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Model
    print(f"\nLoading model: {MODEL_NAME}")
    model = model_mod.EmotionClassifier(model_name=MODEL_NAME, dropout=DROPOUT)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_params:,} total / {trainable:,} trainable")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"  Total steps: {total_steps}  Warmup: {warmup_steps}")

    # Training loop
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    # Clear XPU cache before training
    if device.type == "xpu":
        torch.xpu.empty_cache()
        print(f"  XPU memory: {torch.xpu.memory_allocated()/1024**3:.2f}GB allocated, "
              f"{torch.xpu.max_memory_allocated()/1024**3:.2f}GB peak")

    print(f"\n{'='*60}")
    print("Training")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        try:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, scaler
            )
        except Exception as e:
            print(f"\n[FATAL] train_one_epoch crashed at epoch {epoch}: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)
        try:
            val_loss_w, val_acc, val_mae, val_loss_uw = evaluate(model, val_loader, device)
        except Exception as e:
            print(f"\n[FATAL] evaluate crashed at epoch {epoch}: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

        avg_mae = val_mae.mean().item()
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss_weighted": val_loss_w, "val_loss_unweighted": val_loss_uw,
            "val_acc": val_acc, "val_mae": avg_mae,
            "val_mae_per_dim": {k: float(v) for k, v in zip(EMOTION_KEYS, val_mae)},
        })

        mae_str = "  ".join([f"{k}={v:.4f}" for k, v in zip(EMOTION_KEYS, val_mae)])
        print(f"Epoch {epoch:2d} | train_loss={train_loss:.6f}  "
              f"val_loss(w)={val_loss_w:.4f}  val_loss(uw)={val_loss_uw:.4f}  "
              f"val_acc={val_acc:.4f}  val_mae={avg_mae:.4f}")
        print(f"         | per-dim MAE: {mae_str}")

        # Check if best — use unweighted val_loss for cross-version comparability
        if val_loss_uw < best_val_loss:
            best_val_loss = val_loss_uw
            best_epoch = epoch
            patience_counter = 0
            save_model(model, train_ds.tokenizer, MODEL_OUTPUT_DIR, metadata={
                "model_name": MODEL_NAME,
                "best_epoch": best_epoch,
                "val_loss": val_loss_uw,  # unweighted — comparable to V1/V2
                "val_loss_weighted": val_loss_w,
                "val_acc": val_acc,
                "val_mae": avg_mae,
                "val_mae_per_dim": {k: float(v) for k, v in zip(EMOTION_KEYS, val_mae)},
                "hyperparams": {
                    "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
                    "warmup_ratio": WARMUP_RATIO, "epochs": EPOCHS,
                    "max_length": MAX_LENGTH, "dropout": DROPOUT,
                    "label_temperature": LABEL_TEMPERATURE,
                    "cosine_restart_period": COSINE_RESTART_PERIOD,
                },
                "timestamp": datetime.now().isoformat(),
            })
            print(f"         | ✓ Best model saved (epoch {best_epoch})")
        else:
            patience_counter += 1
            print(f"         | No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save training history
    history_path = LOG_DIR / f"{log_stem}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete. Best epoch: {best_epoch} (val_loss_uw={best_val_loss:.6f})")
    print(f"Model saved to: {MODEL_OUTPUT_DIR}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
