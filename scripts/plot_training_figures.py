"""
Generate publication-quality training figures from training_history.json.
"""
import sys
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = ROOT / "tmp" / "training_logs" / "training_history.json"
META_PATH = ROOT / "models" / "emotion_model" / "training_metadata.json"
FIGURES_DIR = ROOT / "figures"

EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
EMOTION_CN = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Neutral"]
COLORS = ["#E63946", "#457B9D", "#F4A261", "#2A9D8F", "#9B5DE5", "#6C757D"]

# ── Paper-style matplotlib config ────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


def load_history():
    with open(HISTORY_PATH, encoding="utf-8") as f:
        return json.load(f)


def plot_loss_curve(history):
    """Fig 1: Training & Validation Loss."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_loss, "o-", color="#2A9D8F", linewidth=1.8,
            markersize=7, markerfacecolor="white", markeredgewidth=1.5, label="Train Loss")
    ax.plot(epochs, val_loss, "s-", color="#E63946", linewidth=1.8,
            markersize=7, markerfacecolor="white", markeredgewidth=1.5, label="Val Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(0.8, max(epochs) + 0.2)

    # Annotate min val loss
    best = min(val_loss)
    best_epoch = epochs[val_loss.index(best)]
    ax.annotate(f"Best: {best:.4f}",
                xy=(best_epoch, best), xytext=(best_epoch + 0.6, best + 0.008),
                arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.2),
                fontsize=10, color="#E63946", fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_loss_curve.pdf")
    fig.savefig(FIGURES_DIR / "fig1_loss_curve.png")
    plt.close(fig)
    print("  [OK] fig1_loss_curve")


def plot_accuracy_curve(history):
    """Fig 2: Validation Accuracy curve."""
    epochs = [h["epoch"] for h in history]
    val_acc = [h["val_acc"] * 100 for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, val_acc, "D-", color="#1D3557", linewidth=1.8,
            markersize=8, markerfacecolor="white", markeredgewidth=1.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Argmax Accuracy")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(0.8, max(epochs) + 0.2)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    best = max(val_acc)
    best_epoch = epochs[val_acc.index(best)]
    ax.annotate(f"Best: {best:.2f}%",
                xy=(best_epoch, best), xytext=(best_epoch + 0.6, best - 1.5),
                arrowprops=dict(arrowstyle="->", color="#1D3557", lw=1.2),
                fontsize=10, color="#1D3557", fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_accuracy_curve.pdf")
    fig.savefig(FIGURES_DIR / "fig2_accuracy_curve.png")
    plt.close(fig)
    print("  [OK] fig2_accuracy_curve")


def plot_mae_curve(history):
    """Fig 3: Validation MAE curve."""
    epochs = [h["epoch"] for h in history]
    val_mae = [h["val_mae"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, val_mae, "p-", color="#9B5DE5", linewidth=1.8,
            markersize=8, markerfacecolor="white", markeredgewidth=1.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Validation MAE (Average over 6 Dimensions)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(0.8, max(epochs) + 0.2)

    best = min(val_mae)
    best_epoch = epochs[val_mae.index(best)]
    ax.annotate(f"Best: {best:.4f}",
                xy=(best_epoch, best), xytext=(best_epoch + 0.6, best + 0.0005),
                arrowprops=dict(arrowstyle="->", color="#9B5DE5", lw=1.2),
                fontsize=10, color="#9B5DE5", fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_mae_curve.pdf")
    fig.savefig(FIGURES_DIR / "fig3_mae_curve.png")
    plt.close(fig)
    print("  [OK] fig3_mae_curve")


def plot_per_dim_mae(history):
    """Fig 4: Per-dimension MAE at best epoch (horizontal bar)."""
    best = min(history, key=lambda h: h["val_loss"])
    mae_dict = best["val_mae_per_dim"]
    values = [mae_dict[k] for k in EMOTION_KEYS]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = range(len(EMOTION_KEYS))
    bars = ax.barh(y_pos, values, height=0.55, color=COLORS, edgecolor="white", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(EMOTION_CN)
    ax.set_xlabel("Mean Absolute Error")
    ax.set_title(f"Per-Dimension MAE (Epoch {best['epoch']}, val_loss={best['val_loss']:.4f})")
    ax.invert_yaxis()

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    # Goal line
    ax.axvline(x=0.15, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.15 + 0.002, -0.45, "Target (0.15)", fontsize=8, color="gray", va="bottom")

    ax.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_per_dim_mae.pdf")
    fig.savefig(FIGURES_DIR / "fig4_per_dim_mae.png")
    plt.close(fig)
    print("  [OK] fig4_per_dim_mae")


def plot_combined(history):
    """Fig 5: Combined 3-panel training dynamics (academic paper style)."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_acc"] * 100 for h in history]
    val_mae = [h["val_mae"] for h in history]
    best = min(history, key=lambda h: h["val_loss"])
    mae_dict = best["val_mae_per_dim"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, "o-", color="#2A9D8F", linewidth=1.6,
            markersize=6, markerfacecolor="white", markeredgewidth=1.3, label="Train")
    ax.plot(epochs, val_loss, "s-", color="#E63946", linewidth=1.6,
            markersize=6, markerfacecolor="white", markeredgewidth=1.3, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.set_title("(a) Loss", fontsize=12, loc="left", fontweight="bold")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Panel B: Accuracy
    ax = axes[1]
    ax.plot(epochs, val_acc, "D-", color="#1D3557", linewidth=1.6,
            markersize=6, markerfacecolor="white", markeredgewidth=1.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Val Accuracy", fontsize=12, loc="left", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Panel C: Per-dim MAE
    ax = axes[2]
    values = [mae_dict[k] for k in EMOTION_KEYS]
    y_pos = range(len(EMOTION_KEYS))
    ax.barh(y_pos, values, height=0.55, color=COLORS, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(EMOTION_CN, fontsize=9)
    ax.set_xlabel("MAE")
    ax.set_title("(c) Per-Dim MAE (Best)", fontsize=12, loc="left", fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(x=0.15, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, val in zip(ax.containers[0], values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7.5)

    fig.suptitle("Emotion Model Training Dynamics", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_combined.pdf")
    fig.savefig(FIGURES_DIR / "fig5_combined.png")
    plt.close(fig)
    print("  [OK] fig5_combined")


def plot_convergence_detail(history):
    """Fig 6: Loss delta per epoch — shows convergence speed."""
    epochs = [h["epoch"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    deltas = [0.0] + [val_loss[i-1] - val_loss[i] for i in range(1, len(val_loss))]
    # Scale to percentage improvement
    pct_improve = [0.0] + [
        (val_loss[i-1] - val_loss[i]) / val_loss[i-1] * 100
        for i in range(1, len(val_loss))
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Absolute delta
    colors_delta = ["#2A9D8F" if d > 0.001 else "#CCC" for d in deltas]
    ax1.bar(epochs, deltas, color=colors_delta, edgecolor="white", width=0.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Δ Val Loss")
    ax1.set_title("(a) Absolute Improvement per Epoch", loc="left", fontweight="bold")
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # Add value on top
    for e, d in zip(epochs, deltas):
        if d > 0:
            ax1.text(e, d + 0.0005, f"{d:.4f}", ha="center", fontsize=8)

    # Relative
    colors_pct = ["#E63946" if p > 0.1 else "#CCC" for p in pct_improve]
    ax2.bar(epochs, pct_improve, color=colors_pct, edgecolor="white", width=0.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Improvement (%)")
    ax2.set_title("(b) Relative Improvement per Epoch", loc="left", fontweight="bold")
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for e, p in zip(epochs, pct_improve):
        if p > 0:
            ax2.text(e, p + 0.05, f"{p:.2f}%", ha="center", fontsize=8)

    fig.suptitle("Convergence Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_convergence.pdf")
    fig.savefig(FIGURES_DIR / "fig6_convergence.png")
    plt.close(fig)
    print("  [OK] fig6_convergence")


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    if not HISTORY_PATH.exists():
        print(f"[ERROR] Training history not found: {HISTORY_PATH}")
        sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    print(f"Generating figures from {len(history)} epochs of data...\n")

    plot_loss_curve(history)
    plot_accuracy_curve(history)
    plot_mae_curve(history)
    plot_per_dim_mae(history)
    plot_combined(history)
    plot_convergence_detail(history)

    print(f"\nDone. {len(list(FIGURES_DIR.glob('*.pdf')))} figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
