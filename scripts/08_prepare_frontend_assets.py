"""
Script 08: Frontend Static Assets Export
Copies processed data and analysis images to app/public/ for frontend serving.
Writes app/public/data/manifest.json
"""
import shutil
import json
import sys
import hashlib
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
ANALYSIS_DIR = ROOT / "analysis"
APP_PUBLIC = ROOT / "app" / "public"

FILES_TO_COPY = [
    # Data files
    ("data/processed/emotion_panel_weekly.csv", "data/processed/emotion_panel_weekly.csv"),
    ("data/processed/emotion_panel_monthly.csv", "data/processed/emotion_panel_monthly.csv"),
    ("data/processed/emotion_national_timeline.csv", "data/processed/emotion_national_timeline.csv"),
    ("data/processed/province_emotion_vectors.csv", "data/processed/province_emotion_vectors.csv"),
    ("data/processed/anomaly_detection.json", "data/processed/anomaly_detection.json"),
    ("data/processed/cluster_labels.csv", "data/processed/cluster_labels.csv"),
    ("tmp/07_monthly_cluster_labels.csv", "data/processed/monthly_cluster_labels.csv"),
]

NLP_FILES = [
    ("data/processed/nlp_keywords_by_week.json", "data/processed/nlp_keywords_by_week.json"),
    ("data/processed/nlp_emotion_keywords.json", "data/processed/nlp_emotion_keywords.json"),
    ("data/processed/nlp_global_vocabulary.json", "data/processed/nlp_global_vocabulary.json"),
]

IMAGE_GLOBS = [
    "analysis/emotion_validation/*.png",
    "analysis/province_clustering/*.png",
    "analysis/temporal_cluster_evolution/*.png",
    "analysis/wordclouds/*.png",
]


def sha256_short(path):
    """First 12 chars of SHA256 hex"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def count_rows(path):
    """Count rows in CSV (minus header) or items in JSON array/object"""
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            with open(path, "r", encoding="utf-8-sig") as f:
                return max(0, sum(1 for _ in f) - 1)
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                return len(data.get("weeks", data.get("emotions", data.get("vocabulary", data))))
            return 1
        elif ext == ".png":
            return -1  # binary, row count N/A
    except Exception:
        return 0
    return 0


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import argparse
    parser = argparse.ArgumentParser(description="Frontend Static Assets Export")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Processed data directory (default: data/processed/)")
    parser.add_argument("--analysis-dir", type=str, default=None,
                        help="Analysis images directory (default: analysis/)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: app/public/)")
    args = parser.parse_args()

    print("=" * 60)
    print("Script 08: Frontend Static Assets Export")
    print("=" * 60)

    data_dir = Path(args.data_dir) if args.data_dir else PROCESSED_DIR
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else ANALYSIS_DIR
    if not analysis_dir.is_absolute():
        analysis_dir = ROOT / analysis_dir
    app_public = Path(args.output_dir) if args.output_dir else APP_PUBLIC
    if not app_public.is_absolute():
        app_public = ROOT / app_public

    if not app_public.exists():
        print(f"[INFO] {app_public} does not exist. Creating...")
        app_public.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "files": [],
    }

    copied = 0
    errors = []

    # Build dynamic file list based on data_dir and analysis_dir
    files_to_copy = [
        (data_dir / "emotion_panel_weekly.csv", "data/processed/emotion_panel_weekly.csv"),
        (data_dir / "emotion_panel_monthly.csv", "data/processed/emotion_panel_monthly.csv"),
        (data_dir / "emotion_national_timeline.csv", "data/processed/emotion_national_timeline.csv"),
        (data_dir / "province_emotion_vectors.csv", "data/processed/province_emotion_vectors.csv"),
        (data_dir / "anomaly_detection.json", "data/processed/anomaly_detection.json"),
        (data_dir / "cluster_labels.csv", "data/processed/cluster_labels.csv"),
        (ROOT / "tmp" / "07_monthly_cluster_labels.csv", "data/processed/monthly_cluster_labels.csv"),
    ]
    image_globs = [
        str(analysis_dir / "emotion_validation" / "*.png"),
        str(analysis_dir / "province_clustering" / "*.png"),
        str(analysis_dir / "temporal_cluster_evolution" / "*.png"),
        str(analysis_dir / "wordclouds" / "*.png"),
    ]

    # Copy data files
    for src, dst_rel in files_to_copy:
        src_rel = str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src)
        dst = app_public / dst_rel

        if not src.exists():
            errors.append(f"MISSING: {src_rel}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        row_count = count_rows(dst)
        file_hash = sha256_short(dst) if dst.suffix != ".png" else ""

        manifest["files"].append({
            "path": dst_rel,
            "source": src_rel,
            "hash": file_hash,
            "row_count": row_count if row_count >= 0 else "N/A (image)",
            "size_bytes": dst.stat().st_size,
        })
        copied += 1
        print(f"  [OK] {src_rel} -> {dst_rel} ({row_count} rows)" if row_count >= 0
              else f"  [OK] {src_rel} -> {dst_rel}")

    # Copy NLP files (optional, don't fail if missing)
    nlp_available = True
    nlp_files_to_copy = [
        (data_dir / "nlp_keywords_by_week.json", "data/processed/nlp_keywords_by_week.json"),
        (data_dir / "nlp_emotion_keywords.json", "data/processed/nlp_emotion_keywords.json"),
        (data_dir / "nlp_global_vocabulary.json", "data/processed/nlp_global_vocabulary.json"),
    ]
    for src, dst_rel in nlp_files_to_copy:
        src_rel = str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src)
        dst = app_public / dst_rel

        if not src.exists():
            nlp_available = False
            print(f"  [SKIP] NLP file not found: {src_rel}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        row_count = count_rows(dst)
        file_hash = sha256_short(dst)

        manifest["files"].append({
            "path": dst_rel,
            "source": src_rel,
            "hash": file_hash,
            "row_count": row_count,
            "size_bytes": dst.stat().st_size,
        })
        copied += 1
        print(f"  [OK] {src_rel} -> {dst_rel} (NLP)")

    manifest["nlp_keywords_available"] = nlp_available

    # Copy images
    image_dirs = [
        analysis_dir / "emotion_validation",
        analysis_dir / "province_clustering",
        analysis_dir / "temporal_cluster_evolution",
        analysis_dir / "wordclouds",
    ]
    for img_dir in image_dirs:
        if not img_dir.exists():
            continue
        for img in img_dir.glob("*.png"):
            # Compute relative path preserving subdirectory structure
            try:
                rel = img.relative_to(analysis_dir)
                rel = Path("analysis") / rel
            except ValueError:
                rel = Path("analysis") / img_dir.name / img.name
            dst = app_public / str(rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst)

            manifest["files"].append({
                "path": str(rel),
                "source": str(rel),
                "hash": "",
                "row_count": "N/A (image)",
                "size_bytes": dst.stat().st_size,
            })
            copied += 1
            print(f"  [OK] {rel} -> {rel} (image)")

    # Write manifest
    manifest_path = app_public / "data" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Manifest: {manifest_path}")
    print(f"  {copied} files copied, {len(errors)} errors")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  [WARN] {e}")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
