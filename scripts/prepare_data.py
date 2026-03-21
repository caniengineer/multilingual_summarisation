"""One-off script to download evaluation samples and write data files.

Usage:
    python scripts/prepare_data.py

Downloads Malay samples from mesolitica/mixtral-malaysian-abstractive-summarization
and English samples from csebuetnlp/xlsum (via HF parquet conversion),
writes them as individual JSON files under data/evaluation/,
and generates a manifest.json index.

Run once, commit the output, never run again.
"""

import json
import random
from pathlib import Path

import pandas as pd

SEED = 42
DATA_DIR = Path(__file__).parent.parent / "data" / "evaluation"
MS_COUNT = 12
EN_COUNT = 5

MESOLITICA_PARQUET = (
    "https://huggingface.co/datasets/mesolitica/mixtral-malaysian-abstractive-summarization"
    "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
XLSUM_EN_PARQUET = (
    "https://huggingface.co/datasets/csebuetnlp/xlsum"
    "/resolve/refs%2Fconvert%2Fparquet/english/test/0000.parquet"
)


def download_samples():
    """Download Malay and English samples from HF parquet files."""
    rng = random.Random(SEED)

    # Malay samples from mesolitica
    print("Downloading mesolitica dataset (Malay)...")
    ms_df = pd.read_parquet(MESOLITICA_PARQUET)
    # Filter for rows that have non-empty Malay summaries and reasonable text length
    ms_df = ms_df[
        ms_df["summary_ms"].str.len() > 20
    ].reset_index(drop=True)
    ms_indices = rng.sample(range(len(ms_df)), min(MS_COUNT, len(ms_df)))
    ms_samples = [ms_df.iloc[i].to_dict() for i in ms_indices]

    # English samples from XL-Sum
    print("Downloading XL-Sum dataset (English)...")
    en_df = pd.read_parquet(XLSUM_EN_PARQUET)
    en_indices = rng.sample(range(len(en_df)), min(EN_COUNT, len(en_df)))
    en_samples = [en_df.iloc[i].to_dict() for i in en_indices]

    return ms_samples, en_samples


def write_ms_samples(samples, output_dir):
    """Write Malay sample JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples, start=1):
        sample_id = f"mesolitica_ms_{i:03d}"
        data = {
            "id": sample_id,
            "source": "mesolitica",
            "language": "ms",
            "category": "monolingual",
            "has_reference": True,
            "text": sample["text"],
            "reference_summary": sample["summary_ms"],
            "config": {
                "target_language": "ms",
                "summary_type": "brief",
                "max_length": 150,
            },
        }
        filepath = output_dir / f"{i:03d}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Wrote {filepath.relative_to(DATA_DIR)}")


def write_en_samples(samples, output_dir):
    """Write English sample JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples, start=1):
        sample_id = f"xlsum_en_{i:03d}"
        data = {
            "id": sample_id,
            "source": "xlsum",
            "language": "en",
            "category": "english",
            "has_reference": True,
            "text": sample["text"],
            "reference_summary": sample["summary"],
            "config": {
                "target_language": "en",
                "summary_type": "brief",
                "max_length": 150,
            },
        }
        filepath = output_dir / f"{i:03d}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Wrote {filepath.relative_to(DATA_DIR)}")


def write_manifest():
    """Write manifest.json index file."""
    manifest = {
        "version": "1.0.0",
        "categories": {
            "monolingual": {"path": "xlsum_ms", "count": MS_COUNT},
            "codeswitching": {"path": "codeswitching", "count": 0},
            "english": {"path": "english", "count": EN_COUNT},
        },
    }
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {manifest_path}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ms_samples, en_samples = download_samples()

    print(f"\nWriting {len(ms_samples)} Malay samples...")
    write_ms_samples(ms_samples, output_dir=DATA_DIR / "xlsum_ms")

    print(f"\nWriting {len(en_samples)} English samples...")
    write_en_samples(en_samples, output_dir=DATA_DIR / "english")

    # Create empty codeswitching directory (populated manually in Task 3)
    cs_dir = DATA_DIR / "codeswitching"
    cs_dir.mkdir(parents=True, exist_ok=True)

    write_manifest()
    print("\nDone. Commit data/evaluation/ to the repo.")


if __name__ == "__main__":
    main()
