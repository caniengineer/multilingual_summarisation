"""One-off script to download evaluation samples and write data files.

Usage:
    python scripts/prepare_data.py           # legacy mode (existing behavior)
    python scripts/prepare_data.py --all     # download all sources, write all tiers
    python scripts/prepare_data.py --tier smoke  # refresh a specific tier

Downloads from:
  - mesolitica/mixtral-malaysian-abstractive-summarization (Malay, Mixtral-generated refs)
  - csebuetnlp/xlsum (English, human-written refs)
  - huseinzol05/malay-dataset (Malay, human-curated refs)

Writes samples as JSON under data/evaluation/{smoke,regression,benchmark}/
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
MALAY_DATASET_SUMMARIZATION = (
    "https://raw.githubusercontent.com/huseinzol05/malay-dataset"
    "/master/summarization/long-news-with-summaries.json"
)

# Tier sample counts
TIER_COUNTS = {
    "smoke": {"ms_human": 3, "ms_mixtral": 2, "en": 3, "cs": 2},
    "regression": {"ms_human": 30, "ms_mixtral": 30, "en": 25, "cs": 15},
    "benchmark": {"ms_human": 100, "ms_mixtral": 200, "en": 150, "cs": 50},
}


def allocate_tiers(
    all_indices: list[int],
    smoke: int,
    regression: int,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Allocate indices into nested tiers: smoke ⊂ regression ⊂ benchmark."""
    rng = random.Random(seed)
    shuffled = list(all_indices)
    rng.shuffle(shuffled)

    benchmark = shuffled  # all samples
    regression_set = shuffled[:regression]
    smoke_set = shuffled[:smoke]

    return {
        "smoke": smoke_set,
        "regression": regression_set,
        "benchmark": benchmark,
    }


def write_tiered_samples(
    samples: list[dict],
    output_dir: Path,
    source: str,
    language: str,
    ref_quality: str,
    ref_field: str | None,
    id_prefix: str,
) -> int:
    """Write sample JSON files to output directory. Returns count written."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing .gitkeep
    gitkeep = output_dir / ".gitkeep"
    if gitkeep.exists():
        gitkeep.unlink()

    target_language = {"ms": "ms", "en": "en", "ms_en": "auto"}.get(language, "auto")

    for i, sample in enumerate(samples, start=1):
        sample_id = f"{id_prefix}_{i:03d}"
        data = {
            "id": sample_id,
            "source": source,
            "language": language,
            "category": "codeswitching" if language == "ms_en" else "monolingual" if language == "ms" else "english",
            "has_reference": ref_field is not None and ref_field in sample,
            "ref_quality": ref_quality,
            "text": sample["text"],
            "config": {
                "target_language": target_language,
                "summary_type": "brief",
                "max_length": 150,
            },
        }
        if ref_field and ref_field in sample:
            data["reference_summary"] = sample[ref_field]

        filepath = output_dir / f"{i:03d}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return len(samples)


def write_tiered_manifest(data_dir: Path, tier_counts: dict) -> None:
    """Write manifest.json with tier structure."""
    tiers = {}
    tier_descriptions = {
        "smoke": "Fast sanity check (~10 samples)",
        "regression": "Quality gate (~100 samples)",
        "benchmark": "Full benchmark (~500 samples)",
    }
    for tier_name, categories in tier_counts.items():
        tiers[tier_name] = {
            "description": tier_descriptions.get(tier_name, ""),
            "categories": {},
        }
        for cat_name, count in categories.items():
            tiers[tier_name]["categories"][cat_name] = {
                "path": f"{tier_name}/{cat_name}",
                "count": count,
            }

    manifest = {"version": "2.0.0", "tiers": tiers}
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def download_malay_dataset() -> list[dict]:
    """Download Malay-Dataset human-curated summaries."""
    print("Downloading Malay-Dataset (human-curated)...")
    import urllib.request
    with urllib.request.urlopen(MALAY_DATASET_SUMMARIZATION) as resp:
        raw = json.loads(resp.read().decode())
    # Filter for samples with both text and summary
    samples = [s for s in raw if s.get("text") and s.get("summary")]
    print(f"  Found {len(samples)} samples with summaries")
    return samples


def download_all_sources() -> dict:
    """Download all data sources. Returns dict of source -> list of samples."""
    rng = random.Random(SEED)
    sources = {}

    # Malay human references (Malay-Dataset)
    try:
        sources["ms_human"] = download_malay_dataset()
    except Exception as e:
        print(f"  WARNING: Could not download Malay-Dataset: {e}")
        sources["ms_human"] = []

    # Malay machine references (Mesolitica Mixtral)
    print("Downloading Mesolitica Mixtral dataset (Malay)...")
    ms_df = pd.read_parquet(MESOLITICA_PARQUET)
    ms_df = ms_df[ms_df["summary_ms"].str.len() > 20].reset_index(drop=True)
    sources["ms_mixtral"] = [ms_df.iloc[i].to_dict() for i in range(len(ms_df))]
    print(f"  Found {len(sources['ms_mixtral'])} samples")

    # English (XLSum)
    print("Downloading XLSum dataset (English)...")
    en_df = pd.read_parquet(XLSUM_EN_PARQUET)
    sources["en"] = [en_df.iloc[i].to_dict() for i in range(len(en_df))]
    print(f"  Found {len(sources['en'])} samples")

    return sources


def build_tiers(target_tier: str | None = None):
    """Download sources and write tiered sample files."""
    sources = download_all_sources()

    tiers_to_build = (
        [target_tier] if target_tier else ["smoke", "regression", "benchmark"]
    )
    actual_counts = {}

    for tier in tiers_to_build:
        counts = TIER_COUNTS[tier]
        actual_counts[tier] = {}

        # Malay human-ref
        if "ms_human" in counts and sources["ms_human"]:
            indices = allocate_tiers(
                list(range(len(sources["ms_human"]))),
                smoke=TIER_COUNTS["smoke"].get("ms_human", 3),
                regression=TIER_COUNTS["regression"].get("ms_human", 30),
            )
            selected = [sources["ms_human"][i] for i in indices[tier]]
            cat_key = "ms" if tier == "smoke" else "ms_human"
            n = write_tiered_samples(
                samples=selected[:counts.get(cat_key, counts.get("ms_human", 0))],
                output_dir=DATA_DIR / tier / cat_key,
                source="malay-dataset",
                language="ms",
                ref_quality="human",
                ref_field="summary",
                id_prefix=f"malaydata_ms",
            )
            actual_counts[tier][cat_key] = n

        # Malay Mixtral-ref
        ms_mixtral_key = "ms_mixtral" if tier != "smoke" else None
        if ms_mixtral_key and ms_mixtral_key in counts and sources["ms_mixtral"]:
            indices = allocate_tiers(
                list(range(len(sources["ms_mixtral"]))),
                smoke=TIER_COUNTS["smoke"].get("ms_mixtral", 2),
                regression=TIER_COUNTS["regression"].get("ms_mixtral", 30),
            )
            selected = [sources["ms_mixtral"][i] for i in indices[tier]]
            n = write_tiered_samples(
                samples=selected[:counts["ms_mixtral"]],
                output_dir=DATA_DIR / tier / "ms_mixtral",
                source="mesolitica-mixtral",
                language="ms",
                ref_quality="machine",
                ref_field="summary_ms",
                id_prefix=f"mesolitica_ms",
            )
            actual_counts[tier]["ms_mixtral"] = n

        # For smoke tier, Mixtral samples go into ms/ dir
        if tier == "smoke" and "ms_mixtral" in counts and sources["ms_mixtral"]:
            indices = allocate_tiers(
                list(range(len(sources["ms_mixtral"]))),
                smoke=counts["ms_mixtral"],
                regression=TIER_COUNTS["regression"].get("ms_mixtral", 30),
            )
            selected = [sources["ms_mixtral"][i] for i in indices["smoke"]]
            # Append to ms/ dir (offset numbering)
            ms_dir = DATA_DIR / tier / "ms"
            existing = len(list(ms_dir.glob("*.json"))) if ms_dir.exists() else 0
            for j, sample in enumerate(selected, start=existing + 1):
                data = {
                    "id": f"mesolitica_ms_{j:03d}",
                    "source": "mesolitica-mixtral",
                    "language": "ms",
                    "category": "monolingual",
                    "has_reference": True,
                    "ref_quality": "machine",
                    "text": sample["text"],
                    "reference_summary": sample["summary_ms"],
                    "config": {"target_language": "ms", "summary_type": "brief", "max_length": 150},
                }
                filepath = ms_dir / f"{j:03d}.json"
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            actual_counts[tier]["ms"] = actual_counts[tier].get("ms", 0) + len(selected)

        # English
        if "en" in counts and sources["en"]:
            indices = allocate_tiers(
                list(range(len(sources["en"]))),
                smoke=TIER_COUNTS["smoke"]["en"],
                regression=TIER_COUNTS["regression"]["en"],
            )
            selected = [sources["en"][i] for i in indices[tier]]
            n = write_tiered_samples(
                samples=selected[:counts["en"]],
                output_dir=DATA_DIR / tier / "en",
                source="xlsum",
                language="en",
                ref_quality="human",
                ref_field="summary",
                id_prefix=f"xlsum_en",
            )
            actual_counts[tier]["en"] = n

        # Code-switching (empty dirs with README)
        cs_dir = DATA_DIR / tier / "cs"
        cs_dir.mkdir(parents=True, exist_ok=True)
        actual_counts[tier]["cs"] = len(list(cs_dir.glob("*.json")))

        print(f"\n  Tier '{tier}': {sum(actual_counts[tier].values())} samples written")

    write_tiered_manifest(DATA_DIR, actual_counts)


# --- Legacy functions (keep for backward compatibility) ---

def download_samples():
    """Download Malay and English samples from HF parquet files."""
    rng = random.Random(SEED)

    print("Downloading mesolitica dataset (Malay)...")
    ms_df = pd.read_parquet(MESOLITICA_PARQUET)
    ms_df = ms_df[
        ms_df["summary_ms"].str.len() > 20
    ].reset_index(drop=True)
    ms_indices = rng.sample(range(len(ms_df)), min(MS_COUNT, len(ms_df)))
    ms_samples = [ms_df.iloc[i].to_dict() for i in ms_indices]

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
    import argparse

    parser = argparse.ArgumentParser(description="Prepare evaluation data")
    parser.add_argument("--all", action="store_true", help="Download all sources, write all tiers")
    parser.add_argument("--tier", choices=["smoke", "regression", "benchmark"], help="Refresh a specific tier")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.tier:
        build_tiers(target_tier=args.tier)
        print("\nDone. Commit data/evaluation/ to the repo.")
        return

    # Legacy mode (original behavior)
    ms_samples, en_samples = download_samples()
    print(f"\nWriting {len(ms_samples)} Malay samples...")
    write_ms_samples(ms_samples, output_dir=DATA_DIR / "xlsum_ms")
    print(f"\nWriting {len(en_samples)} English samples...")
    write_en_samples(en_samples, output_dir=DATA_DIR / "english")
    cs_dir = DATA_DIR / "codeswitching"
    cs_dir.mkdir(parents=True, exist_ok=True)
    write_manifest()
    print("\nDone. Commit data/evaluation/ to the repo.")


if __name__ == "__main__":
    main()
