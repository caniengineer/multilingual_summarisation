# Testing & Benchmarking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand evaluation datasets from 3 sources (Malay-Dataset, Mesolitica Mixtral, Malaysian-Dataset + existing XLSum) into a tiered structure (smoke/regression/benchmark) with baseline comparison and ref_quality-aware chrF++ reporting.

**Architecture:** Extend `scripts/prepare_data.py` to pull from all sources and allocate samples into nested tiers. Extend `scripts/evaluate.py` with `--tier` flag, `--save-baseline` flag, and ref_quality-split reporting. All samples committed to `data/evaluation/` as JSON files.

**Tech Stack:** Python 3.11, pandas, httpx, sacrebleu, datasets (HuggingFace), pytest, pytest-asyncio

---

### Task 1: Add `ref_quality` field to existing samples

**Files:**
- Modify: `data/evaluation/xlsum_ms/*.json` (12 files)
- Modify: `data/evaluation/english/*.json` (5 files)
- Modify: `data/evaluation/codeswitching/*.json` (8 files)

**Step 1: Write a migration script to add ref_quality to all existing samples**

Create a one-off script (run inline, don't save):

```python
# Run from project root
import json
from pathlib import Path

data_dir = Path("data/evaluation")

# Malay samples from Mesolitica Mixtral = machine-generated references
for f in sorted((data_dir / "xlsum_ms").glob("*.json")):
    with open(f) as fh:
        sample = json.load(fh)
    sample["ref_quality"] = "machine"
    with open(f, "w") as fh:
        json.dump(sample, fh, ensure_ascii=False, indent=2)

# English samples from XLSum = human-written (BBC journalists)
for f in sorted((data_dir / "english").glob("*.json")):
    with open(f) as fh:
        sample = json.load(fh)
    sample["ref_quality"] = "human"
    with open(f, "w") as fh:
        json.dump(sample, fh, ensure_ascii=False, indent=2)

# Code-switching = no reference summary
for f in sorted((data_dir / "codeswitching").glob("*.json")):
    with open(f) as fh:
        sample = json.load(fh)
    sample["ref_quality"] = "none"
    with open(f, "w") as fh:
        json.dump(sample, fh, ensure_ascii=False, indent=2)

print("Done: ref_quality added to all samples")
```

**Step 2: Verify a couple of files have the new field**

Run: `python -c "import json; print(json.load(open('data/evaluation/xlsum_ms/001.json'))['ref_quality'])"`
Expected: `machine`

Run: `python -c "import json; print(json.load(open('data/evaluation/english/001.json'))['ref_quality'])"`
Expected: `human`

**Step 3: Commit**

```bash
git add data/evaluation/
git commit -m "chore: add ref_quality field to existing evaluation samples"
```

---

### Task 2: Create tiered directory structure and update manifest

**Files:**
- Create: `data/evaluation/smoke/ms/` (directory)
- Create: `data/evaluation/smoke/en/` (directory)
- Create: `data/evaluation/smoke/cs/` (directory)
- Create: `data/evaluation/regression/ms_human/` (directory)
- Create: `data/evaluation/regression/ms_mixtral/` (directory)
- Create: `data/evaluation/regression/en/` (directory)
- Create: `data/evaluation/regression/cs/` (directory)
- Create: `data/evaluation/benchmark/ms_human/` (directory)
- Create: `data/evaluation/benchmark/ms_mixtral/` (directory)
- Create: `data/evaluation/benchmark/en/` (directory)
- Create: `data/evaluation/benchmark/cs/` (directory)
- Modify: `data/evaluation/manifest.json`

**Step 1: Create directory structure**

```bash
mkdir -p data/evaluation/{smoke/{ms,en,cs},regression/{ms_human,ms_mixtral,en,cs},benchmark/{ms_human,ms_mixtral,en,cs}}
```

**Step 2: Update manifest.json to support tiered structure**

Write new `data/evaluation/manifest.json`:

```json
{
  "version": "2.0.0",
  "tiers": {
    "smoke": {
      "description": "Fast sanity check (~10 samples)",
      "categories": {
        "ms": {"path": "smoke/ms", "count": 0},
        "en": {"path": "smoke/en", "count": 0},
        "cs": {"path": "smoke/cs", "count": 0}
      }
    },
    "regression": {
      "description": "Quality gate (~100 samples)",
      "categories": {
        "ms_human": {"path": "regression/ms_human", "count": 0},
        "ms_mixtral": {"path": "regression/ms_mixtral", "count": 0},
        "en": {"path": "regression/en", "count": 0},
        "cs": {"path": "regression/cs", "count": 0}
      }
    },
    "benchmark": {
      "description": "Full benchmark (~500 samples)",
      "categories": {
        "ms_human": {"path": "benchmark/ms_human", "count": 0},
        "ms_mixtral": {"path": "benchmark/ms_mixtral", "count": 0},
        "en": {"path": "benchmark/en", "count": 0},
        "cs": {"path": "benchmark/cs", "count": 0}
      }
    }
  },
  "legacy": {
    "categories": {
      "monolingual": {"path": "xlsum_ms", "count": 12},
      "codeswitching": {"path": "codeswitching", "count": 8},
      "english": {"path": "english", "count": 5}
    }
  }
}
```

**Step 3: Add .gitkeep files so empty dirs are tracked**

```bash
for d in data/evaluation/{smoke/{ms,en,cs},regression/{ms_human,ms_mixtral,en,cs},benchmark/{ms_human,ms_mixtral,en,cs}}; do
  touch "$d/.gitkeep"
done
```

**Step 4: Commit**

```bash
git add data/evaluation/
git commit -m "feat: add tiered directory structure for evaluation datasets"
```

---

### Task 3: Expand `prepare_data.py` to download from all sources

**Files:**
- Modify: `scripts/prepare_data.py`
- Test: `tests/test_prepare_data.py` (create)

**Step 1: Write failing test for tiered sample writing**

Create `tests/test_prepare_data.py`:

```python
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


def test_write_tiered_ms_human_samples(tmp_path):
    """Verify Malay human-ref samples are written with correct format."""
    from scripts.prepare_data import write_tiered_samples

    fake_samples = [
        {"text": "Teks berita pertama.", "summary": "Ringkasan pertama."},
        {"text": "Teks berita kedua.", "summary": "Ringkasan kedua."},
    ]

    write_tiered_samples(
        samples=fake_samples,
        output_dir=tmp_path / "ms_human",
        source="malay-dataset",
        language="ms",
        ref_quality="human",
        ref_field="summary",
        id_prefix="malaydata_ms",
    )

    files = sorted((tmp_path / "ms_human").glob("*.json"))
    assert len(files) == 2

    with open(files[0]) as f:
        data = json.load(f)

    assert data["id"] == "malaydata_ms_001"
    assert data["source"] == "malay-dataset"
    assert data["language"] == "ms"
    assert data["ref_quality"] == "human"
    assert data["has_reference"] is True
    assert data["text"] == "Teks berita pertama."
    assert data["reference_summary"] == "Ringkasan pertama."


def test_write_tiered_samples_no_reference(tmp_path):
    """Code-switching samples with no reference summary."""
    from scripts.prepare_data import write_tiered_samples

    fake_samples = [
        {"text": "Mixed bahasa and English sentence."},
    ]

    write_tiered_samples(
        samples=fake_samples,
        output_dir=tmp_path / "cs",
        source="manual",
        language="ms_en",
        ref_quality="none",
        ref_field=None,
        id_prefix="cs",
    )

    files = list((tmp_path / "cs").glob("*.json"))
    assert len(files) == 1

    with open(files[0]) as f:
        data = json.load(f)

    assert data["ref_quality"] == "none"
    assert data["has_reference"] is False
    assert "reference_summary" not in data


def test_nested_tiers_smoke_subset_of_regression(tmp_path):
    """Smoke tier samples must be a subset of regression tier samples."""
    from scripts.prepare_data import allocate_tiers

    all_indices = list(range(100))
    tiers = allocate_tiers(all_indices, smoke=5, regression=30, seed=42)

    # Smoke is a subset of regression
    assert set(tiers["smoke"]).issubset(set(tiers["regression"]))
    # Regression is a subset of benchmark
    assert set(tiers["regression"]).issubset(set(tiers["benchmark"]))
    assert len(tiers["smoke"]) == 5
    assert len(tiers["regression"]) == 30
    assert len(tiers["benchmark"]) == 100


def test_manifest_generation(tmp_path):
    """Verify manifest.json is generated with correct counts."""
    from scripts.prepare_data import write_tiered_manifest

    tier_counts = {
        "smoke": {"ms": 3, "en": 3, "cs": 2},
        "regression": {"ms_human": 30, "ms_mixtral": 30, "en": 25, "cs": 15},
        "benchmark": {"ms_human": 100, "ms_mixtral": 200, "en": 150, "cs": 50},
    }

    write_tiered_manifest(tmp_path, tier_counts)

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["version"] == "2.0.0"
    assert manifest["tiers"]["smoke"]["categories"]["ms"]["count"] == 3
    assert manifest["tiers"]["benchmark"]["categories"]["ms_human"]["count"] == 100
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prepare_data.py -v`
Expected: FAIL — `write_tiered_samples`, `allocate_tiers`, `write_tiered_manifest` not defined

**Step 3: Implement the new functions in `prepare_data.py`**

Add to `scripts/prepare_data.py` (keep existing functions, add new ones):

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prepare_data.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add scripts/prepare_data.py tests/test_prepare_data.py
git commit -m "feat: expand prepare_data.py with multi-source tiered downloads"
```

---

### Task 4: Add `--tier` flag to `evaluate.py`

**Files:**
- Modify: `scripts/evaluate.py`
- Test: `tests/test_evaluate_tiers.py` (create)

**Step 1: Write failing test for tier-aware sample loading**

Create `tests/test_evaluate_tiers.py`:

```python
import json
import pytest
from pathlib import Path
from unittest.mock import patch


def test_load_tiered_samples(tmp_path):
    """Load samples from a specific tier directory."""
    from scripts.evaluate import load_tiered_samples

    # Create smoke tier with 2 samples
    smoke_ms = tmp_path / "smoke" / "ms"
    smoke_ms.mkdir(parents=True)

    for i in range(1, 3):
        sample = {
            "id": f"test_ms_{i:03d}",
            "source": "test",
            "language": "ms",
            "category": "monolingual",
            "has_reference": True,
            "ref_quality": "human",
            "text": f"Sample text {i}",
            "reference_summary": f"Summary {i}",
            "config": {"target_language": "ms", "summary_type": "brief", "max_length": 150},
        }
        with open(smoke_ms / f"{i:03d}.json", "w") as f:
            json.dump(sample, f)

    manifest = {
        "version": "2.0.0",
        "tiers": {
            "smoke": {
                "description": "test",
                "categories": {
                    "ms": {"path": "smoke/ms", "count": 2}
                }
            }
        }
    }
    with open(tmp_path / "manifest.json", "w") as f:
        json.dump(manifest, f)

    samples = load_tiered_samples(tmp_path, manifest, tier="smoke")
    assert len(samples) == 2
    assert samples[0]["ref_quality"] == "human"


def test_load_tiered_samples_all_categories(tmp_path):
    """Load samples from all categories in a tier."""
    from scripts.evaluate import load_tiered_samples

    manifest = {
        "version": "2.0.0",
        "tiers": {
            "regression": {
                "description": "test",
                "categories": {
                    "ms_human": {"path": "regression/ms_human", "count": 1},
                    "en": {"path": "regression/en", "count": 1},
                }
            }
        }
    }

    for cat_path in ["regression/ms_human", "regression/en"]:
        cat_dir = tmp_path / cat_path
        cat_dir.mkdir(parents=True)
        sample = {
            "id": f"test_{cat_path}",
            "source": "test",
            "language": "ms" if "ms" in cat_path else "en",
            "category": "monolingual" if "ms" in cat_path else "english",
            "has_reference": True,
            "ref_quality": "human",
            "text": "Text",
            "reference_summary": "Summary",
            "config": {"target_language": "auto", "summary_type": "brief", "max_length": 150},
        }
        with open(cat_dir / "001.json", "w") as f:
            json.dump(sample, f)

    with open(tmp_path / "manifest.json", "w") as f:
        json.dump(manifest, f)

    samples = load_tiered_samples(tmp_path, manifest, tier="regression")
    assert len(samples) == 2


def test_aggregate_results_splits_by_ref_quality():
    """chrF++ scores should be split by ref_quality."""
    from scripts.evaluate import aggregate_results_tiered

    results = [
        {
            "id": "a", "category": "monolingual", "ref_quality": "human",
            "chrf": 45.0, "latency_ms": 100, "compression_ratio": 0.1,
            "language_match": True, "input_tokens": 100, "output_tokens": 50,
        },
        {
            "id": "b", "category": "monolingual", "ref_quality": "machine",
            "chrf": 35.0, "latency_ms": 200, "compression_ratio": 0.15,
            "language_match": True, "input_tokens": 150, "output_tokens": 60,
        },
    ]

    agg = aggregate_results_tiered(results)

    assert "chrf_human_avg" in agg["monolingual"]
    assert "chrf_machine_avg" in agg["monolingual"]
    assert agg["monolingual"]["chrf_human_avg"] == 45.0
    assert agg["monolingual"]["chrf_machine_avg"] == 35.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluate_tiers.py -v`
Expected: FAIL — `load_tiered_samples`, `aggregate_results_tiered` not defined

**Step 3: Add tier-aware functions to `evaluate.py`**

Add these functions to `scripts/evaluate.py` (keep existing functions):

```python
def load_tiered_samples(data_dir: Path, manifest: dict, tier: str) -> list[dict]:
    """Load all sample JSON files from a specific tier."""
    samples = []
    tier_info = manifest["tiers"][tier]
    for cat_name, cat_info in tier_info["categories"].items():
        cat_dir = data_dir / cat_info["path"]
        if not cat_dir.exists():
            continue
        for json_file in sorted(cat_dir.glob("*.json")):
            with open(json_file) as f:
                sample = json.load(f)
            samples.append(sample)
    return samples


def aggregate_results_tiered(results: list[dict]) -> dict:
    """Aggregate results by category with chrF++ split by ref_quality."""
    categories = {}
    for r in results:
        if "error" in r:
            continue
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    aggregated = {}
    for cat, cat_results in categories.items():
        # Split chrF++ by ref_quality
        human_chrf = [r["chrf"] for r in cat_results if r.get("chrf") is not None and r.get("ref_quality") == "human"]
        machine_chrf = [r["chrf"] for r in cat_results if r.get("chrf") is not None and r.get("ref_quality") == "machine"]
        all_chrf = [r["chrf"] for r in cat_results if r.get("chrf") is not None]
        latencies = [r["latency_ms"] for r in cat_results]
        compressions = [r["compression_ratio"] for r in cat_results]
        lang_matches = [r for r in cat_results if r.get("language_match")]
        total_input = sum(r["input_tokens"] for r in cat_results)
        total_output = sum(r["output_tokens"] for r in cat_results)

        agg = {
            "count": len(cat_results),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
        }

        if human_chrf:
            agg["chrf_human_avg"] = sum(human_chrf) / len(human_chrf)
        if machine_chrf:
            agg["chrf_machine_avg"] = sum(machine_chrf) / len(machine_chrf)
        if all_chrf:
            agg["chrf_avg"] = sum(all_chrf) / len(all_chrf)
            agg["chrf_min"] = min(all_chrf)
            agg["chrf_max"] = max(all_chrf)

        agg["compression_avg"] = sum(compressions) / len(compressions) if compressions else 0
        agg["latency_avg_ms"] = round(sum(latencies) / len(latencies)) if latencies else 0
        agg["language_match"] = f"{len(lang_matches)}/{len(cat_results)}"

        # LLM-as-Judge averages
        eval_results = [r["evaluation"] for r in cat_results if r.get("evaluation")]
        if eval_results:
            dims = ["faithfulness", "coherence", "coverage", "language_quality", "conciseness"]
            agg["judge_scores"] = {}
            for dim in dims:
                scores = [e[dim] for e in eval_results if dim in e]
                if scores:
                    agg["judge_scores"][dim] = round(sum(scores) / len(scores), 2)

        aggregated[cat] = agg

    return aggregated
```

Also update `run_evaluation` to pass `ref_quality` through to results (add after line 127):

```python
            result = {
                "id": sample_id,
                "category": sample["category"],
                "language": sample["language"],
                "has_reference": sample.get("has_reference", True),
                "ref_quality": sample.get("ref_quality", "human"),  # ADD THIS LINE
                "summary": data["summary"],
                # ... rest unchanged
            }
```

Update `parse_args` to add `--tier` flag:

```python
    parser.add_argument(
        "--tier",
        choices=["smoke", "regression", "benchmark"],
        default=None,
        help="Evaluation tier (uses tiered directory structure)",
    )
```

Update `main()` to use tier when specified:

```python
def main():
    args = parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        tier_prefix = f"{args.tier}_" if args.tier else ""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"{tier_prefix}evaluation_{ts}.json"

    print("Loading evaluation samples...")
    manifest = load_manifest()

    if args.tier:
        if "tiers" not in manifest:
            print("Manifest does not have tiered structure. Run scripts/prepare_data.py --all first.")
            sys.exit(1)
        samples = load_tiered_samples(DATA_DIR, manifest, tier=args.tier)
    else:
        samples = load_samples(manifest, category_filter=args.category)

    print(f"Loaded {len(samples)} samples")

    if not samples:
        print("No samples found. Run scripts/prepare_data.py first.")
        sys.exit(1)

    print(f"\nRunning evaluation (LLM-as-Judge: {'ON' if args.evaluate else 'OFF'})...")
    start = time.time()
    results = asyncio.run(run_evaluation(samples, use_judge=args.evaluate))
    total_time = time.time() - start

    if args.tier:
        aggregated = aggregate_results_tiered(results)
    else:
        aggregated = aggregate_results(results)

    print_results(aggregated, results, total_time)
    save_results(results, aggregated, output_path, total_time)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluate_tiers.py -v`
Expected: All 3 tests PASS

**Step 5: Run existing tests to verify no regression**

Run: `pytest tests/ -v`
Expected: All existing tests still PASS

**Step 6: Commit**

```bash
git add scripts/evaluate.py tests/test_evaluate_tiers.py
git commit -m "feat: add --tier flag and ref_quality-split chrF++ to evaluate.py"
```

---

### Task 5: Add baseline comparison

**Files:**
- Modify: `scripts/evaluate.py`
- Test: `tests/test_evaluate_tiers.py` (extend)

**Step 1: Write failing test for baseline save and compare**

Add to `tests/test_evaluate_tiers.py`:

```python
def test_save_and_compare_baseline(tmp_path):
    """Save a baseline and compare against it."""
    from scripts.evaluate import save_baseline, compare_baseline

    baseline_dir = tmp_path / "baselines"

    aggregated = {
        "monolingual": {
            "chrf_avg": 40.0,
            "chrf_human_avg": 45.0,
            "chrf_machine_avg": 35.0,
            "latency_avg_ms": 500,
        }
    }

    save_baseline(aggregated, tier="regression", baseline_dir=baseline_dir)

    # Baseline file exists
    assert (baseline_dir / "regression_baseline.json").exists()

    # Compare: no regression
    new_aggregated = {
        "monolingual": {
            "chrf_avg": 42.0,
            "chrf_human_avg": 46.0,
            "chrf_machine_avg": 36.0,
            "latency_avg_ms": 480,
        }
    }
    diffs = compare_baseline(new_aggregated, tier="regression", baseline_dir=baseline_dir)
    assert len(diffs) == 0  # no regressions

    # Compare: regression detected (>10% drop)
    regressed = {
        "monolingual": {
            "chrf_avg": 34.0,  # -15%
            "chrf_human_avg": 38.0,  # -15.6%
            "chrf_machine_avg": 33.0,
            "latency_avg_ms": 600,
        }
    }
    diffs = compare_baseline(regressed, tier="regression", baseline_dir=baseline_dir)
    assert len(diffs) > 0
    assert any("chrf_avg" in d["metric"] for d in diffs)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate_tiers.py::test_save_and_compare_baseline -v`
Expected: FAIL — `save_baseline`, `compare_baseline` not defined

**Step 3: Implement baseline functions in `evaluate.py`**

Add to `scripts/evaluate.py`:

```python
BASELINE_DIR = Path(__file__).parent.parent / "results" / "baselines"
REGRESSION_THRESHOLD = 0.10  # 10% drop = flag

def save_baseline(aggregated: dict, tier: str, baseline_dir: Path = BASELINE_DIR) -> None:
    """Save current aggregated results as the baseline for a tier."""
    baseline_dir.mkdir(parents=True, exist_ok=True)
    path = baseline_dir / f"{tier}_baseline.json"
    with open(path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Baseline saved to {path}")


def compare_baseline(
    aggregated: dict,
    tier: str,
    baseline_dir: Path = BASELINE_DIR,
    threshold: float = REGRESSION_THRESHOLD,
) -> list[dict]:
    """Compare current results against saved baseline. Returns list of regressions."""
    path = baseline_dir / f"{tier}_baseline.json"
    if not path.exists():
        print("No baseline found. Run with --save-baseline first.")
        return []

    with open(path) as f:
        baseline = json.load(f)

    regressions = []
    metrics_to_check = ["chrf_avg", "chrf_human_avg", "chrf_machine_avg"]

    for cat, agg in aggregated.items():
        if cat not in baseline:
            continue
        base = baseline[cat]
        for metric in metrics_to_check:
            if metric in agg and metric in base and base[metric] > 0:
                pct_change = (agg[metric] - base[metric]) / base[metric]
                if pct_change < -threshold:
                    regressions.append({
                        "category": cat,
                        "metric": metric,
                        "baseline": base[metric],
                        "current": agg[metric],
                        "pct_change": round(pct_change * 100, 1),
                    })

    return regressions
```

Add `--save-baseline` flag to `parse_args`:

```python
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as baseline for future comparison",
    )
```

Update `main()` to use baseline:

```python
    # After aggregation and printing:
    if args.tier and args.save_baseline:
        save_baseline(aggregated, tier=args.tier)
    elif args.tier:
        diffs = compare_baseline(aggregated, tier=args.tier)
        if diffs:
            print("\n!! REGRESSIONS DETECTED !!")
            for d in diffs:
                print(f"  {d['category']}/{d['metric']}: "
                      f"{d['baseline']:.1f} -> {d['current']:.1f} ({d['pct_change']}%)")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluate_tiers.py -v`
Expected: All 4 tests PASS

**Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add scripts/evaluate.py tests/test_evaluate_tiers.py
git commit -m "feat: add baseline save/compare with regression detection"
```

---

### Task 6: Download data and populate tiers

**Files:**
- Modify: `data/evaluation/` (new sample files)

**Step 1: Run the expanded prepare_data.py**

Run: `python scripts/prepare_data.py --all`
Expected: Downloads from all 3 sources, writes samples to smoke/regression/benchmark dirs, updates manifest.json

**Step 2: Verify sample counts**

Run: `find data/evaluation/smoke -name "*.json" | wc -l`
Expected: ~10

Run: `find data/evaluation/regression -name "*.json" | wc -l`
Expected: ~100

Run: `find data/evaluation/benchmark -name "*.json" | wc -l`
Expected: ~500

**Step 3: Verify a sample has correct format**

Run: `python -c "import json; d=json.load(open('data/evaluation/smoke/ms/001.json')); print(d['ref_quality'], d['source'])"`
Expected: `human malay-dataset`

**Step 4: Verify manifest is v2**

Run: `python -c "import json; m=json.load(open('data/evaluation/manifest.json')); print(m['version'], list(m['tiers'].keys()))"`
Expected: `2.0.0 ['smoke', 'regression', 'benchmark']`

**Step 5: Commit**

```bash
git add data/evaluation/
git commit -m "feat: populate tiered evaluation datasets from 3 sources"
```

---

### Task 7: Smoke test the full pipeline

**Files:** None (validation only)

**Step 1: Run smoke tier evaluation (no LLM-as-Judge)**

Run: `python scripts/evaluate.py --tier smoke`
Expected: Runs ~10 samples, prints chrF++ scores split by ref_quality, no errors

**Step 2: Run full test suite to ensure nothing broke**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 3: Save a smoke baseline**

Run: `python scripts/evaluate.py --tier smoke --save-baseline`
Expected: Baseline saved to `results/baselines/smoke_baseline.json`

**Step 4: Commit baseline**

```bash
git add results/baselines/
git commit -m "chore: save initial smoke tier baseline"
```
