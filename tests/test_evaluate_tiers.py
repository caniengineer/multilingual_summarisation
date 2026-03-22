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
