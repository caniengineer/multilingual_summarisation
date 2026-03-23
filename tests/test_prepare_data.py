import json


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
