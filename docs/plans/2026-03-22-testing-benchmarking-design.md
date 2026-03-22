# Testing & Benchmarking Plan Design

## Goal

Build a tiered evaluation dataset from multiple Malaysian/English sources to enable regression testing, benchmarking, and stress testing of the summarisation application.

## Data Sources

| Dataset | Language | Reference Quality | Tier Usage |
|---------|----------|-------------------|------------|
| Malay-Dataset (huseinzol05) | Malay | Human-curated news summaries | Smoke + Regression |
| Mesolitica Mixtral | Malay + English | Mixtral-8x7B generated | Regression + Benchmark |
| Malaysian-Dataset (mesolitica) | Malay | Mixed (newer, actively maintained) | Benchmark |
| XLSum English (existing) | English | Human-written (BBC) | Smoke + Regression |
| Code-switching (manual) | ms/en mix | No reference | Smoke + Regression |

## Tiered Structure

Tiers are nested: benchmark ⊃ regression ⊃ smoke.

| Tier | Malay (human) | Malay (Mixtral) | English | Code-switching | Total |
|------|---------------|-----------------|---------|----------------|-------|
| Smoke | 3 | 2 | 3 | 2 | ~10 |
| Regression | 30 | 30 | 25 | 15 | ~100 |
| Benchmark | 100 | 200 | 150 | 50 | ~500 |

## Directory Structure

```
data/evaluation/
├── smoke/
│   ├── ms/
│   ├── en/
│   └── cs/
├── regression/
│   ├── ms_human/
│   ├── ms_mixtral/
│   ├── en/
│   └── cs/
├── benchmark/
│   ├── ms_human/
│   ├── ms_mixtral/
│   ├── en/
│   └── cs/
└── manifest.json
```

## Sample Format

Existing JSON format with `ref_quality` field added:

```json
{
  "id": "malaydata_ms_001",
  "source": "malay-dataset",
  "language": "ms",
  "category": "monolingual",
  "has_reference": true,
  "ref_quality": "human",
  "text": "...",
  "reference_summary": "...",
  "config": {
    "target_language": "ms",
    "summary_type": "brief",
    "max_length": 150
  }
}
```

`ref_quality`: `"human"`, `"machine"`, or `"none"`.

## Evaluation Harness

### Usage

```bash
python scripts/evaluate.py --tier smoke                    # CI, every change
python scripts/evaluate.py --tier regression --evaluate    # before merges
python scripts/evaluate.py --tier benchmark --evaluate     # weekly / on-demand
```

### Metrics by Tier

| Tier | chrF++ | Compression | Language Match | LLM-as-Judge | Latency |
|------|--------|-------------|----------------|--------------|---------|
| Smoke | Yes | Yes | Yes | No | Yes |
| Regression | Yes | Yes | Yes | Yes | Yes |
| Benchmark | Yes | Yes | Yes | Yes | Yes |

### Key Behaviors

- chrF++ scores split by `ref_quality` — human-ref reported separately from machine-ref
- Regression gating: flag if any metric drops >10% vs saved baseline
- Results saved to `results/<tier>_<timestamp>.json`

### Baseline Comparison

```bash
python scripts/evaluate.py --tier regression --evaluate --save-baseline
# Future runs auto-compare against saved baseline
```

## Data Preparation

`scripts/prepare_data.py` expanded to:

1. Download all three dataset sources
2. Filter for quality (minimum text length, non-empty references)
3. Sample with fixed seed (SEED=42) for reproducibility
4. Allocate across nested tiers (smoke ⊂ regression ⊂ benchmark)
5. Write JSON files + manifest.json
6. Code-switching dirs created empty for manual curation

```bash
python scripts/prepare_data.py --all
python scripts/prepare_data.py --tier smoke
```

## Testing

- **Unit tests**: tier flag parsing, ref_quality validation, baseline diff logic, manifest generation
- **Integration tests**: end-to-end smoke tier with mock provider, nested tier validation
- **CI**: unit tests → smoke tier (mock provider) → merge. No real API calls in CI.
