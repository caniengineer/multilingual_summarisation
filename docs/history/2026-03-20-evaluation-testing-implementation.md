# Evaluation & Testing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a runnable evaluation script that tests the multilingual summarization API against ~25 curated samples from XL-Sum and code-switching sources, computing chrF++ and optional LLM-as-Judge scores.

**Architecture:** A data preparation script downloads XL-Sum samples and writes them as JSON files. A standalone evaluation script loads these samples, runs them through the FastAPI app in-process (via ASGITransport), computes chrF++ against reference summaries, and outputs a results table + JSON file.

**Tech Stack:** Python 3.11+, sacrebleu (chrF++), datasets (HuggingFace, prep only), httpx (ASGITransport), existing FastAPI app

**Prerequisites:** The MVP app must be fully implemented (Tasks 1-9 from the lean MVP implementation plan) before this plan can execute. The evaluation script imports `create_app` from `app.main`.

---

### Task 1: Add Evaluation Dependencies

**Files:**
- Modify: `pyproject.toml:15-21`

**Step 1: Add sacrebleu and datasets to dev dependencies**

Add `sacrebleu` and `datasets` to the `[project.optional-dependencies] dev` section in `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "pytest-httpx>=0.30.0",
    "ruff>=0.6.0",
    "sacrebleu>=2.4.0",
    "datasets>=2.0",
]
```

**Step 2: Add results/ to .gitignore**

Append to `.gitignore`:

```
# Evaluation results
results/
```

**Step 3: Create results directory with .gitkeep**

```bash
mkdir -p results
touch results/.gitkeep
```

**Step 4: Install updated dependencies**

Run: `pip install -e ".[dev]"`
Expected: Successful install including sacrebleu and datasets.

**Step 5: Verify sacrebleu is available**

Run: `python -c "import sacrebleu; print(sacrebleu.__version__)"`
Expected: Prints version number, no errors.

**Step 6: Commit**

```bash
git add pyproject.toml .gitignore results/.gitkeep
git commit -m "chore: add sacrebleu and datasets to dev dependencies"
```

---

### Task 2: Data Preparation Script

**Files:**
- Create: `scripts/prepare_data.py`

**Step 1: Create the data preparation script**

```python
# scripts/prepare_data.py
"""One-off script to download XL-Sum samples and write evaluation data files.

Usage:
    python scripts/prepare_data.py

Downloads Malay and English samples from XL-Sum (HuggingFace),
writes them as individual JSON files under data/evaluation/,
and generates a manifest.json index.

Run once, commit the output, never run again.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
DATA_DIR = Path(__file__).parent.parent / "data" / "evaluation"
MS_COUNT = 12
EN_COUNT = 5


def download_xlsum_samples():
    """Download XL-Sum test split for Malay and English."""
    print("Downloading XL-Sum dataset (this may take a moment)...")

    ms_dataset = load_dataset("csebuetnlp/xlsum", "malay", split="test", trust_remote_code=True)
    en_dataset = load_dataset("csebuetnlp/xlsum", "english", split="test", trust_remote_code=True)

    rng = random.Random(SEED)

    ms_indices = rng.sample(range(len(ms_dataset)), min(MS_COUNT, len(ms_dataset)))
    en_indices = rng.sample(range(len(en_dataset)), min(EN_COUNT, len(en_dataset)))

    ms_samples = [ms_dataset[i] for i in ms_indices]
    en_samples = [en_dataset[i] for i in en_indices]

    return ms_samples, en_samples


def write_samples(samples, category, language, target_language, source_name, output_dir):
    """Write sample JSON files to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples, start=1):
        sample_id = f"{source_name}_{language}_{i:03d}"
        data = {
            "id": sample_id,
            "source": source_name,
            "language": language,
            "category": category,
            "has_reference": True,
            "text": sample["text"],
            "reference_summary": sample["summary"],
            "config": {
                "target_language": target_language,
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

    ms_samples, en_samples = download_xlsum_samples()

    print(f"\nWriting {len(ms_samples)} Malay samples...")
    write_samples(
        ms_samples,
        category="monolingual",
        language="ms",
        target_language="ms",
        source_name="xlsum",
        output_dir=DATA_DIR / "xlsum_ms",
    )

    print(f"\nWriting {len(en_samples)} English samples...")
    write_samples(
        en_samples,
        category="english",
        language="en",
        target_language="en",
        source_name="xlsum",
        output_dir=DATA_DIR / "english",
    )

    # Create empty codeswitching directory (populated manually in Task 3)
    cs_dir = DATA_DIR / "codeswitching"
    cs_dir.mkdir(parents=True, exist_ok=True)

    write_manifest()
    print("\nDone. Commit data/evaluation/ to the repo.")


if __name__ == "__main__":
    main()
```

**Step 2: Run the script**

Run: `python scripts/prepare_data.py`
Expected: Downloads XL-Sum, writes 12 Malay + 5 English JSON files under `data/evaluation/`, prints file paths.

**Step 3: Verify output structure**

Run: `find data/evaluation -type f | sort`
Expected:
```
data/evaluation/english/001.json
data/evaluation/english/002.json
data/evaluation/english/003.json
data/evaluation/english/004.json
data/evaluation/english/005.json
data/evaluation/manifest.json
data/evaluation/xlsum_ms/001.json
...
data/evaluation/xlsum_ms/012.json
```

**Step 4: Verify a sample file has expected structure**

Run: `python -c "import json; d=json.load(open('data/evaluation/xlsum_ms/001.json')); print(d['id'], d['language'], len(d['text']), 'ref:', len(d['reference_summary']))"`
Expected: Prints sample ID, language "ms", text length, and reference summary length.

**Step 5: Commit**

```bash
git add scripts/prepare_data.py data/evaluation/
git commit -m "feat: add data preparation script and XL-Sum evaluation samples"
```

---

### Task 3: Code-Switching Samples

**Files:**
- Create: `data/evaluation/codeswitching/001.json` through `008.json`
- Modify: `data/evaluation/manifest.json`

**Step 1: Create 8 code-switching sample files**

These are manually curated. 5 have reference summaries, 3 don't (marked `has_reference: false`).

```python
# Helper: run this to generate the files
# python scripts/create_cs_samples.py
# Or create them manually — the content below is what matters.
```

Create `data/evaluation/codeswitching/001.json`:
```json
{
  "id": "cs_001",
  "source": "manual",
  "language": "ms_en",
  "category": "codeswitching",
  "has_reference": true,
  "text": "Kerajaan Malaysia telah announce new digital economy blueprint semalam. Menteri kata investment dalam AI dan cloud computing akan create lebih banyak job opportunities untuk rakyat. \"We need to upskill our workforce,\" beliau tambah. Banyak company dah start implement automation, tapi masih lack of talent dalam data science dan machine learning. Government akan allocate RM500 juta untuk training programmes tahun depan.",
  "reference_summary": "Kerajaan Malaysia mengumumkan pelan ekonomi digital baharu dengan pelaburan dalam AI dan cloud computing untuk mewujudkan peluang pekerjaan, dengan peruntukan RM500 juta untuk program latihan kemahiran.",
  "config": {
    "target_language": "auto",
    "summary_type": "brief",
    "max_length": 100
  }
}
```

Create `data/evaluation/codeswitching/002.json`:
```json
{
  "id": "cs_002",
  "source": "manual",
  "language": "ms_en",
  "category": "codeswitching",
  "has_reference": true,
  "text": "Meeting tadi discuss pasal system migration dari on-premise ke cloud. Team lead suggest kita guna AWS sebab pricing dia more competitive. Tapi ada concern pasal data sovereignty — undang-undang Malaysia require certain data kena store locally. CTO kata kita boleh consider hybrid approach, keep sensitive data on local servers tapi leverage cloud untuk processing. Timeline dia agak tight, kena siap before Q3.",
  "reference_summary": "Mesyuarat membincangkan migrasi sistem ke cloud dengan AWS sebagai pilihan, tetapi terdapat kebimbangan mengenai kedaulatan data. CTO mencadangkan pendekatan hibrid dengan data sensitif kekal di pelayan tempatan. Projek perlu siap sebelum Q3.",
  "config": {
    "target_language": "auto",
    "summary_type": "brief",
    "max_length": 100
  }
}
```

Create `data/evaluation/codeswitching/003.json`:
```json
{
  "id": "cs_003",
  "source": "manual",
  "language": "ms_en",
  "category": "codeswitching",
  "has_reference": true,
  "text": "Startup scene kat Malaysia memang growing rapidly. Banyak founder muda yang graduate dari local uni dah start their own companies. Ecosystem support pun dah improve — ada MDEC, Cradle Fund, dan macam-macam accelerator programme. Last year alone, Malaysian startups raised more than USD 500 million in funding. Tapi challenge dia still ada — talent retention jadi issue sebab banyak engineer prefer kerja Singapore for higher salary.",
  "reference_summary": "Ekosistem startup Malaysia berkembang pesat dengan sokongan MDEC dan Cradle Fund. Startup tempatan mengumpul lebih USD 500 juta tahun lepas, namun pengekalan bakat kekal mencabar kerana ramai jurutera memilih bekerja di Singapura.",
  "config": {
    "target_language": "auto",
    "summary_type": "brief",
    "max_length": 100
  }
}
```

Create `data/evaluation/codeswitching/004.json`:
```json
{
  "id": "cs_004",
  "source": "manual",
  "language": "ms_en",
  "category": "codeswitching",
  "has_reference": true,
  "text": "Hari ni kita launch product baru — AI-powered chatbot untuk customer service bank. System ni boleh handle query dalam Bahasa Melayu, English, dan even Mandarin. Response time dia under 2 seconds, much faster than human agents. Tapi untuk complex cases macam fraud detection atau loan disputes, chatbot akan escalate ke human agent. Privacy compliance pun dah handle — semua conversation encrypted end-to-end dan data retention follow Bank Negara guidelines.",
  "reference_summary": "Chatbot AI baharu untuk perkhidmatan pelanggan bank dilancarkan hari ini, mampu mengendalikan pertanyaan dalam tiga bahasa dengan masa respons bawah 2 saat. Kes kompleks akan dieskalasi kepada ejen manusia, dengan pematuhan privasi mengikut garis panduan Bank Negara.",
  "config": {
    "target_language": "auto",
    "summary_type": "brief",
    "max_length": 100
  }
}
```

Create `data/evaluation/codeswitching/005.json`:
```json
{
  "id": "cs_005",
  "source": "manual",
  "language": "ms_en",
  "category": "codeswitching",
  "has_reference": true,
  "text": "Research paper dari UM dan YTL AI Labs baru je publish kat EMNLP pasal MalayMMLU benchmark. Dataset ni contain 24,000 soalan multiple-choice dari kurikulum pendidikan Malaysia. Result dia interesting — ILMU model score 86.98% accuracy, beating GPT-4o yang dapat 84.98%. Ni first time Malaysian sovereign AI model outperform international model on local benchmark. Implications dia besar untuk development of Malay-centric AI applications.",
  "reference_summary": "Kertas penyelidikan UM-YTL AI Labs mengenai penanda aras MalayMMLU menunjukkan model ILMU mencapai ketepatan 86.98%, mengatasi GPT-4o pada 84.98%, menjadi model AI tempatan pertama yang mengatasi model antarabangsa.",
  "config": {
    "target_language": "auto",
    "summary_type": "brief",
    "max_length": 100
  }
}
```

Create `data/evaluation/codeswitching/006.json`:
```json
{
  "id": "cs_006",
  "source": "manual",
  "language": "ms_en",
  "category": "codeswitching",
  "has_reference": false,
  "text": "Bro, kau dah try guna new payment system tak? Aku try register semalam tapi keep getting error. Support team kata server maintenance, tapi dah 3 hari still down. Macam mana nak process payment kalau system tak function? Customer dah start complain, boss pun dah pressure. IT team cakap they're working on it but no ETA. Honestly, I think we should consider backup plan — maybe temporarily switch back to manual processing.",
  "reference_summary": null,
  "config": {
    "target_language": "auto",
    "summary_type": "brief",
    "max_length": 80
  }
}
```

Create `data/evaluation/codeswitching/007.json`:
```json
{
  "id": "cs_007",
  "source": "manual",
  "language": "ms_en",
  "category": "codeswitching",
  "has_reference": false,
  "text": "Projek highway baru dari KL ke Seremban dah 60% complete. Contractor kata akan siap ahead of schedule kalau weather permits. Total cost dia approximately RM2.3 billion, funded through public-private partnership. Environmental groups ada raise concern pasal deforestation tapi EIA report dah approved last month. Once completed, travel time will reduce from 1.5 hours to roughly 45 minutes during peak hours.",
  "reference_summary": null,
  "config": {
    "target_language": "auto",
    "summary_type": "brief",
    "max_length": 80
  }
}
```

Create `data/evaluation/codeswitching/008.json`:
```json
{
  "id": "cs_008",
  "source": "manual",
  "language": "ms_en",
  "category": "codeswitching",
  "has_reference": false,
  "text": "Universiti tempatan kena adapt dengan AI revolution ni. Banyak course masih teach outdated syllabus — student graduate tapi skill tak match industry requirement. Ada professor suggest integrate AI tools dalam curriculum, let students use ChatGPT untuk assignment tapi with proper citation. Some faculty members resist sebab takut academic integrity compromised. Dean kata kita perlu find balance between embracing technology dan maintaining educational standards.",
  "reference_summary": null,
  "config": {
    "target_language": "auto",
    "summary_type": "brief",
    "max_length": 80
  }
}
```

**Step 2: Update manifest.json**

Update `data/evaluation/manifest.json` to set codeswitching count:

```json
{
  "version": "1.0.0",
  "categories": {
    "monolingual": {"path": "xlsum_ms", "count": 12},
    "codeswitching": {"path": "codeswitching", "count": 8},
    "english": {"path": "english", "count": 5}
  }
}
```

**Step 3: Verify sample structure**

Run: `python -c "import json; d=json.load(open('data/evaluation/codeswitching/001.json')); print(d['id'], d['category'], d['has_reference'])"`
Expected: `cs_001 codeswitching True`

Run: `python -c "import json; d=json.load(open('data/evaluation/codeswitching/006.json')); print(d['id'], d['has_reference'])"`
Expected: `cs_006 False`

**Step 4: Commit**

```bash
git add data/evaluation/codeswitching/ data/evaluation/manifest.json
git commit -m "feat: add code-switching evaluation samples (5 with references, 3 without)"
```

---

### Task 4: Evaluation Script — Core Structure

**Files:**
- Create: `scripts/evaluate.py`

**Step 1: Create the evaluation script with argument parsing and sample loading**

```python
# scripts/evaluate.py
"""Evaluation script for the multilingual summarization API.

Loads curated samples from data/evaluation/, runs them through the API
in-process, computes chrF++ against reference summaries, and optionally
runs LLM-as-Judge evaluation.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --evaluate
    python scripts/evaluate.py --category monolingual
    python scripts/evaluate.py --output results/my_run.json
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import sacrebleu

# Add project root to path so we can import the app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import create_app


DATA_DIR = Path(__file__).parent.parent / "data" / "evaluation"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_manifest() -> dict:
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


def load_samples(manifest: dict, category_filter: str = "all") -> list[dict]:
    """Load all sample JSON files based on manifest and optional category filter."""
    samples = []
    for cat_name, cat_info in manifest["categories"].items():
        if category_filter != "all" and cat_name != category_filter:
            continue
        cat_dir = DATA_DIR / cat_info["path"]
        if not cat_dir.exists():
            continue
        for json_file in sorted(cat_dir.glob("*.json")):
            with open(json_file) as f:
                sample = json.load(f)
            samples.append(sample)
    return samples


def compute_chrf(hypothesis: str, reference: str) -> float:
    """Compute chrF++ score between hypothesis and reference."""
    score = sacrebleu.sentence_chrf(hypothesis, [reference])
    return score.score


def compute_compression_ratio(source: str, summary: str) -> float:
    """Compute compression ratio (summary length / source length)."""
    if len(source) == 0:
        return 0.0
    return len(summary) / len(source)


async def run_evaluation(samples: list[dict], use_judge: bool) -> list[dict]:
    """Run all samples through the API and collect results."""
    app = create_app()
    results = []

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://eval",
        timeout=60.0,
    ) as client:
        for i, sample in enumerate(samples, start=1):
            sample_id = sample["id"]
            print(f"  [{i}/{len(samples)}] {sample_id}...", end=" ", flush=True)

            config = dict(sample["config"])
            if use_judge:
                config["evaluate"] = True

            start = time.time()
            resp = await client.post(
                "/v1/summarize",
                json={
                    "document": sample["text"],
                    "document_type": "txt",
                    "config": config,
                },
            )
            elapsed_ms = int((time.time() - start) * 1000)

            if resp.status_code != 200:
                print(f"FAILED ({resp.status_code})")
                results.append({
                    "id": sample_id,
                    "category": sample["category"],
                    "error": resp.text,
                })
                continue

            data = resp.json()
            result = {
                "id": sample_id,
                "category": sample["category"],
                "language": sample["language"],
                "has_reference": sample.get("has_reference", True),
                "summary": data["summary"],
                "detected_language": data["metadata"]["detected_language"],
                "code_switching_detected": data["metadata"]["code_switching_detected"],
                "model_used": data["metadata"]["model_used"],
                "input_tokens": data["metadata"]["input_tokens"],
                "output_tokens": data["metadata"]["output_tokens"],
                "latency_ms": data["metadata"]["latency_ms"],
                "client_elapsed_ms": elapsed_ms,
                "compression_ratio": compute_compression_ratio(
                    sample["text"], data["summary"]
                ),
            }

            # chrF++ only if reference exists
            if sample.get("has_reference") and sample.get("reference_summary"):
                result["chrf"] = compute_chrf(
                    data["summary"], sample["reference_summary"]
                )
            else:
                result["chrf"] = None

            # LLM-as-Judge scores
            if data["metadata"].get("evaluation"):
                result["evaluation"] = data["metadata"]["evaluation"]
            else:
                result["evaluation"] = None

            # Language match check
            expected_lang = sample["language"]
            if expected_lang == "ms_en":
                # Code-switching: accept either ms or en as detected
                result["language_match"] = True
                result["cs_detected"] = data["metadata"]["code_switching_detected"]
            else:
                result["language_match"] = (
                    data["metadata"]["detected_language"] == expected_lang
                )
                result["cs_detected"] = None

            print(f"OK (chrF++={result['chrf']:.1f})" if result["chrf"] is not None else "OK (no ref)")
            results.append(result)

    return results


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate results by category."""
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
        chrf_scores = [r["chrf"] for r in cat_results if r["chrf"] is not None]
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

        if chrf_scores:
            agg["chrf_avg"] = sum(chrf_scores) / len(chrf_scores)
            agg["chrf_min"] = min(chrf_scores)
            agg["chrf_max"] = max(chrf_scores)

        agg["compression_avg"] = sum(compressions) / len(compressions) if compressions else 0
        agg["latency_avg_ms"] = sum(latencies) // len(latencies) if latencies else 0
        agg["language_match"] = f"{len(lang_matches)}/{len(cat_results)}"

        # Code-switching detection rate (only for codeswitching category)
        if cat == "codeswitching":
            cs_detected = [r for r in cat_results if r.get("cs_detected")]
            agg["cs_detection_rate"] = f"{len(cs_detected)}/{len(cat_results)}"

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


def print_results(aggregated: dict, results: list[dict], total_time_s: float):
    """Print formatted results table to terminal."""
    category_labels = {
        "monolingual": "Malay Monolingual",
        "codeswitching": "Code-Switching",
        "english": "English",
    }

    print("\n" + "=" * 50)
    print("  Multilingual Summarization Evaluation")
    print("=" * 50)

    for cat, agg in aggregated.items():
        label = category_labels.get(cat, cat)
        print(f"\nCategory: {label} ({agg['count']} samples)")

        if "chrf_avg" in agg:
            print(f"  chrF++:     avg {agg['chrf_avg']:.1f}  "
                  f"min {agg['chrf_min']:.1f}  max {agg['chrf_max']:.1f}")

        print(f"  Compress:   avg {agg['compression_avg']:.2f}")
        print(f"  Lang match: {agg['language_match']}")

        if "cs_detection_rate" in agg:
            print(f"  CS detect:  {agg['cs_detection_rate']}")

        print(f"  Latency:    avg {agg['latency_avg_ms']}ms")

        if "judge_scores" in agg:
            scores = agg["judge_scores"]
            score_str = "  ".join(f"{k}={v}" for k, v in scores.items())
            print(f"  Judge:      {score_str}")

    # Overall
    all_results = [r for r in results if "error" not in r]
    total_input = sum(r["input_tokens"] for r in all_results)
    total_output = sum(r["output_tokens"] for r in all_results)
    all_chrf = [r["chrf"] for r in all_results if r["chrf"] is not None]
    errors = [r for r in results if "error" in r]

    print(f"\n{'=' * 50}")
    print("  Overall")
    print(f"{'=' * 50}")
    if all_chrf:
        print(f"  chrF++ avg: {sum(all_chrf) / len(all_chrf):.1f}")
    print(f"  Total tokens: {total_input:,} in / {total_output:,} out")
    print(f"  Total time: {total_time_s:.0f}s")
    if errors:
        print(f"  Errors: {len(errors)}")
    print()


def save_results(results: list[dict], aggregated: dict, output_path: Path, total_time_s: float):
    """Save full results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(results),
        "total_time_s": round(total_time_s, 1),
        "aggregated": aggregated,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multilingual summarization API")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Include LLM-as-Judge evaluation (doubles API cost)",
    )
    parser.add_argument(
        "--category",
        choices=["monolingual", "codeswitching", "english", "all"],
        default="all",
        help="Filter samples by category (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/evaluation_<timestamp>.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"evaluation_{ts}.json"

    print("Loading evaluation samples...")
    manifest = load_manifest()
    samples = load_samples(manifest, category_filter=args.category)
    print(f"Loaded {len(samples)} samples")

    if not samples:
        print("No samples found. Run scripts/prepare_data.py first.")
        sys.exit(1)

    print(f"\nRunning evaluation (LLM-as-Judge: {'ON' if args.evaluate else 'OFF'})...")
    start = time.time()
    results = asyncio.run(run_evaluation(samples, use_judge=args.evaluate))
    total_time = time.time() - start

    aggregated = aggregate_results(results)
    print_results(aggregated, results, total_time)
    save_results(results, aggregated, output_path, total_time)


if __name__ == "__main__":
    main()
```

**Step 2: Verify the script loads and parses args**

Run: `python scripts/evaluate.py --help`
Expected: Prints usage with `--evaluate`, `--category`, `--output` flags.

**Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: add evaluation script with chrF++ and LLM-as-Judge support"
```

---

### Task 5: Smoke Test with ILMU Stub

**Step 1: Run evaluation against the ILMU stub provider (no API key needed)**

Run: `ANTHROPIC_API_KEY=sk-test PROVIDER=ilmu python scripts/evaluate.py --category monolingual`

Expected: Runs all 12 Malay samples through the stub provider, prints chrF++ scores (will be low since stub returns placeholder text), latency, compression ratios. No API calls made. Should complete in under 5 seconds.

**Step 2: Verify JSON output was saved**

Run: `ls results/evaluation_*.json`
Expected: One JSON file exists.

Run: `python -c "import json; d=json.load(open(sorted(__import__('pathlib').Path('results').glob('evaluation_*.json'))[-1])); print(f'Samples: {d[\"total_samples\"]}, Time: {d[\"total_time_s\"]}s')"`
Expected: Prints sample count and time.

**Step 3: Run all categories**

Run: `ANTHROPIC_API_KEY=sk-test PROVIDER=ilmu python scripts/evaluate.py`
Expected: Runs all 25 samples through stub, prints results table for all 3 categories. chrF++ scores will be low (stub text vs real references) but the pipeline works end-to-end.

**Step 4: No commit needed — this is a verification step**

---

### Task 6: Final Verification & Commit

**Step 1: Run linter on new files**

Run: `ruff check scripts/`
Expected: No errors (fix any that appear).

**Step 2: Verify project structure**

Run: `find scripts data/evaluation -type f | sort`
Expected:
```
data/evaluation/codeswitching/001.json
data/evaluation/codeswitching/002.json
data/evaluation/codeswitching/003.json
data/evaluation/codeswitching/004.json
data/evaluation/codeswitching/005.json
data/evaluation/codeswitching/006.json
data/evaluation/codeswitching/007.json
data/evaluation/codeswitching/008.json
data/evaluation/english/001.json
...
data/evaluation/english/005.json
data/evaluation/manifest.json
data/evaluation/xlsum_ms/001.json
...
data/evaluation/xlsum_ms/012.json
scripts/evaluate.py
scripts/prepare_data.py
```

**Step 3: Verify existing tests still pass**

Run: `pytest -v --tb=short`
Expected: All existing tests pass (no regressions).

**Step 4: Commit any lint fixes**

```bash
git add -A
git commit -m "chore: lint fixes for evaluation scripts"
```
