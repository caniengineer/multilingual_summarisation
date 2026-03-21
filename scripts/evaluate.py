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
