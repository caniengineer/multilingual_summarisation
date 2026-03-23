"""Examples for calling the /v1/summarize API.

Usage:
    # Start the server first:
    make dev

    # Then run this script:
    uv run python examples/call_api.py
"""

import base64
import json
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"


def summarize_text():
    """Summarize a plain text document."""
    resp = httpx.post(
        f"{BASE_URL}/v1/summarize",
        json={
            "document": (
                "Malaysia's Budget 2026 focuses on strengthening the MADANI economy. "
                "The government has allocated RM421 billion for national development, "
                "with emphasis on digital transformation, green energy, and rakyat welfare. "
                "Key initiatives include expanding 5G coverage, increasing minimum wage, "
                "and providing targeted subsidies for B40 households."
            ),
            "document_type": "txt",
            "config": {
                "target_language": "en",
                "summary_type": "brief",
                "max_length": 100,
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    print("=== Text Summarization ===")
    print(f"Summary: {data['summary']}")
    print(f"Language: {data['metadata']['detected_language']}")
    print(f"Latency: {data['metadata']['latency_ms']}ms")
    print()


def summarize_pdf(pdf_path: str):
    """Summarize a PDF document by sending base64-encoded bytes."""
    path = Path(pdf_path)
    if not path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    b64 = base64.b64encode(path.read_bytes()).decode("ascii")

    resp = httpx.post(
        f"{BASE_URL}/v1/summarize",
        json={
            "document": b64,
            "document_type": "pdf",
            "config": {
                "target_language": "en",
                "summary_type": "executive",
                "max_length": 300,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"=== PDF Summarization: {path.name} ===")
    print(f"Summary: {data['summary']}")
    print(f"Language: {data['metadata']['detected_language']}")
    print(f"Code-switching: {data['metadata']['code_switching_detected']}")
    print(f"Latency: {data['metadata']['latency_ms']}ms")
    print()


def summarize_with_evaluation():
    """Summarize with built-in quality evaluation."""
    resp = httpx.post(
        f"{BASE_URL}/v1/summarize",
        json={
            "document": (
                "Bank Negara Malaysia mengekalkan Kadar Dasar Semalaman (OPR) "
                "pada 3.00 peratus. Keputusan ini mengambil kira pertumbuhan "
                "ekonomi domestik yang stabil serta tekanan inflasi yang terkawal. "
                "Majlis Dasar Monetari akan terus memantau perkembangan ekonomi "
                "global dan domestik."
            ),
            "config": {
                "target_language": "auto",
                "summary_type": "brief",
                "evaluate": True,
            },
        },
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    print("=== Summarization with Evaluation ===")
    print(f"Summary: {data['summary']}")
    print(f"Language: {data['metadata']['detected_language']}")
    if data["metadata"]["evaluation"]:
        eval_scores = data["metadata"]["evaluation"]
        print(f"Faithfulness: {eval_scores['faithfulness']}/5")
        print(f"Coherence: {eval_scores['coherence']}/5")
        print(f"Coverage: {eval_scores['coverage']}/5")
        print(f"Justification: {eval_scores['justification']}")
    print()


def health_check():
    """Check API health."""
    resp = httpx.get(f"{BASE_URL}/v1/health")
    resp.raise_for_status()
    print(f"=== Health Check ===")
    print(json.dumps(resp.json(), indent=2))
    print()


if __name__ == "__main__":
    health_check()
    summarize_text()
    summarize_with_evaluation()

    # Uncomment to test with a real PDF:
    summarize_pdf("tests/fixtures/pdf/budget_speech_2026_en.pdf")
