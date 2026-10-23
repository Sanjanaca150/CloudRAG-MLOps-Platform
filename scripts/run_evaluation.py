
"""Evaluate CloudRAG retrieval and answer quality, and log results to MLflow."""

import json
import os
import sys
import time
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    TOP_K,
)
from app.evaluation import keyword_coverage
from app.ollama_client import generate_answer
from app.vector_store import collection_count, search_chunks

DATASET_PATH = PROJECT_ROOT / "evaluation" / "eval_dataset.json"
OUTPUT_DIR = PROJECT_ROOT / "reports"


def find_expected_source_rank(matches: list[dict], expected_source: str) -> int | None:
    """Return the 1-based rank of the first matching source, or None."""
    for rank, item in enumerate(matches, start=1):
        if item["source"] == expected_source:
            return rank
    return None


def main() -> int:
    indexed_count = collection_count()

    if indexed_count == 0:
        print(
            "ERROR: No indexed chunks found. Start the API and call "
            "POST /ingest at http://127.0.0.1:8000/docs first."
        )
        return 1

    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    if not cases:
        print("ERROR: The evaluation dataset is empty.")
        return 1

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("CloudRAG-RAG-Evaluation")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    with mlflow.start_run(run_name="local-ollama-rag-evaluation") as run:
        mlflow.log_params(
            {
                "chat_model": OLLAMA_CHAT_MODEL,
                "embedding_model": OLLAMA_EMBED_MODEL,
                "top_k": TOP_K,
                "retrieval_metric_k": 3,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "evaluation_cases": len(cases),
                "indexed_chunks": indexed_count,
            }
        )

        for case in cases:
            started = time.perf_counter()

            # Retrieve up to three results so Hit@3 is measured consistently.
            matches = search_chunks(case["question"], max(TOP_K, 3))

            # Preserve the configured number of chunks for answer generation.
            answer_matches = matches[:TOP_K]

            context = "\n\n".join(
                f"[Source: {item['source']} | Chunk: {item['chunk_index']}]\n"
                f"{item['text']}"
                for item in answer_matches
            )

            if context:
                answer = generate_answer(case["question"], context)
            else:
                answer = (
                    "I could not find enough evidence in the indexed documents "
                    "to answer that."
                )

            latency_seconds = time.perf_counter() - started

            rank = find_expected_source_rank(
                matches, case["expected_source"]
            )

            source_hit = 1.0 if rank is not None else 0.0
            hit_at_1 = 1.0 if rank == 1 else 0.0
            hit_at_3 = 1.0 if rank is not None and rank <= 3 else 0.0
            reciprocal_rank = 1.0 / rank if rank is not None else 0.0

            answer_coverage = keyword_coverage(
                answer, case["required_keywords"]
            )

            avg_distance = (
                sum(float(item["distance"]) for item in matches)
                / len(matches)
                if matches
                else 1.0
            )

            rows.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "expected_answer": case["expected_answer"],
                    "answer": answer,
                    "expected_source": case["expected_source"],
                    "expected_source_rank": rank,
                    "retrieved_sources_ranked": [
                        item["source"] for item in matches
                    ],
                    "retrieved_sources": ";".join(
                        sorted({item["source"] for item in matches})
                    ),
                    "source_hit": source_hit,
                    "hit_at_1": hit_at_1,
                    "hit_at_3": hit_at_3,
                    "reciprocal_rank": round(reciprocal_rank, 4),
                    "keyword_coverage": round(answer_coverage, 4),
                    "avg_cosine_distance": round(avg_distance, 4),
                    "latency_seconds": round(latency_seconds, 3),
                }
            )

            rank_label = str(rank) if rank is not None else "not found"
            print(
                f"[{case['id']}] "
                f"source_rank={rank_label} "
                f"hit@1={hit_at_1:.0f} "
                f"hit@3={hit_at_3:.0f} "
                f"keyword_coverage={answer_coverage:.2f} "
                f"latency={latency_seconds:.1f}s"
            )

        case_count = len(rows)

        metrics = {
            "source_hit_rate": sum(r["source_hit"] for r in rows) / case_count,
            "retrieval_hit_at_1": sum(r["hit_at_1"] for r in rows) / case_count,
            "retrieval_hit_at_3": sum(r["hit_at_3"] for r in rows) / case_count,
            "retrieval_mrr": sum(r["reciprocal_rank"] for r in rows) / case_count,
            "mean_keyword_coverage": (
                sum(r["keyword_coverage"] for r in rows) / case_count
            ),
            "mean_latency_seconds": (
                sum(r["latency_seconds"] for r in rows) / case_count
            ),
            "mean_retrieval_cosine_distance": (
                sum(r["avg_cosine_distance"] for r in rows) / case_count
            ),
        }

        mlflow.log_metrics(metrics)

        report_path = OUTPUT_DIR / "evaluation_results.json"
        report_path.write_text(
            json.dumps(
                {
                    "run_id": run.info.run_id,
                    "tracking_uri": tracking_uri,
                    "metrics": metrics,
                    "results": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(report_path))

    print("\nEvaluation summary")
    print(f"Cases: {case_count}")
    print(f"Expected-source hit rate: {metrics['source_hit_rate']:.1%}")
    print(f"Retrieval Hit@1: {metrics['retrieval_hit_at_1']:.1%}")
    print(f"Retrieval Hit@3: {metrics['retrieval_hit_at_3']:.1%}")
    print(f"Mean reciprocal rank (MRR): {metrics['retrieval_mrr']:.4f}")
    print(f"Mean answer keyword coverage: {metrics['mean_keyword_coverage']:.1%}")
    print(f"Mean latency: {metrics['mean_latency_seconds']:.2f} seconds")
    print(
        "Mean retrieval cosine distance: "
        f"{metrics['mean_retrieval_cosine_distance']:.4f} "
        "(lower is generally closer)"
    )
    print(f"Report: {report_path}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(
        "Start the MLflow UI with: "
        f'python -m mlflow ui --backend-store-uri "{tracking_uri}" '
        "--host 127.0.0.1 --port 5000"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
