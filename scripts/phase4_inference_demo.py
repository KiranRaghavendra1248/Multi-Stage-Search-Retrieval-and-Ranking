"""
Phase 4: Interactive Inference Demo

Interactive CLI demonstrating the full pipeline with timing at each stage.
Choose Pipeline A (ColBERT re-ranker) or Pipeline B (Cross-Encoder re-ranker).

Usage:
    python scripts/phase4_inference_demo.py --pipeline A
    python scripts/phase4_inference_demo.py --pipeline B --rewrite
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.inference.query_processor import process_query
from src.inference.hyde import generate_hypothetical_doc
from src.inference.stage1_dense import DenseRetriever

logger = get_logger("phase4")


def run_pipeline(query: str, pipeline: str, cfg) -> None:
    from src.inference.stage2_colbert import ColBERTReranker
    from src.inference.stage2_crossencoder import CrossEncoderReranker

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Pipeline: {pipeline} | Query rewriting: {cfg.inference.query_rewriting}")
    print("=" * 60)

    # Query processing
    t0 = time.perf_counter()
    processed_q = process_query(query, cfg)
    if processed_q != query:
        print(f"\n[Query Rewriting] {query!r} → {processed_q!r}  ({(time.perf_counter()-t0)*1000:.0f}ms)")

    # HyDE
    t0 = time.perf_counter()
    hyde_doc = generate_hypothetical_doc(processed_q, cfg)
    print(f"\n[HyDE] ({(time.perf_counter()-t0)*1000:.0f}ms)")
    print(f"  Hypothetical: {hyde_doc[:150]}...")

    # Stage 1
    retriever = DenseRetriever.from_config(cfg)
    retriever.load()
    t0 = time.perf_counter()
    stage1_results = retriever.retrieve(hyde_doc, top_k=1000)
    print(f"\n[Stage 1 - Dense Retrieval] Top-1000 ({(time.perf_counter()-t0)*1000:.0f}ms)")
    print(f"  Top result: {stage1_results[0]['passage'][:120]}...")

    # Stage 2
    if pipeline == "A":
        reranker = ColBERTReranker(cfg)
        t0 = time.perf_counter()
        final = reranker.rerank(query, stage1_results)
        label = "ColBERT"
    else:
        reranker = CrossEncoderReranker(cfg)
        t0 = time.perf_counter()
        final = reranker.rerank(query, stage1_results)
        label = "Cross-Encoder"

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n[Stage 2 - {label}] Re-ranked ({elapsed:.0f}ms)")
    print(f"\nFinal Results:")
    for i, r in enumerate(final[:10], start=1):
        print(f"  {i:2d}. [score: {r['score']:.4f}] {r['passage'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Interactive retrieval demo")
    parser.add_argument("--pipeline", choices=["A", "B"], default="A",
                        help="A=ColBERT re-ranker, B=Cross-Encoder re-ranker")
    parser.add_argument("--rewrite", action="store_true",
                        help="Enable query rewriting (spell check + synonym expansion)")
    args = parser.parse_args()

    cfg = load_config()
    cfg.inference.query_rewriting = args.rewrite

    print(f"Pipeline {args.pipeline} | Query rewriting: {args.rewrite}")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        run_pipeline(query, args.pipeline, cfg)


if __name__ == "__main__":
    main()
