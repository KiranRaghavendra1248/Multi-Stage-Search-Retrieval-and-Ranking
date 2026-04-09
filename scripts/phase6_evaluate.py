"""
Phase 5: Full Evaluation

Runs all 6 pipeline variants on the MS MARCO dev set and prints a comparison table:
  | Variant                            | MRR@10 | Recall@100 | Latency(ms) |

Variants:
  1. BM25 baseline
  2. Bi-encoder only
  3. Pipeline A: Bi-encoder + ColBERT
  4. Pipeline B: Bi-encoder + Cross-Encoder
  5. Pipeline A + Query Rewriting
  6. Pipeline B + Query Rewriting
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.evaluation.compare import run_comparison, print_comparison_table

logger = get_logger("phase5")


def main():
    cfg = load_config()
    logger.info("Environment: %s", cfg.environment)
    logger.info("Starting full evaluation...")

    results = run_comparison(cfg)
    table = print_comparison_table(results)

    # Save table to file
    out_path = Path("data/evaluation_results.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write("```\n")
        f.write(table)
        f.write("\n```\n")
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
