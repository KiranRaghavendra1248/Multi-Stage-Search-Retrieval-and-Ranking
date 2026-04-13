"""
Phase 6: Full Evaluation

Runs all 7 pipeline variants on the MS MARCO dev set and prints a comparison table:
  | Variant                            | MRR@10 | Recall@100 | Latency(ms) |

Variants:
  1. BM25 baseline
  2. Pre-trained MS MARCO bi-encoder (no fine-tuning)
  3. Fine-tuned bi-encoder only
  4. Pipeline A: Fine-tuned bi-encoder + ColBERT
  5. Pipeline B: Fine-tuned bi-encoder + Cross-Encoder
  6. Pipeline A + Query Rewriting
  7. Pipeline B + Query Rewriting
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.evaluation.compare import run_comparison, print_comparison_table

logger = get_logger("phase6")


def main():
    cfg = load_config()
    logger.info("Environment: %s", cfg.environment)
    logger.info("Starting full evaluation...")

    results = run_comparison(cfg)
    table = print_comparison_table(results)

    # Save full comparison table
    out_path = Path("results/comparison_table.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write("```\n")
        f.write(table)
        f.write("\n```\n")
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
