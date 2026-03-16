"""
Phase 3: Fine-Tune Bi-Encoder (Vast.ai)

Trains BERT-base-uncased with MNRL loss on mined triplets.
  - DataParallel across 2 GPUs
  - Cosine annealing with warm restarts (3 cycles) after 10% linear warmup
  - Evaluates Recall@100 every 10k steps
  - Saves best checkpoint to cfg.paths.best_model_dir
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.training.trainer import train

logger = get_logger("phase3")


def main():
    cfg = load_config()
    logger.info("Environment: %s", cfg.environment)
    logger.info("Bi-encoder model: %s", cfg.model.bi_encoder)
    logger.info("Triplets file: %s", cfg.paths.triplets_file)
    logger.info(
        "Training: global_batch=%d fp16=%s max_steps=%d",
        cfg.training.global_batch_size,
        cfg.training.fp16,
        cfg.training.max_steps,
    )

    best_model = train(cfg)
    logger.info("Training complete. Best model saved to: %s", cfg.paths.best_model_dir)


if __name__ == "__main__":
    main()
