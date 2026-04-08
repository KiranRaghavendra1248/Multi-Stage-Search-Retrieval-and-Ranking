import json
from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_triplets(jsonl_path: str) -> list[dict]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("negatives"):
                records.append({
                    "anchor":   record["query"],
                    "positive": record["positive"],
                    "negative": record["negatives"][0],  # take first hard negative
                })
    logger.info("Loaded %d triplets from %s", len(records), jsonl_path)
    return records


def train(cfg: DictConfig) -> SentenceTransformer:
    """
    Fine-tune a bi-encoder using SentenceTransformerTrainer.

    Same training objective as trainer_manual.py (MNRL + hard negatives)
    but uses the sentence-transformers high-level API instead of a custom loop.
    """
    model = SentenceTransformer(cfg.model.bi_encoder)

    triplets = _load_triplets(cfg.paths.triplets_file)
    train_dataset = Dataset.from_list(triplets)

    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=cfg.paths.checkpoint_dir,
        num_train_epochs=1,
        per_device_train_batch_size=cfg.training.per_gpu_batch_size,
        gradient_accumulation_steps=cfg.training.grad_accumulation_steps,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type="cosine",
        learning_rate=cfg.training.learning_rate,
        fp16=cfg.training.fp16,
        max_steps=cfg.training.max_steps,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_every_steps,
        save_strategy="steps",
        save_steps=cfg.training.checkpoint_every_steps,
        load_best_model_at_end=True,
        logging_steps=100,
        dataloader_num_workers=4,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    trainer.train()

    best_model_dir = cfg.paths.best_model_dir
    Path(best_model_dir).mkdir(parents=True, exist_ok=True)
    model.save(best_model_dir)
    logger.info("Best model saved to %s", best_model_dir)

    return model
