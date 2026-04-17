import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from omegaconf import DictConfig

from src.training.bi_encoder import BiEncoder
from src.training.mnrl_loss import MNRLWithHardNegatives
from src.data.triplet_dataset import TripletDataset, build_collate_fn
from src.training.validate import evaluate_recall
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def train(cfg: DictConfig) -> BiEncoder:
    """
    Full training loop for the bi-encoder.

    - DataParallel across cfg.training.num_gpus GPUs
    - MNRL loss with 1 hard negative per query
    - Cosine annealing with hard restarts after linear warmup
    - fp16 AMP (skipped if cfg.training.fp16 == False)
    - Checkpoints every cfg.training.checkpoint_every_steps steps
    - Saves best Recall@100 checkpoint but runs to max_steps regardless
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    # --- Model ---
    model = BiEncoder(cfg.model.bi_encoder)
    if cfg.training.num_gpus > 1 and torch.cuda.device_count() >= cfg.training.num_gpus:
        device_ids = list(range(cfg.training.num_gpus))
        model = nn.DataParallel(model, device_ids=device_ids)
        logger.info("Using DataParallel across GPUs: %s", device_ids)
    model = model.to(device)

    # --- Data ---
    dataset = TripletDataset(cfg.paths.triplets_file, k_hard_negatives=1)
    collate_fn = build_collate_fn(
        tokenizer_name=cfg.model.bi_encoder,
        max_length=256,
        query_prefix=cfg.model.get("bi_encoder_query_prefix", ""),
        passage_prefix=cfg.model.get("bi_encoder_passage_prefix", ""),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.per_gpu_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # --- Loss & Optimizer ---
    criterion = MNRLWithHardNegatives(temperature=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=0.01)

    # --- Scheduler ---
    warmup_steps = int(cfg.training.max_steps * cfg.training.warmup_ratio)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=cfg.training.max_steps,
        num_cycles=cfg.training.num_cycles,
    )

    # --- AMP ---
    use_fp16 = cfg.training.fp16 and torch.cuda.is_available()
    scaler = GradScaler() if use_fp16 else None

    # --- Checkpointing ---
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = Path(cfg.paths.best_model_dir)

    # --- Training state ---
    global_step = 0
    accum_steps = cfg.training.grad_accumulation_steps
    best_recall = 0.0

    logger.info("Starting training. max_steps=%d warmup=%d", cfg.training.max_steps, warmup_steps)

    optimizer.zero_grad()

    while global_step < cfg.training.max_steps:
        for batch in loader:
            if global_step >= cfg.training.max_steps:
                break

            # Move to device
            query_enc = {k: v.to(device) for k, v in batch["query_enc"].items()}
            pos_enc   = {k: v.to(device) for k, v in batch["pos_enc"].items()}
            hn_enc    = {k: v.to(device) for k, v in batch["hn_enc"].items()}
            B, K = batch["B"], batch["K"]

            # Forward
            _model = model.module if isinstance(model, nn.DataParallel) else model
            if use_fp16:
                with autocast():
                    q_emb  = model(query_enc)           # [B, D]
                    p_emb  = model(pos_enc)             # [B, D]
                    hn_emb = model(hn_enc)              # [B*K, D]
                    hn_emb = hn_emb.view(B, K, -1)[:, 0, :]  # [B, D] take first hard neg
                    loss   = criterion(q_emb, p_emb, hn_emb) / accum_steps
                scaler.scale(loss).backward()
            else:
                q_emb  = model(query_enc)
                p_emb  = model(pos_enc)
                hn_emb = model(hn_enc)
                hn_emb = hn_emb.view(B, K, -1)[:, 0, :]
                loss   = criterion(q_emb, p_emb, hn_emb) / accum_steps
                loss.backward()

            # Gradient accumulation
            if (global_step + 1) % accum_steps == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % 100 == 0:
                lr_now = scheduler.get_last_lr()[0]
                logger.info("step=%d loss=%.4f lr=%.2e", global_step, loss.item() * accum_steps, lr_now)

            # Checkpoint
            if global_step % cfg.training.checkpoint_every_steps == 0:
                ckpt_path = ckpt_dir / f"step_{global_step}"
                _save_model(model, str(ckpt_path))
                logger.info("Checkpoint saved: %s", ckpt_path)

            # Evaluation
            if global_step % cfg.training.eval_every_steps == 0:
                recall = evaluate_recall(model, cfg, device=str(device))
                logger.info("step=%d Recall@100=%.4f", global_step, recall)
                if recall > best_recall:
                    best_recall = recall
                    _save_model(model, str(best_model_dir))
                    logger.info("New best Recall@100=%.4f — saved to %s", best_recall, best_model_dir)

    return _load_best_model(str(best_model_dir), cfg)


def _save_model(model: nn.Module, path: str) -> None:
    m = model.module if isinstance(model, nn.DataParallel) else model
    m.save(path)


def _load_best_model(path: str, cfg: DictConfig) -> BiEncoder:
    return BiEncoder.load(path)
