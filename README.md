# Multi-Stage Document Retrieval & Ranking Pipeline

A production-grade information retrieval system built on MS MARCO (3.2M documents), implementing hard negative mining, bi-encoder fine-tuning, HyDE query expansion, and a comparative study of two re-ranking strategies.

---

## Architecture

```
Raw Query
    │
    ▼ (optional)
┌─────────────────────┐
│   Query Processor   │  Spell check + WordNet synonym expansion
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│       HyDE          │  Llama-3-8B generates a hypothetical answer passage
└─────────────────────┘  (vLLM on GPU, Ollama locally)
    │
    ▼
┌─────────────────────┐
│  Stage 1 · Bi-Enc   │  Fine-tuned BERT bi-encoder + FAISS → Top-1000
└─────────────────────┘
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
┌──────────────┐              ┌──────────────────┐
│  Pipeline A  │              │   Pipeline B     │
│   ColBERT    │              │  Cross-Encoder   │
│   Top-50     │              │   Top-10         │
│  (fast)      │              │  (precise)       │
└──────────────┘              └──────────────────┘
```

**Phase 5 compares 6 variants** — both pipelines with and without query rewriting — measuring MRR@10 and end-to-end latency.

---

## Results (Expected)

| Variant                                   | MRR@10 | Recall@100 | Avg Latency |
|-------------------------------------------|--------|------------|-------------|
| BM25 Baseline                             | ~0.17  | ~0.60      | ~15ms       |
| Pre-trained MS MARCO bi-encoder (no FT)   | ~0.31  | ~0.86      | ~30ms       |
| Our fine-tuned bi-encoder only            | ~0.30+ | ~0.85+     | ~30ms       |
| Pipeline A: ColBERT                       | ~0.38  | ~0.87      | ~150ms      |
| Pipeline B: Cross-Encoder                 | ~0.40  | ~0.87      | ~600ms      |
| Pipeline A + Query Rewriting              | TBD    | TBD        | TBD         |
| Pipeline B + Query Rewriting              | TBD    | TBD        | TBD         |

> **Key question:** Does our hard-negative fine-tuning match or beat the pre-trained MS MARCO bi-encoder (Variant 2)? If yes, the training pipeline adds measurable value.

---

## Project Structure

```
├── config/                    # YAML configs (master + local/remote overrides)
├── requirements/              # base.txt / local.txt / remote.txt
├── src/
│   ├── data/
│   │   ├── ms_marco_loader.py    # HF streaming loader
│   │   ├── chunker.py            # Paragraph-aware 256-token chunking
│   │   └── triplet_dataset.py    # PyTorch Dataset over JSONL triplets
│   ├── indexing/
│   │   ├── bm25_index.py         # bm25s wrapper (build/save/load/search)
│   │   └── faiss_index.py        # FAISS Flat (local) / IVFFlat+GPU (remote)
│   ├── mining/
│   │   ├── hard_negative_miner.py  # BM25 top-100 → filter gold → top-5 HNs
│   │   └── triplet_writer.py       # Crash-safe append-mode JSONL writer
│   ├── training/
│   │   ├── bi_encoder.py         # BERT mean-pool + L2-norm
│   │   ├── mnrl_loss.py          # MNRL: 255 in-batch + 1 hard negative
│   │   ├── trainer.py            # DataParallel, AMP, cosine warm-restart
│   │   └── validate.py           # Recall@100 on MS MARCO Dev
│   ├── inference/
│   │   ├── query_processor.py    # Spell check + synonym expansion
│   │   ├── hyde.py               # HyDE: vLLM (remote) / Ollama (local)
│   │   ├── stage1_dense.py       # Bi-encoder + FAISS retrieval
│   │   ├── stage2_colbert.py     # Pipeline A: ColBERT v2 re-ranker
│   │   └── stage2_crossencoder.py # Pipeline B: MiniLM cross-encoder
│   └── evaluation/
│       ├── metrics.py            # MRR@K, Recall@K, NDCG@K (from scratch)
│       └── compare.py            # 6-variant comparison table
├── scripts/
│   ├── phase1_local_dev.py       # Local prototype (no disk writes, < 2 min)
│   ├── phase2_mine_negatives.py  # Full BM25 indexing + 500k triplet mining
│   ├── phase3_train_biencoder.py # Fine-tune bi-encoder on Vast.ai
│   ├── phase4_inference_demo.py  # Interactive CLI demo
│   └── phase5_evaluate.py        # Full MRR@10 + latency evaluation
└── tests/                        # pytest unit tests
```

---

## Quickstart

### Local (Phase 1 — Mac, no GPU needed)

```bash
# 1. Install dependencies
make setup-local
# Note: install faiss-cpu separately on arm64:
conda install -c conda-forge faiss-cpu

# 2. Copy and fill env file
cp .env.example .env

# 3. Run local prototype (streams 1k rows, mines triplets, prints samples)
make phase1
```

### Remote (Phases 2–5 — Vast.ai 2×16GB GPU)

```bash
# 1. Push code to Vast.ai
make sync-push

# 2. SSH into instance and install
make setup-remote

# 3. Run phases sequentially
python scripts/phase2_mine_negatives.py   # ~4–6 hrs, crash-safe resume
python scripts/phase3_train_biencoder.py  # ~8–12 hrs, checkpoints every 5k steps
python scripts/phase4_inference_demo.py --pipeline A   # interactive demo
python scripts/phase5_evaluate.py                      # full comparison
```

---

## Key Design Decisions

### Hard Negative Mining
BM25 retrieves the top-100 candidates for each query. The gold passage (identified by `is_selected==1`) is filtered by exact ID match, and the top-5 remaining passages become **hard negatives** — documents that look relevant but are not the correct answer. These force the bi-encoder to learn semantic relevance beyond keyword overlap.

### Training Loss: MNRL
With a batch size of 256, each query gets:
- **255 easy in-batch negatives** (other positives in the batch)
- **1 hard-mined negative** (from BM25)

This curriculum mix avoids training instability from hard-negative-only batches while still providing the discriminative signal BM25 hard negatives offer.

### HyDE (Hypothetical Document Embeddings)
Instead of embedding the short, keyword-like query, a Llama-3-8B model generates a plausible answer passage. The dense retriever searches for real passages similar to this hypothetical answer — significantly improving recall for questions where the query and relevant passages share few literal words.

### Pipeline A vs Pipeline B
| | Pipeline A | Pipeline B |
|---|---|---|
| Re-ranker | ColBERT v2 (late interaction) | Cross-Encoder MiniLM |
| Mechanism | MaxSim over token embeddings | Full cross-attention (query, passage) |
| Speed | ~200ms/query | ~800ms/query |
| Precision | High | Higher |

ColBERT is the practical production choice. Cross-Encoder is the precision ceiling.

### Query Rewriting
An optional pre-retrieval step applies pyspellchecker for typo correction and WordNet synonym expansion for content words (nouns + verbs only). The experiment measures whether this pre-processing improves MRR@10 and at what latency cost.

---

## Hardware

| Phase | Hardware | Est. Time |
|---|---|---|
| Phase 1 (local dev) | MacBook (arm64, no GPU) | < 2 min |
| Phase 2 (mining) | Vast.ai 2×16GB GPU | 4–6 hrs |
| Phase 3 (training) | Vast.ai 2×16GB GPU | 8–12 hrs |
| Phase 4 (inference) | Vast.ai 2×16GB GPU | interactive |
| Phase 5 (eval) | Vast.ai 2×16GB GPU | 2–4 hrs |

**GPU memory allocation (remote):**
- GPU 0 (40%): Llama-3-8B-Instruct AWQ (~6.4GB) via vLLM
- GPU 0 (remaining) + GPU 1: Bi-encoder, FAISS IVFFlat index, ColBERT/Cross-Encoder

---

## Running Tests

```bash
make test
# or
pytest tests/ -v
```

---

## Dependencies

- `datasets` — MS MARCO streaming via HuggingFace
- `transformers` + `sentence-transformers` — BERT bi-encoder, Cross-Encoder
- `bm25s` — Pure Python BM25, no Java required
- `faiss-gpu` / `faiss-cpu` — Approximate nearest neighbour search
- `ragatouille` — ColBERT v2 re-ranking
- `vllm` — High-throughput LLM inference for HyDE
- `omegaconf` — Hierarchical YAML config with local/remote overrides
- `pyspellchecker` + `nltk` — Query rewriting pre-processing

---

## Notes & Design Decisions

See [docs/architecture_notes.md](docs/architecture_notes.md) for detailed explanations of:
- Bi-encoder vs Cross-Encoder architecture and why they can't be swapped
- Why contrastive learning enables retrieval but ranking objective doesn't
- Hard negative mining strategy and batch composition
- Model choices and the training benchmark (Variant 2)
- LR scheduling rationale (warmup + cosine restarts)
- HyDE query expansion mechanics
