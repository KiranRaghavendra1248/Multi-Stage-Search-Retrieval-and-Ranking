# Multi-Stage Document Retrieval & Ranking Pipeline

A production-grade information retrieval system benchmarked on the full MS MARCO passage corpus (8.8M passages), implementing hard negative mining, bi-encoder fine-tuning, HyDE query expansion, and a comparative study of two re-ranking strategies.

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
│       HyDE          │  Mistral-7B-AWQ generates a hypothetical answer passage
└─────────────────────┘  (vLLM on GPU, Ollama locally)
    │
    ▼
┌─────────────────────┐
│  Stage 1 · Bi-Enc   │  Fine-tuned MiniLM bi-encoder + FAISS → Top-1000
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

**Phase 6 compares 7 variants** — baselines, ablations, and both pipelines with and without query rewriting — measuring MRR@10, Recall@100, and end-to-end latency.

---

## Dataset

| Component | Source | Size |
|---|---|---|
| Passage corpus (BM25 + FAISS index) | `BeIR/msmarco` corpus | **8.8M passages** |
| Hard negative mining queries | `microsoft/ms_marco v1.1` train | 532k queries |
| Evaluation queries + qrels | `BeIR/msmarco-qrels` validation | 7,440 query-gold pairs |

Using the full BeIR/msmarco corpus ensures MRR@10 and Recall@100 are directly comparable to published MS MARCO benchmarks. The gold passage for each dev query is looked up via `BeIR/msmarco-qrels` — a `query-id → corpus-id` mapping — so there is no train/dev leakage.

---

## Experiments

Phase 6 runs 7 variants on the standard MS MARCO Dev set (7,440 queries against 8.8M passages). Each variant isolates one variable so the contribution of each component can be measured independently.

### Variant 1 — BM25 Baseline
Pure lexical retrieval using `bm25s` over the full 8.8M passage corpus. No neural components. Sets the floor — establishes how much value dense retrieval adds over keyword matching alone.

### Variant 2 — Pre-trained MS MARCO Bi-Encoder (no fine-tuning)
`sentence-transformers/msmarco-MiniLM-L-6-v3` used as-is, with no additional training. This is the **benchmark** for our training pipeline — it answers the key question: does our hard-negative fine-tuning improve over a model already pre-trained on MS MARCO?

### Variant 3 — Our Fine-Tuned Bi-Encoder Only (Stage 1 ablation)
`all-MiniLM-L6-v2` fine-tuned with MNRL loss + BM25 hard negatives, no re-ranker. Isolates the contribution of Stage 1 dense retrieval alone. Compared directly against Variant 2 to measure the impact of our training.

### Variant 4 — Pipeline A: Fine-Tuned Bi-Encoder + ColBERT (no query rewriting)
Stage 1 top-1000 → ColBERT v2 re-ranks to top-50. ColBERT uses **late interaction**: each query token independently attends to each passage token, taking the max similarity per query token then summing. Fast because token embeddings are pre-computed offline. Measures the quality gain from re-ranking at low latency cost.

### Variant 5 — Pipeline B: Fine-Tuned Bi-Encoder + Cross-Encoder (no query rewriting)
Stage 1 top-1000 → Cross-Encoder re-ranks to top-10. The cross-encoder concatenates `[query, passage]` and runs full cross-attention — every query token sees every passage token jointly. Slower but higher precision. This is the **precision ceiling** for this architecture family.

### Variant 6 — Pipeline A + Query Rewriting
Same as Variant 4 but with query rewriting enabled before HyDE: pyspellchecker corrects typos, WordNet appends one synonym per content word (nouns + verbs only). Measures whether pre-retrieval query expansion improves MRR@10 and at what latency cost.

### Variant 7 — Pipeline B + Query Rewriting
Same as Variant 5 with query rewriting enabled. The most expensive variant end-to-end: spell check + synonym expansion + HyDE + dense retrieval + cross-encoder re-ranking.

---

## Research Questions

| Question | Variants compared |
|---|---|
| How much does dense retrieval gain over BM25? | V1 vs V3 |
| Does our hard-negative training beat the public MS MARCO model? | V2 vs V3 |
| How much does re-ranking improve over Stage 1 alone? | V3 vs V4 vs V5 |
| What is the latency/quality tradeoff between ColBERT and Cross-Encoder? | V4 vs V5 |
| Does query rewriting help? At what latency cost? | V4 vs V6, V5 vs V7 |

---

## Results (Expected — 8.8M corpus)

| Variant | MRR@10 | Recall@100 | Avg Latency |
|---|---|---|---|
| V1: BM25 Baseline | ~0.18 | ~0.67 | ~15ms |
| V2: Pre-trained MS MARCO bi-encoder (no FT) | ~0.31 | ~0.86 | ~30ms |
| V3: Our fine-tuned bi-encoder only | ~0.30+ | ~0.85+ | ~30ms |
| V4: Pipeline A — ColBERT (no rewriting) | ~0.38 | ~0.87 | ~150ms |
| V5: Pipeline B — Cross-Encoder (no rewriting) | ~0.40 | ~0.87 | ~600ms |
| V6: Pipeline A + Query Rewriting | TBD | TBD | TBD |
| V7: Pipeline B + Query Rewriting | TBD | TBD | TBD |

> Expected values align with published MS MARCO leaderboard results since we now evaluate against the same 8.8M passage corpus.

---

## Project Structure

```
├── config/                    # YAML configs (master + local/remote overrides)
├── requirements/              # base.txt / local.txt / remote.txt
├── src/
│   ├── data/
│   │   ├── ms_marco_loader.py    # HF streaming loader (train queries + hard neg mining)
│   │   ├── beir_loader.py        # BeIR/msmarco corpus loader (8.8M passages + dev eval)
│   │   ├── chunker.py            # Paragraph-aware 256-token chunking
│   │   └── triplet_dataset.py    # PyTorch Dataset over JSONL triplets
│   ├── indexing/
│   │   ├── bm25_index.py         # bm25s wrapper (build/save/load/search)
│   │   └── faiss_index.py        # FAISS Flat (local) / IVFFlat (remote)
│   ├── mining/
│   │   ├── hard_negative_miner.py  # BM25 top-100 → filter gold → top-5 HNs
│   │   └── triplet_writer.py       # Crash-safe append-mode JSONL writer
│   ├── training/
│   │   ├── bi_encoder.py         # MiniLM mean-pool + L2-norm
│   │   ├── mnrl_loss.py          # MNRL: 255 in-batch + 1 hard negative
│   │   ├── trainer.py            # AMP, cosine warm-restart, early stopping
│   │   └── validate.py           # Recall@100 on MS MARCO Dev
│   ├── inference/
│   │   ├── query_processor.py    # Spell check + synonym expansion
│   │   ├── hyde.py               # HyDE: vLLM (remote) / Ollama (local)
│   │   ├── stage1_dense.py       # Bi-encoder + FAISS retrieval (single + batch)
│   │   ├── stage2_colbert.py     # Pipeline A: ColBERT v2 re-ranker (single + batch)
│   │   └── stage2_crossencoder.py # Pipeline B: MiniLM cross-encoder (single + batch)
│   └── evaluation/
│       ├── metrics.py            # MRR@K, Recall@K, NDCG@K (from scratch)
│       └── compare.py            # 7-variant batched comparison (batch_size=32)
├── scripts/
│   ├── phase1_local_dev.py       # Local prototype (no disk writes, < 2 min)
│   ├── phase2_mine_negatives.py  # BM25 index (8.8M) + triplet mining (all train queries)
│   ├── phase3_train_biencoder.py # Fine-tune bi-encoder on Vast.ai
│   ├── phase4_build_index.py     # Build FAISS index over full 8.8M corpus
│   ├── phase5_inference_demo.py  # Interactive CLI demo
│   └── phase6_evaluate.py        # Full MRR@10 + latency evaluation (7 variants)
└── tests/                        # pytest unit tests
```

---

## Quickstart

### Local (Phase 1 — Mac, no GPU needed)

```bash
# 1. Install dependencies
make setup-local

# 2. Copy and fill env file
cp .env.example .env

# 3. Run local prototype (streams 1k rows, mines triplets, prints samples)
make phase1
```

### Remote (Phases 2–6 — Vast.ai 16GB GPU)

```bash
# 1. Push code to Vast.ai
make sync-push

# 2. SSH into instance and install
make setup-remote

# 3. Start vLLM server (for HyDE in phase 5/6)
make start-vllm   # wait ~60s before running inference

# 4. Run phases sequentially (or chain with &&)
make phase2   # ~4-6 hrs: BM25 over 8.8M + mine all train triplets
make phase3   # ~8-12 hrs: fine-tune bi-encoder, early stops on Recall@100
make phase4   # ~2-3 hrs: encode 8.8M passages, build FAISS IVFFlat index
make phase5   # interactive demo — sanity check before full eval
make phase6   # ~2-4 hrs: 7-variant evaluation on 7,440 dev queries
```

Or run everything at once in a tmux session:
```bash
make phase2 && make phase3 && make phase4 && make phase6
```

### Sync results back to local

```bash
make sync-pull-model      # trained bi-encoder checkpoint
make sync-pull-triplets   # mined hard negatives JSONL
make sync-pull-bm25       # BM25 index
```

---

## Batched Evaluation

The evaluation loop processes queries in outer batches of 32 to maximize GPU utilization:

| Stage | Strategy | VRAM peak |
|---|---|---|
| FAISS retrieval | 32 queries encoded + searched at once | ~100 MB |
| ColBERT rerank | Query vecs encoded together; doc encoding per-query (batch=32) | ~46 MB/query |
| Cross-Encoder rerank | 32,000 pairs in one `predict()` call; internal mini-batch=8 | ~48 MB |

VRAM usage is logged after each variant so batch sizes can be tuned up if headroom allows.

---

## Key Design Decisions

### Hard Negative Mining
BM25 retrieves the top-100 candidates for each query. The gold passage (identified by `is_selected==1`) is filtered out, and the top-5 remaining passages become **hard negatives** — documents that look relevant but are not the correct answer. These force the bi-encoder to learn semantic relevance beyond keyword overlap.

### Training Loss: MNRL
With a batch size of 64 per GPU (128 effective with gradient accumulation), each query gets:
- **In-batch negatives** (other positives in the batch)
- **1 hard-mined negative** (from BM25)

This curriculum mix avoids training instability from hard-negative-only batches while still providing the discriminative signal BM25 hard negatives offer.

### HyDE (Hypothetical Document Embeddings)
Instead of embedding the short, keyword-like query, Mistral-7B-AWQ (via vLLM) generates a plausible answer passage. The dense retriever searches for real passages similar to this hypothetical answer — improving recall for queries where the question and relevant passages share few literal words.

### Pipeline A vs Pipeline B
| | Pipeline A | Pipeline B |
|---|---|---|
| Re-ranker | ColBERT v2 (late interaction) | Cross-Encoder MiniLM |
| Mechanism | MaxSim over token embeddings | Full cross-attention (query, passage) |
| Speed | ~150ms/query | ~600ms/query |
| Precision | High | Higher |

ColBERT is the practical production choice. Cross-Encoder is the precision ceiling.

### Query Rewriting
An optional pre-retrieval step applies pyspellchecker for typo correction and WordNet synonym expansion for content words (nouns + verbs only). The experiment measures whether this pre-processing improves MRR@10 and at what latency cost.

---

## Hardware

| Phase | Hardware | Est. Time |
|---|---|---|
| Phase 1 (local dev) | MacBook (arm64, no GPU) | < 2 min |
| Phase 2 (mining) | Vast.ai 1×16GB GPU, 128GB RAM | ~4-6 hrs |
| Phase 3 (training) | Vast.ai 1×16GB GPU | ~8-12 hrs |
| Phase 4 (FAISS index build) | Vast.ai 1×16GB GPU, 128GB RAM | ~2-3 hrs |
| Phase 5 (inference demo) | Vast.ai 1×16GB GPU | interactive |
| Phase 6 (eval) | Vast.ai 1×16GB GPU | ~2-4 hrs |

**GPU memory allocation (remote, 16GB):**
- vLLM (Mistral-7B-AWQ): `--gpu-memory-utilization 0.6` → ~9.6 GB
- Bi-encoder, ColBERT, Cross-Encoder reranking: ~6 GB available

**System RAM (Phase 2 + Phase 4):**
BM25 and FAISS indexes are built in CPU RAM before being saved to disk. At 8.8M passages: ~12 GB for BM25, ~30 GB peak for FAISS (embeddings + index + passage store). 128 GB RAM on the VM handles this comfortably.

---

## Running Tests

```bash
make test
# or
pytest tests/ -v
```

---

## Dependencies

- `datasets` — HuggingFace streaming (ms_marco + BeIR/msmarco)
- `transformers` + `sentence-transformers` — MiniLM bi-encoder, Cross-Encoder
- `bm25s` — Pure Python BM25, no Java required
- `faiss-cpu` — Approximate nearest neighbour search (CPU, IVFFlat)
- `pylate` — ColBERT v2 re-ranking (late interaction MaxSim)
- `vllm` — High-throughput LLM inference for HyDE (AWQ quantization)
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
