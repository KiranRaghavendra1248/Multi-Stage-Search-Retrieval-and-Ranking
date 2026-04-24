# Multi-Stage Document Retrieval & Ranking Pipeline

A production-grade information retrieval system benchmarked on the full MS MARCO passage corpus (8.8M passages). Two iterations of training are implemented:

- **Iteration 1**: Hard negative mining with BM25 teacher, `all-MiniLM-L6-v2` bi-encoder
- **Iteration 2**: Dense teacher mining (`e5-large-unsupervised` or `e5-mistral-7b-instruct`) with positive-aware false negative filtering (NV-Retriever, arxiv:2407.15831), upgraded to `e5-large-unsupervised` student

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
│  Stage 1 · Bi-Enc   │  Fine-tuned e5-large-unsupervised + FAISS → Top-1000
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

**Phase 6 compares 11 variants** — baselines, ablations, RRF fusion pipelines, and both re-ranking pipelines with and without query rewriting — measuring MRR@10, Recall@100, and end-to-end latency.

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

Both iterations are evaluated on the same 11 pipeline variants against the MS MARCO Dev set (7,440 queries, 8.8M passage corpus). The pipeline structure is identical across iterations — what changes is the training recipe for the fine-tuned bi-encoder (V3 and above).

### Shared Variants (both iterations)

#### Variant 1 — BM25 Baseline
Pure lexical retrieval using `bm25s` over the full 8.8M passage corpus. No neural components. Sets the floor — establishes how much value dense retrieval adds over keyword matching alone. Same result in both iterations (BM25 is unchanged).

#### Variant 2 — e5-large-unsupervised (no fine-tuning)
`intfloat/e5-large-unsupervised` used as-is, no additional training — same architecture and prefixes as our fine-tuned student (V3). The **direct benchmark** for our training pipeline: does our dense-teacher hard-negative fine-tuning improve over the off-the-shelf model? Any gain in V3 over V2 is attributable purely to our training recipe.

#### Variant 3 — Our Fine-Tuned Bi-Encoder Only (Stage 1 ablation)
Fine-tuned bi-encoder with MNRL loss, no re-ranker. Isolates Stage 1 quality.
- **Iteration 1**: `all-MiniLM-L6-v2` + BM25 hard negatives (no false negative filtering)
- **Iteration 2**: `intfloat/e5-large-unsupervised` + dense teacher hard negatives + TopK-PercPos filtering

#### Variant 4 — Pipeline A: Fine-Tuned Bi-Encoder + ColBERT (no query rewriting)
Stage 1 top-1000 → ColBERT v2 re-ranks to top-50. ColBERT uses **late interaction**: each query token independently attends to each passage token (MaxSim), then scores are summed. Token embeddings are pre-computed offline — fast at query time.

#### Variant 5 — Pipeline B: Fine-Tuned Bi-Encoder + Cross-Encoder (no query rewriting)
Stage 1 top-1000 → Cross-Encoder re-ranks to top-10. The cross-encoder runs full cross-attention over the concatenated `[query, passage]` — every token sees every other token. Slower but higher precision. **Precision ceiling** for this architecture family.

#### Variant 6 — Pipeline A + Query Rewriting
Same as Variant 4 but with query rewriting before HyDE: pyspellchecker corrects typos, WordNet appends one synonym per content word (nouns + verbs). Measures pre-retrieval expansion benefit and latency cost.

#### Variant 7 — Pipeline B + Query Rewriting
Same as Variant 5 with query rewriting enabled. Most expensive variant end-to-end: spell check + synonym expansion + HyDE + dense retrieval + cross-encoder re-ranking.

#### Variant 8 — RRF: BM25 + Fine-Tuned Bi-Encoder
Reciprocal Rank Fusion of BM25 (top-1000) and fine-tuned dense retrieval (top-1000). Each passage scored by `1/(60 + rank_bm25) + 1/(60 + rank_dense)`, top-1000 returned with no further re-ranking. Tests whether lexical + semantic fusion improves over either alone.

#### Variant 9 — RRF: BM25 + Pre-trained Bi-Encoder
Same RRF using the pre-trained MS MARCO model. Directly comparable to Variant 8 to isolate fine-tuning impact within the RRF setup.

#### Variant 10 — RRF: BM25 + Fine-Tuned → ColBERT → Cross-Encoder
Full pipeline: BM25(1000) + fine-tuned dense(1000) fused via RRF → ColBERT top-100 → Cross-Encoder top-10. Tests whether RRF as Stage 1 improves the precision ceiling over dense-only Stage 1.

#### Variant 11 — RRF: BM25 + Pre-trained → ColBERT → Cross-Encoder
Same full pipeline using the pre-trained bi-encoder for the dense leg of RRF. Compared against Variant 10 to measure fine-tuning value within the full re-ranking stack.

---

## Research Questions

| Question | Variants compared |
|---|---|
| How much does dense retrieval gain over BM25? | V1 vs V3 |
| Does our hard-negative training beat the public MS MARCO model? | V2 vs V3 |
| How much does re-ranking improve over Stage 1 alone? | V3 vs V4 vs V5 |
| What is the latency/quality tradeoff between ColBERT and Cross-Encoder? | V4 vs V5 |
| Does query rewriting help? At what latency cost? | V4 vs V6, V5 vs V7 |
| Does RRF fusion improve over dense-only retrieval? | V3 vs V8, V2 vs V9 |
| Does fine-tuning help within the RRF setup? | V8 vs V9, V10 vs V11 |
| Does RRF as Stage 1 improve the full re-ranking pipeline? | V5 vs V10, V5 vs V11 |
| Does dense teacher mining beat BM25 teacher mining? | I1-V3 vs I2-V3 |
| Does positive-aware filtering improve the full pipeline? | I1-V5 vs I2-V5, I1-V10 vs I2-V10 |
| Does the larger student model (e5-large vs MiniLM) help independently of training? | I1-V3 vs I2-V3 (controlled for training recipe) |

---

## Iteration 1 — Results & Diagnosis

### What we observed

Running all 11 variants on the 7,440 MS MARCO dev queries produced a surprising result: **the fine-tuned bi-encoder (V3) scored worse than both BM25 (V1) and the pretrained MS MARCO model (V2)**. ColBERT and Cross-Encoder re-ranking (V4/V5) partially recovered quality over V3, but the Stage 1 representations remained the bottleneck. Our training run made the model *worse*, not better.

### Root cause — BM25 false negatives

BM25 retrieves by keyword overlap. Passages that are semantically relevant but share few literal query tokens rank low and never surface as candidates — so they never become hard negatives. But passages that share many keywords with the query rank high in BM25, even when they aren't genuinely relevant. These become "hard negatives" in our training data.

The problem: many of these high-BM25-rank passages *are* relevant — they're false negatives. Training with them as negatives directly penalizes the model for ranking semantically similar passages highly, corrupting the contrastive signal throughout training.

The NV-Retriever paper (arxiv:2407.15831) quantified this precisely: approximately **70% of top-ranked BM25 candidates for MS MARCO queries should be labeled positive** but are incorrectly treated as negatives. With 5 hard negatives per query from BM25, most training triplets are poisoned.

### What the re-rankers recover

ColBERT (V4) and Cross-Encoder (V5) re-score the top-1000 candidates with full query-passage attention — a much stronger signal than bi-encoder similarity. This partially undoes the damage from bad Stage 1 representations, which explains why the re-ranking variants score higher than V3 alone.

### What this motivates — Iteration 2

The fix is not in the loss function (MNRL ≈ InfoNCE; they're equivalent). The fix is in the **quality of hard negatives**. Iteration 2 replaces BM25 with a dense teacher model and adds **positive-aware filtering** (TopK-PercPos from NV-Retriever) to remove false negatives before they enter training:

- **Teacher**: `intfloat/e5-large-unsupervised` (local, TensorRT) or `intfloat/e5-mistral-7b-instruct` (vLLM embedding server)
- **Filtering**: keep only candidates with similarity ≤ 0.95 × positive_score (TopK-PercPos)
- **Student**: upgraded to `intfloat/e5-large-unsupervised` (335M params, 1024-dim) from MiniLM (22M, 384-dim)

---

## Results (8.8M corpus)

### Iteration 1 — BM25 Teacher, `all-MiniLM-L6-v2` Student

*NDCG@10 not tracked in this run (added to eval code in Iteration 2).*

| Variant | MRR@10 | Recall@100 | Latency (ms) |
|---|---|---|---|
| V1: BM25 Baseline | 0.1364 | 0.5181 | 94.6 |
| V2: e5-large-unsupervised (no fine-tuning) | 0.2961 | 0.7669 | 19.0 |
| V3: Fine-tuned bi-encoder only | 0.1344 | 0.2728 | 5.9 |
| V4: Pipeline A — ColBERT (no rewriting) | 0.2886 | 0.5968 | 1511.5 |
| V5: Pipeline B — Cross-Encoder (no rewriting) | 0.2928 | 0.4917 | 666.8 |
| V6: Pipeline A + Query Rewriting | 0.2499 | 0.5181 | 1504.8 |
| V7: Pipeline B + Query Rewriting | 0.2536 | 0.4245 | 660.2 |
| V8: RRF — BM25 + Fine-tuned | 0.2010 | 0.6966 | 111.4 |
| V9: RRF — BM25 + Pre-trained | 0.2306 | 0.8027 | 81.7 |
| V10: RRF — BM25 + Fine-tuned → ColBERT → Cross-Encoder | 0.3848 | 0.6542 | 1662.9 |
| V11: RRF — BM25 + Pre-trained → ColBERT → Cross-Encoder | 0.3921 | 0.6765 | 1606.1 |

### Iteration 2 — Dense Teacher (`e5-large-unsupervised` + TopK-PercPos), `e5-large-unsupervised` Student

| Variant | MRR@10 | NDCG@10 | Recall@100 | Latency (ms) |
|---|---|---|---|---|
| V1: BM25 Baseline | *(same as I1)* | *(same as I1)* | *(same as I1)* | *(same as I1)* |
| V2: e5-large-unsupervised (no fine-tuning) | *(same as I1)* | *(same as I1)* | *(same as I1)* | *(same as I1)* |
| V3: Fine-tuned bi-encoder only | TBD | TBD | TBD | TBD |
| V4: Pipeline A — ColBERT (no rewriting) | TBD | TBD | TBD | TBD |
| V5: Pipeline B — Cross-Encoder (no rewriting) | TBD | TBD | TBD | TBD |
| V6: Pipeline A + Query Rewriting | TBD | TBD | TBD | TBD |
| V7: Pipeline B + Query Rewriting | TBD | TBD | TBD | TBD |
| V8: RRF — BM25 + Fine-tuned | TBD | TBD | TBD | TBD |
| V9: RRF — BM25 + Pre-trained | TBD | TBD | TBD | TBD |
| V10: RRF — BM25 + Fine-tuned → ColBERT → Cross-Encoder | TBD | TBD | TBD | TBD |
| V11: RRF — BM25 + Pre-trained → ColBERT → Cross-Encoder | TBD | TBD | TBD | TBD |

> Results populated after each Phase 6 run. Each variant saves immediately to `results/<slug>.json`. Pull with `make sync-pull-results`.

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
│   │   ├── hard_negative_miner.py  # Dense/BM25 teacher → positive-aware filter → top-5 HNs
│   │   ├── dense_teacher.py        # TensorRTDenseTeacher (e5-large) + VLLMDenseTeacher (e5-mistral)
│   │   └── triplet_writer.py       # Crash-safe append-mode JSONL writer
│   ├── training/
│   │   ├── bi_encoder.py         # e5-large-unsupervised mean-pool + L2-norm, query/passage prefix support
│   │   ├── mnrl_loss.py          # MNRL: 255 in-batch + 1 hard negative
│   │   ├── trainer.py            # AMP, cosine warm-restart, saves best Recall@100 checkpoint
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
│   ├── phase4_build_index.py     # Build both FAISS indexes (fine-tuned + pretrained) over 8.8M corpus
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

### Remote — Iteration 2 (Phases 2–6 — Vast.ai 16GB GPU)

```bash
# 1. Push code to Vast.ai
make sync-push

# 2. SSH into instance and install
make setup-remote

# 3a. Dense teacher mining (e5-large-unsupervised — no server needed)
#     Set config: mining.teacher = "intfloat/e5-large-unsupervised"
make phase2   # ~6-10 hrs: BM25 index + dense teacher FAISS + mine all train triplets

# 3b. OR: e5-mistral-7b-instruct teacher (start vLLM embedding server first)
#     Set config: mining.teacher = "intfloat/e5-mistral-7b-instruct"
#     Set .env: TEACHER_MODEL=intfloat/e5-mistral-7b-instruct
make start-vllm-teacher   # starts vLLM on port 8001 (INT8, ~8GB VRAM)
make phase2               # stop teacher server before phase 5/6

# 4. Train bi-encoder
make phase3   # ~8-12 hrs: fine-tune e5-large-unsupervised to max_steps

# 5. Build FAISS index (1024-dim — delete old indexes if upgrading from Iteration 1)
make phase4   # ~4-6 hrs: encode 8.8M passages at 1024-dim

# 6. Stop teacher server (if used), start HyDE server
make stop-vllm-teacher   # only needed if step 3b was used
make start-vllm          # Mistral-7B-AWQ for HyDE (port 8000)
make phase5              # interactive demo — sanity check
make phase6              # ~2-4 hrs: 11-variant evaluation on 7,440 dev queries
```

Or chain in a tmux session:
```bash
make phase2 && make phase3 && make phase4 && make phase6
```

### Sync results back to local

```bash
make sync-pull-model      # trained bi-encoder checkpoint
make sync-pull-triplets   # mined hard negatives JSONL
make sync-pull-bm25       # BM25 index
make sync-pull-results    # per-variant JSON results from phase6
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

### Hard Negative Mining — Iteration 2: Dense Teacher + Positive-Aware Filtering

Iteration 1 used BM25 as the teacher for mining hard negatives. The evaluation results revealed that the fine-tuned model performed *worse* than both the BM25 baseline and the pretrained MS MARCO model — diagnosed as a false negative poisoning problem.

BM25 retrieves by keyword overlap. Many top-ranked BM25 candidates are semantically relevant but incorrectly treated as negatives. The NV-Retriever paper (arxiv:2407.15831) found ~70% of top-ranked BM25 candidates for MS MARCO should be labeled positive. Training with these false negatives corrupts the contrastive signal: the model is penalized for ranking genuinely relevant passages highly.

Iteration 2 fixes this with two changes:

**1. Dense teacher**: `intfloat/e5-large-unsupervised` (335M params, 1024-dim) replaces BM25 for candidate retrieval. The teacher builds a FAISS index over the full 8.8M passage corpus and retrieves semantically similar candidates — far fewer false negatives than keyword overlap.

For the highest-quality negatives, `intfloat/e5-mistral-7b-instruct` (7B) can be used as teacher via a vLLM embedding server (`make start-vllm-teacher`, separate from the HyDE server).

**2. Positive-aware filtering (TopK-PercPos)**: Even a dense teacher can surface near-duplicate positives. After retrieval, candidates are filtered using the positive passage score as an anchor:

```
threshold = perc_pos × sim(query, positive)    # default: 0.95
keep only candidates where sim(query, candidate) ≤ threshold
```

This removes candidates that are too similar to the gold passage before they enter training. Both `topk_perc_pos` (relative, recommended) and `topk_margin_pos` (absolute) are supported via config.

**Teacher/student prefixes**: `e5-large-unsupervised` requires `query: ` and `passage: ` prefixes — explicitly required by the model card for correct retrieval. These are applied consistently at mining time (teacher), training time (collate_fn), and inference time (encode_queries/encode_passages).

### Training Loss: MNRL
With a batch size of 64 per GPU (128 effective with gradient accumulation), each query gets:
- **In-batch negatives** (other positives in the batch)
- **1 hard-mined negative** (from dense teacher, false-negative filtered)

MNRL and InfoNCE are functionally identical for this setup (cross-entropy over normalized cosines). The quality gain in Iteration 2 comes entirely from better negatives, not from changing the loss.

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
| Phase 4 (FAISS index build) | Vast.ai 1×16GB GPU, 128GB RAM | ~4-6 hrs (two indexes) |
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
