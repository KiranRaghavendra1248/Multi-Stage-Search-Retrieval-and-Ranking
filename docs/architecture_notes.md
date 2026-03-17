# Architecture Notes

Design decisions, model choices, and conceptual foundations behind this pipeline.

---

## Bi-Encoder vs Cross-Encoder

These two architectures solve the same problem — measuring relevance between a query and a passage — but in fundamentally different ways.

### Bi-Encoder (Two-Tower)

```
query   → [CLS] query tokens [SEP]   → mean pool → 384-dim vector ─┐
                                                                      ├→ cosine similarity
passage → [CLS] passage tokens [SEP] → mean pool → 384-dim vector ─┘
```

- Query and passage are encoded **independently** — they never see each other's tokens
- Produces standalone vectors that can be stored in a FAISS index
- Encode all 3.2M passages **once**, store them; at query time encode query and do a single nearest-neighbour search
- Fast: O(1) search via ANN index regardless of corpus size

### Cross-Encoder

```
[CLS] query tokens [SEP] passage tokens [SEP] → single relevance score
```

- Query and passage are concatenated and processed **together**
- Every query token attends to every passage token (full cross-attention)
- Cannot produce standalone embeddings — there's no "passage vector" without a query
- Slow: requires a separate forward pass for every (query, passage) pair

### Why You Can't Swap Them

The architectures are identical (both MiniLM-L6 transformer blocks) but the **training objective** makes them incompatible:

| | Bi-Encoder | Cross-Encoder |
|---|---|---|
| Objective | Contrastive (MNRL) — push similar embeddings together, dissimilar apart | Ranking/Pointwise — given concatenated pair, predict relevance score |
| Loss | InfoNCE / Multiple Negatives Ranking Loss | Binary Cross-Entropy |
| Output | Embedding vector (meaningful in isolation) | Scalar score (only meaningful with both texts present) |
| Learned representation | "What does this text mean?" | "How relevant is passage to query jointly?" |

Running a cross-encoder in bi-encoder mode (encoding texts separately) produces garbage embeddings — the model was never trained to make its pooled output meaningful as a standalone vector.

---

## Why the Pipeline Order is Fixed

```
Bi-Encoder (fast) → narrows 3.2M to 1,000
Cross-Encoder (slow) → narrows 1,000 to 10
```

Swapping them is computationally impossible:

- Cross-encoder on 3.2M passages at query time = 3.2M forward passes = **~hours per query**
- Bi-encoder on 3.2M passages = 1 ANN lookup = **~10ms per query**

The cross-encoder's higher precision is only usable because the bi-encoder has already narrowed the search space down to a tractable number of candidates.

---

## Training Objective: Why Contrastive Learning Enables Retrieval

The contrastive objective (MNRL) teaches the model to create an **embedding space** where:

```
similar(query, relevant_passage) >> similar(query, irrelevant_passage)
```

This creates a geometry in vector space that FAISS can exploit — relevant passages are literally closer in Euclidean/cosine distance to the query vector.

A ranking objective doesn't create this geometry. It teaches the model to compare texts jointly, which requires both to be present — you can't index one half of a pair.

---

## Hard Negative Mining

### Why Hard Negatives?

Random negatives (any random passage ≠ gold) are too easy — the model trivially learns to reject them. Gradients from easy negatives approach zero, providing no learning signal.

Hard negatives are passages that:
- Score highly on BM25 (share keywords with the query)
- Are **not** the correct answer

These force the model to learn semantic relevance beyond keyword overlap — the core skill for dense retrieval.

### Mining Strategy

```
For each training query:
  1. BM25 top-100 candidates          (lexically similar passages)
  2. Remove gold passage by exact ID  (MS MARCO provides is_selected==1)
  3. Take top-5 remaining             (hard negatives — look relevant but aren't)
```

### Training Batch Composition

With batch size 256 and K=1 hard negative per query:

```
For query_i, negatives =
  255 in-batch easy negatives   (other queries' positives)
  + 1 hard-mined negative
  = 256 total negatives
```

Easy negatives provide training stability (the model needs to know what "clearly wrong" looks like). Hard negatives provide the discriminative signal. Pure hard-negative training causes instability and overfitting.

A curriculum approach is used: start with K=1 hard negative (easy-dominated batches), increase to K=3 once the model stabilises.

---

## Model Choices

### Bi-Encoder Base: `all-MiniLM-L6-v2`

| Property | Value |
|---|---|
| Parameters | 22M (vs 110M for BERT-base) |
| Embedding dim | 384 |
| Pre-training | Semantic similarity on 1B sentence pairs |
| Why not BERT-base | BERT knows language, not sentence similarity — you'd spend training budget teaching it what semantic similarity means before hard negatives help |
| Why not `msmarco-MiniLM-L-6-v3` | Already fine-tuned on MS MARCO — defeats the purpose of training |

### Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Already fine-tuned on MS MARCO passage pairs. Used as-is (no additional training). Serves as the **precision ceiling** — the best possible re-ranker for this dataset. Making it a stronger upper bound makes the comparison more meaningful.

### Benchmark: `msmarco-MiniLM-L-6-v3` (Variant 2)

The pre-trained MS MARCO bi-encoder with no additional training. Answers the key question: **does our hard-negative fine-tuning improve over what's already publicly available?** If yes, the training pipeline adds value. If no, it reveals the pre-training data already covers the distribution.

---

## LR Scheduling

```
Step 0 ──── warmup (10%) ────→ peak LR ──── cosine decay with restarts ────→ Step 100k
  LR: 0                        2e-5                                            ~0
```

- **Warmup**: BERT-family weights are already in a good loss landscape region. A full LR at step 0 on noisy early batches can push weights far from that region (catastrophic early updates). Warmup takes small steps while the model calibrates to the new task.
- **Cosine annealing with hard restarts**: After each cosine decay to near-zero, LR resets to peak. The reset allows the model to escape local minima and explore. `T_mult=2` doubles cycle length each restart — early cycles explore broadly, later cycles refine.
- **LR = 2e-5**: Lower than standard BERT fine-tuning (3e-5) because MNRL computes softmax over 256 negatives, producing larger gradients. Warmup handles most instability, so 2e-5 rather than 1e-5.

---

## HyDE (Hypothetical Document Embeddings)

**Problem:** A query like `"what causes inflation"` is short and keyword-sparse. The bi-encoder embeds it into a vector that looks like a question. Real relevant passages look like answers. These live in different regions of embedding space.

**Solution:** Use an LLM to generate a hypothetical passage that *would* answer the query. Embed that instead.

```
Query:     "what causes inflation"
           → embedding near other questions about inflation

HyDE:      "Inflation is primarily caused by excess money supply, demand-pull
            factors where consumer demand exceeds supply, and cost-push factors..."
           → embedding near actual encyclopedic passages about inflation
```

The hypothetical passage is dense, factual prose — much closer in embedding space to the real answer than the question itself. Even if the generated passage contains factual errors, it still lands in the right neighbourhood of the embedding space.
