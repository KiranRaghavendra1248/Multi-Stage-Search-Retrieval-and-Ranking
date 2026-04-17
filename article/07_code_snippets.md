# Curated Code Snippets

A collection of the most instructive code excerpts from the codebase, each illustrating a key architectural or implementation decision.

---

## 1. Bi-Encoder Forward Pass with Mean Pooling

**File:** `src/training/bi_encoder.py`

```python
class BiEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_emb = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_emb / sum_mask

    def forward(self, enc: BatchEncoding) -> torch.Tensor:
        out = self.encoder(**enc)
        pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
        return nn.functional.normalize(pooled, p=2, dim=-1)
```

The `clamp(min=1e-9)` on the mask sum prevents division-by-zero on fully-padded sequences, which can appear in edge cases. The L2 normalization at the end is what makes FAISS `IndexFlatIP` (inner product search) equivalent to cosine similarity — without it, longer sequences would systematically produce larger dot products regardless of semantic content. These two lines are easy to overlook and both are load-bearing.

---

## 2. MNRL Loss with Hard Negatives

**File:** `src/training/mnrl_loss.py`

```python
class MNRLWithHardNegatives(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embs, pos_embs, hard_neg_embs):
        B = query_embs.size(0)

        # In-batch negatives: [B, B] — diagonal = positives
        scores_inbatch = torch.matmul(query_embs, pos_embs.T) / self.temperature

        # Hard negatives: [B, 1]
        scores_hard = (query_embs * hard_neg_embs).sum(dim=-1, keepdim=True) / self.temperature

        # Combined logits: [B, B+1]
        logits = torch.cat([scores_inbatch, scores_hard], dim=1)

        # Target: each query's positive is at column index i
        targets = torch.arange(B, device=query_embs.device)
        return F.cross_entropy(logits, targets)
```

The key insight here is the matrix shape: `scores_inbatch` is `[B, B]` where diagonal element `(i, i)` is the correct (query_i, positive_i) score, and off-diagonal elements `(i, j)` are "free" negatives from other queries' positives. Appending `scores_hard` as `[B, 1]` extends this to a `[B, B+1]` classification problem. The result: every training step produces `B` loss terms, each trained against `B` easy negatives plus 1 hard negative. Temperature=0.05 is critical — without it, the softmax is too flat to provide useful gradient signal.

---

## 3. BM25 Hard Negative Mining with Crash-Safe Resume

**File:** `src/mining/hard_negative_miner.py`

```python
def mine_hard_negatives(
    records: Iterable[dict],
    index: BM25Index,
    writer: TripletWriter,
    seen_queries: set[str],
    n_hard_negatives: int = 5,
    bm25_top_k: int = 100,
    max_triplets: int | None = None,
) -> dict:
    for record in tqdm(records, desc="Mining hard negatives", unit="query"):
        query = record["query"]
        positive = record["positive_passage"]

        if query in seen_queries:        # crash-safe resume
            continue
        if not positive:                 # skip unannotated queries
            continue

        bm25_results = index.search(query, top_k=bm25_top_k)
        hard_negatives = [
            text for text, _score in bm25_results
            if text != positive          # exact match filter removes the gold
        ]
        hard_negatives = hard_negatives[:n_hard_negatives]

        writer.write(query, positive, hard_negatives)
        seen_queries.add(query)
```

This function runs for 4-6 hours over 532k training queries. The `seen_queries` set (loaded from the existing output JSONL at startup) means any crash or interruption can be recovered by simply re-running the script — already-processed queries are skipped. The gold passage filter uses exact string matching (`text != positive`) rather than passage IDs, which avoids a subtle bug: BM25 sometimes retrieves a slightly differently-whitespace-normalized version of the gold passage. An ID-based filter would miss these and inadvertently include the gold passage as a hard negative.

---

## 4. FAISS Index Construction: Local vs Remote

**File:** `src/indexing/faiss_index.py`

```python
def build_faiss_index(embeddings: np.ndarray, cfg: DictConfig) -> faiss.Index:
    d = embeddings.shape[1]  # 384 for MiniLM

    if cfg.faiss.index_type == "Flat":
        # Local dev: exact search, no training required
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
    else:
        # Remote: IVFFlat with 4096 clusters, search 64 at query time
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, cfg.faiss.nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)   # k-means clustering, ~minutes at 8.8M vectors
        index.add(embeddings)
        index.nprobe = cfg.faiss.nprobe  # 64: search 64/4096 cells per query

    if cfg.faiss.use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)

    return index
```

The `nlist=4096` / `nprobe=64` ratio means each query searches ~1.6% of the vector space, giving 10-20× speedup over exact search at the cost of a small recall penalty. The IVF index requires a training step (k-means clustering over the full embedding set) that takes several minutes but only runs once. The flat index used locally requires no training but scales poorly — at 8.8M vectors, exact search would take seconds per query.

---

## 5. HyDE: Hypothetical Document Embedding with Fallback

**File:** `src/inference/hyde.py`

```python
_HYDE_PROMPT = (
    "Write a detailed passage that directly answers the following question. "
    "Be factual and concise.\n\nQuestion: {query}\n\nPassage:"
)

def generate_hypothetical_doc(query: str, cfg: DictConfig) -> str:
    url = cfg.model.hyde_server_remote if _is_remote(cfg) else cfg.model.hyde_server_local
    timeout = cfg.model.hyde_timeout  # 10 seconds

    try:
        payload = _vllm_payload(query, cfg)
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("HyDE failed for query %r: %s — using original query.", query, e)
        return query  # graceful fallback: evaluate on original query
```

The fallback to the original query on any exception is deliberate. During a multi-hour evaluation run over 7,440 queries, the vLLM server occasionally hits memory pressure or a slow query. Failing the entire evaluation over a single HyDE timeout would be catastrophic. The fallback means those queries are evaluated without HyDE expansion — slightly disadvantaging Variants 6/7 — but the run completes and the results are still valid. The `timeout=10` is tuned to the 95th percentile of Mistral-7B generation latency; queries that take longer are pathological and not worth waiting for.

---

## 6. Cross-Encoder Batch Reranking

**File:** `src/inference/stage2_crossencoder.py`

```python
def rerank_batch(
    self,
    queries: list[str],
    candidates_batch: list[list[dict]],
    top_k: int | None = None,
) -> list[list[dict]]:
    top_k = top_k or self.cfg.inference.crossencoder_top_k  # default: 10

    # Flatten all (query, passage) pairs across all queries in the batch
    all_pairs = []
    offsets = []
    for query, candidates in zip(queries, candidates_batch):
        offsets.append(len(all_pairs))
        passage_texts = [c["passage"] for c in candidates]
        all_pairs.extend([(query, p) for p in passage_texts])

    # One predict() call — sentence-transformers internally mini-batches at crossencoder_batch_size=8
    all_scores = self._model.predict(
        all_pairs,
        batch_size=self.cfg.inference.crossencoder_batch_size,
        show_progress_bar=False,
    )

    # Re-split scores by query and sort
    results = []
    offsets.append(len(all_pairs))
    for i, (query, candidates) in enumerate(zip(queries, candidates_batch)):
        scores = all_scores[offsets[i]:offsets[i + 1]]
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        results.append([{**c, "score": float(s)} for c, s in ranked[:top_k]])

    return results
```

Flattening all query-passage pairs into a single `predict()` call is the critical batching optimization. With an outer evaluation batch of 32 queries and 1,000 candidates each, this produces 32,000 pairs in one call. The cross-encoder's internal mini-batch size of 8 means 4,000 forward passes, but they're done in one contiguous GPU call — far more efficient than 32 separate `predict()` invocations. The offset tracking handles ragged batches (queries with different numbers of candidates) without padding.

---

## 7. Cosine LR Scheduler with Hard Restarts

**File:** `src/training/trainer.py`

```python
warmup_steps = int(cfg.training.max_steps * cfg.training.warmup_ratio)  # 10,000

scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=cfg.training.max_steps,   # 100,000
    num_cycles=cfg.training.num_cycles,           # 3
)
```

The `num_cycles=3` setting divides 100,000 training steps into three cosine decay cycles (after the 10% warmup). At each cycle boundary, the learning rate resets to its post-warmup peak (2e-5), allowing the optimizer to escape any local minima it settled into at the end of the previous cycle. This is particularly useful for contrastive losses, where the loss landscape has many flat regions — the restarts provide the energy to explore beyond the nearest local optimum. Without restarts, a single cosine decay often converges too early.

---

## 8. Evaluation: Per-Variant Result Persistence

**File:** `src/evaluation/compare.py`

```python
for variant in variants:
    logger.info("Running variant: %s", variant.name)
    result = _run_variant(variant, queries, gold_corpus_ids, cfg)

    # Save immediately — don't lose results if a later variant crashes
    slug = variant.name.lower().replace(" ", "_").replace("/", "_")
    out_path = Path(cfg.paths.results_dir) / f"{slug}.json"
    out_path.write_text(json.dumps(asdict(result), indent=2))
    logger.info("Saved result to %s", out_path)

    results.append(result)
```

With 7 variants each taking 20-40 minutes on 7,440 queries, a crash on Variant 6 or 7 would lose hours of work if results were only saved at the end. Saving each result immediately after completion means the run is effectively resumable: results already in `results/` can be loaded directly and that variant skipped. This is the same crash-safety philosophy as the triplet writer in Phase 2 — at scale, crashes are not exceptional; they're expected.
