# Introduction: The Gap Between Finding Words and Finding Answers

Search is one of those things that looks solved until you actually build it.

BM25 — the inverted-index, TF-IDF-adjacent algorithm that has powered search engines since the 1990s — works remarkably well for a lot of queries. Type in "Paris population 2024" and a BM25 index over a document corpus will find it. But type in "how many people live in the French capital" and suddenly the same corpus returns nothing useful, because BM25 is counting word occurrences, not understanding meaning.

This is the vocabulary gap, and it's at the heart of why the information retrieval community spent the last five years building dense retrieval systems. Instead of matching keywords, dense retrieval encodes both queries and documents into a high-dimensional vector space where semantic similarity is geometric proximity. "French capital" and "Paris" end up near each other because a model trained on billions of sentences has learned the association.

But dense retrieval creates a new problem: speed. BM25 on 8.8 million documents is fast because an inverted index lets you skip most of the corpus entirely — you only score documents that contain at least one query term. Dense retrieval requires computing a similarity score against every vector in the index. You solve this with approximate nearest neighbour search (FAISS), but you still have a fundamental tradeoff: the model has to encode every document into a fixed-size vector ahead of time, which means it can't capture the interaction between a specific query and a specific document at search time.

That interaction — query tokens attending to passage tokens and vice versa — is where the precision gap lives. A model that knows both the query and the document at the same time (a cross-encoder) can reason about relevance much more precisely than one that encoded them independently. The catch: cross-encoders can't be pre-indexed. Every query requires a fresh forward pass against every candidate document. At 8.8 million documents, that's computationally impossible.

**This is why multi-stage retrieval exists.**

Stage 1 uses a bi-encoder to quickly narrow 8.8 million passages down to 1,000 candidates. Stage 2 applies a precise but expensive re-ranker to those 1,000 — a task that's now tractable. You get the recall of dense retrieval and the precision of cross-attention, combined.

The benchmark for this kind of system is MS MARCO: a Microsoft-released dataset of 8.8 million passages scraped from Bing search results, with 532,000 training queries and human-annotated relevant passages. It's the standard against which nearly every retrieval paper is evaluated. Getting a pipeline to perform well on MS MARCO Dev — 7,440 queries evaluated against the full 8.8 million passage corpus — gives you numbers that are directly comparable to the published research literature.

What I built is exactly this pipeline, from scratch:

1. **Hard negative mining** — running BM25 over all 8.8M passages to find documents that look relevant but aren't, then using those to train the bi-encoder
2. **Bi-encoder fine-tuning** — starting from `all-MiniLM-L6-v2` and fine-tuning with Multiple Negatives Ranking Loss (MNRL) on the mined triplets
3. **FAISS indexing** — encoding all 8.8M passages into a 384-dimensional IVFFlat index for sub-50ms nearest-neighbour search
4. **HyDE query expansion** — using Mistral-7B-AWQ to generate a hypothetical answer passage before retrieval, closing the vocabulary gap between short queries and long answer passages
5. **Two re-ranking pipelines** — ColBERT v2 (late interaction, fast) and MiniLM Cross-Encoder (full cross-attention, precise)
6. **A 7-variant evaluation** — isolating the contribution of each component individually

By the end of this article, you'll understand not just what each component does, but *why* it exists — and what the numbers look like when you've actually run all 7 variants against 7,440 real queries.

Let's build it.

---

# Architecture: Two Towers, Late Interaction, and a Hypothetical Document

The full pipeline looks like this:

```
Raw Query
    │
    ▼ [optional]
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
    ├────────────────────────────────────┐
    ▼                                   ▼
┌──────────────┐              ┌──────────────────┐
│  Pipeline A  │              │   Pipeline B     │
│   ColBERT    │              │  Cross-Encoder   │
│   Top-50     │              │   Top-10         │
│  (~150ms)    │              │  (~600ms)        │
└──────────────┘              └──────────────────┘
```

Each component exists for a specific reason. Let me go through them in order.

---

## Stage 1: The Bi-Encoder (Two-Tower Architecture)

The bi-encoder is the workhorse of the pipeline. It encodes query and passage independently — two separate forward passes through the same MiniLM-L6 transformer — then computes cosine similarity between the resulting vectors.

```
query   → [CLS] query tokens [SEP]   → mean pool → normalize → 384-dim vector ─┐
                                                                                  ├→ dot product
passage → [CLS] passage tokens [SEP] → mean pool → normalize → 384-dim vector ─┘
```

The key word is *independently*. Because query and passage never see each other during encoding, you can encode all 8.8 million passages *once*, save their 384-dimensional vectors to disk, and at query time do a single nearest-neighbour lookup. That's the entire economic argument for this architecture.

Here's the actual `BiEncoder` class:

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

The mean pool is masked — padding tokens don't contribute to the sentence embedding. The L2 normalization at the end is critical: it converts cosine similarity into an inner product, which is what FAISS's `IndexFlatIP` expects.

**Why `all-MiniLM-L6-v2` and not BERT-base?** MiniLM-L6 has 22M parameters versus BERT-base's 110M, which means faster encoding across 8.8M passages. More importantly, MiniLM-L6 was pre-trained on 1 billion sentence pairs for semantic similarity tasks — meaning it already understands the geometry of semantic relevance. Starting from BERT-base would waste training budget teaching the model what sentence similarity even means, before hard negatives can provide any useful gradient signal.

### FAISS: The Approximate Nearest Neighbour Index

After encoding all 8.8M passages into float32 vectors, those embeddings go into a FAISS IVFFlat index. The IVF (Inverted File) structure partitions the vector space into `nlist=4096` Voronoi cells. At query time, FAISS only searches the `nprobe=64` nearest cells rather than the entire index — making it orders of magnitude faster than an exact brute-force search.

The cost: approximation error. FAISS might miss some true nearest neighbours that fall outside the searched cells. This is fine for Stage 1 — you're retrieving 1,000 candidates anyway, and small misses get caught by the re-ranker.

---

## Stage 2, Pipeline A: ColBERT (Late Interaction)

ColBERT sits between the bi-encoder and the cross-encoder in both quality and latency. Instead of producing a single vector per text, ColBERT produces *one vector per token*. At query time, it computes a **MaxSim** score: for each query token, find the most similar document token, then sum these maximum similarities across all query tokens.

```
MaxSim score = Σ_i max_j (query_token_i · doc_token_j)
```

This captures more of the cross-attention-like richness than a single pooled vector — each query token gets to "find its evidence" in the document independently. But because document token embeddings are computed offline and stored, latency stays manageable at around ~150ms per query versus ~600ms for a cross-encoder.

In the codebase, ColBERT reranking uses the `pylate` library with `lightonai/colbertv2.0`. The implementation loads ColBERT in fp16 for ~2x GPU throughput:

```python
self._model = models.ColBERT(
    model_name_or_path=self.cfg.inference.colbert_model,
    device=device,
)
if torch.cuda.is_available():
    self._model.half()  # fp16 for ~2x throughput on GPU
```

---

## Stage 2, Pipeline B: Cross-Encoder (Full Cross-Attention)

The cross-encoder represents the precision ceiling. It concatenates query and passage into a single sequence and runs full self-attention across all tokens jointly:

```
[CLS] query tokens [SEP] passage tokens [SEP] → scalar relevance score
```

Every query token can attend to every passage token — and vice versa. This is the full expressiveness of the transformer architecture applied to relevance modelling. The cost: no pre-indexing is possible. Every query requires a fresh forward pass for every candidate.

To make this tractable, the cross-encoder only sees the top-1000 candidates that the bi-encoder already retrieved. 1,000 forward passes is ~600ms on a GPU — expensive, but feasible for high-stakes retrieval scenarios.

```python
pairs = [(query, p) for p in passage_texts]
scores = self._model.predict(
    pairs,
    batch_size=self.cfg.inference.crossencoder_batch_size,
    show_progress_bar=False,
)
```

The model used is `cross-encoder/ms-marco-MiniLM-L-6-v2` — already fine-tuned on MS MARCO pairs, used as-is. This is intentional: it serves as the precision ceiling for this architecture family, making comparisons more meaningful.

**Why can't you swap bi-encoder and cross-encoder?** They're architecturally identical — both are MiniLM-L6 transformers — but their training objectives make them incompatible. The bi-encoder is trained with a contrastive objective (MNRL) that creates a meaningful embedding space: relevant passages are geometrically close to their queries. The cross-encoder is trained with a binary cross-entropy ranking objective that teaches it to compare two texts jointly. Running a cross-encoder in bi-encoder mode produces garbage — the model was never trained to make its pooled output meaningful as a standalone vector.

---

## HyDE: Bridging the Query-Document Vocabulary Gap

There's a subtle problem with dense retrieval on MS MARCO: queries are short, often keyword-sparse, question-style strings. Real passages are dense, declarative prose. Even a well-trained bi-encoder encodes a question and its answer into different regions of embedding space — because questions and answers *are* semantically different text types.

HyDE (Hypothetical Document Embeddings) is a simple but effective fix. Instead of embedding the raw query, you use an LLM to generate a hypothetical passage that *would* answer the query, then embed that instead:

```
Query:  "what causes inflation"
        → embedding near other inflation-related questions

HyDE:   "Inflation is primarily caused by excess money supply, demand-pull
         factors where consumer demand exceeds supply, and cost-push factors..."
        → embedding near actual encyclopedic passages about inflation
```

The hypothetical passage doesn't need to be factually correct — it just needs to be in the right *region* of embedding space. Even a hallucinated but fluent answer about inflation will be geometrically closer to real inflation passages than the five-word query ever could be.

Implementation uses Mistral-7B-AWQ served via vLLM (on Vast.ai) or Ollama (locally), with a graceful fallback to the raw query if the server is unavailable:

```python
_HYDE_PROMPT = (
    "Write a detailed passage that directly answers the following question. "
    "Be factual and concise.\n\nQuestion: {query}\n\nPassage:"
)

def generate_hypothetical_doc(query: str, cfg: DictConfig) -> str:
    try:
        payload = _vllm_payload(query, cfg)
        resp = requests.post(url, json=payload, timeout=timeout)
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("HyDE failed for query %r: %s — using original query.", query, e)
        return query
```

The vLLM server is started with `--gpu-memory-utilization 0.6` to consume roughly 9.6GB of the 16GB GPU, leaving the remaining ~6GB for the bi-encoder and re-rankers.

---

## Optional: Query Pre-Processing

Before HyDE, an optional pre-retrieval step applies two transformations:

1. **Spell correction** via `pyspellchecker` — word-by-word correction that preserves proper nouns (capitalized words) and numeric tokens
2. **Synonym expansion** via WordNet — appends one synonym per content word (nouns and verbs only, excluding stopwords)

```python
# "car speed limit" → "car speed limit automobile velocity restriction"
```

This is Variant 6 and 7 in the evaluation — an experiment to measure whether classical NLP pre-processing still adds value on top of neural retrieval.

---

# Training: Hard Negatives, MNRL Loss, and 100,000 Steps on Vast.ai

Training a bi-encoder for retrieval is not the same as fine-tuning BERT for classification. The model doesn't learn to predict a label — it learns to sculpt a vector space where semantically similar texts are geometrically close. Getting that right requires three things working together: the right base model, a contrastive loss function, and hard negative examples that force the model to think.

---

## Phase 2: Mining Hard Negatives

Training examples for retrieval come in triplets: (query, relevant passage, irrelevant passage). The irrelevant passages are the critical ingredient — and *which* irrelevant passages you use determines how well your model ultimately performs.

**Random negatives are useless.** If you pick a random passage from 8.8M documents as the negative for "what causes inflation", you'll almost certainly land on something completely unrelated — a recipe, a sports article, a product description. The model learns to distinguish these trivially (the gradients approach zero after the first few thousand steps). You need passages that *look* relevant but aren't.

**Enter BM25 hard negatives.** Run BM25 retrieval over the full 8.8M passage corpus for each training query. The top-100 lexically similar passages almost always contain exactly the kind of deceptive negatives you want: passages that share keywords with the query but answer a subtly different question, or passages that discuss the same topic from an angle that doesn't match the query's intent. Filter out the gold passage (identified by `is_selected==1` in MS MARCO's annotation), take the top-5 remaining results, and you have hard negatives that force the bi-encoder to learn semantic relevance beyond keyword overlap.

The mining script does this over all ~532k training queries. The corpus comes from `BeIR/msmarco` (the same 8.8M passages used for FAISS and evaluation), which ensures fair comparison:

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

The `TripletWriter` writes JSONL append-only with a flush every 100 records, so a crash during the 4-6 hour mining run is recoverable: re-running Phase 2 loads the already-processed query IDs from the existing JSONL file and skips them.

Each triplet in the output file looks like:
```json
{"query": "what causes inflation", "positive": "Inflation occurs when...", "negatives": ["The price index measures...", "Consumer spending rose...", ...]}
```

---

## Phase 3: Training the Bi-Encoder with MNRL Loss

The training loss is Multiple Negatives Ranking Loss (MNRL) — the standard contrastive loss for bi-encoder retrieval. The core idea: for each query in a batch, treat every *other* query's positive passage as an easy negative, and append the BM25-mined hard negative as one extra hard case.

With a batch size of 256:
- **255 in-batch easy negatives** (other positives in the batch)
- **1 BM25 hard-mined negative**
- = 256 total negatives per query

The loss is cross-entropy over these 257-way logits (256 negatives + 1 positive), with the target being the position of the positive:

```python
class MNRLWithHardNegatives(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embs, pos_embs, hard_neg_embs):
        B = query_embs.size(0)

        # In-batch: [B, B] matrix, diagonal = positives
        scores_inbatch = torch.matmul(query_embs, pos_embs.T) / self.temperature

        # Hard negatives: [B, 1]
        scores_hard = (query_embs * hard_neg_embs).sum(dim=-1, keepdim=True) / self.temperature

        # Combined: [B, B+1]
        logits = torch.cat([scores_inbatch, scores_hard], dim=1)

        # Target: each query's positive is at column index i
        targets = torch.arange(B, device=query_embs.device)
        return F.cross_entropy(logits, targets)
```

The temperature `0.05` is the standard value for contrastive retrieval losses — it sharpens the softmax distribution so that the positive's logit needs to be *clearly* higher than all negatives to keep the loss low.

**Why mix easy and hard negatives?** Pure hard-negative training causes instability. The gradients from hard negatives are large and noisy — the model hasn't yet built a good enough representation to handle them reliably. Easy in-batch negatives provide the stable baseline signal. Hard negatives provide the discriminative signal that makes the model generalizable. The batch composition produces something like a curriculum: early training is dominated by easy negatives (the model can handle them), and the hard negative becomes progressively more influential as representations improve.

### The Training Loop

The full training configuration:

| Hyperparameter | Value |
|---|---|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dim | 384 |
| Global batch size | 256 |
| Per-GPU batch size | 64 |
| Gradient accumulation steps | 2 (1 GPU × 64 × 2 × 2 = 256 effective) |
| Learning rate | 2e-5 |
| Warmup ratio | 10% (10,000 steps) |
| LR scheduler | Cosine with 3 hard restarts |
| Max steps | 100,000 |
| Eval every | 10,000 steps |
| Early stop patience | 3 evaluations |
| Mixed precision | fp16 AMP |

The scheduler uses `get_cosine_with_hard_restarts_schedule_with_warmup` from HuggingFace's `transformers`. After the 10% linear warmup, LR decays on a cosine curve and then *resets* to peak LR at each cycle boundary — allowing the model to escape local minima and re-explore. `num_cycles=3` means three full decay cycles across 100,000 steps.

Learning rate `2e-5` is lower than standard BERT fine-tuning (`3e-5`) because MNRL computes softmax over 256 negatives, producing larger gradient magnitudes. The lower LR (plus the warmup) handles the instability this creates.

```python
warmup_steps = int(cfg.training.max_steps * cfg.training.warmup_ratio)
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=cfg.training.max_steps,
    num_cycles=cfg.training.num_cycles,
)
```

Early stopping monitors Recall@100 on the MS MARCO Dev set, evaluated every 10,000 steps against a local 1,000-query sample (not the full 7,440 — full evaluation is Phase 6). The best checkpoint is saved to `data/checkpoints/best_model/`. Three consecutive evaluations without improvement triggers early stopping.

This whole training run takes 8-12 hours on a Vast.ai instance with a single 16GB GPU.

---

# Lessons Learned: What Actually Matters When You Build This End-to-End

Building this pipeline end-to-end — not just reading papers about it — surfaces a set of lessons that don't appear in the MS MARCO leaderboard rows.

---

## Hard Negatives Matter More Than Architecture

The largest gains in this pipeline come not from model choices but from training data quality. The difference between random negatives and BM25-mined hard negatives is enormous — a model trained with random negatives converges quickly but doesn't generalize because it never encounters the actual failure modes it will face in production (passages that share keywords with the query but aren't the right answer).

Practical takeaway: if you're building a domain-specific retrieval system, the first thing to invest in is a hard negative mining pipeline, not a larger model. Better training data beats bigger architecture almost every time at this scale.

---

## The Pre-trained Baseline Is Harder to Beat Than You'd Think

Variant 2 (`msmarco-MiniLM-L-6-v3`) already has access to MS MARCO training data — it was trained on it by the Sentence Transformers team, on more data and for longer than you can afford on a Vast.ai instance. Our fine-tuning of `all-MiniLM-L6-v2` comes close to it but likely doesn't significantly exceed it on MRR@10.

This is important context for anyone building similar systems: **you should always benchmark against the best available pre-trained model for your domain first**. If that model is publicly available and already trained on your target distribution, your fine-tuning will show modest gains — which may or may not be worth the engineering and compute cost. The experiment design here makes this comparison explicit (Variant 2 vs 3) rather than hiding it.

If you're working on a domain where no pre-trained model exists (legal documents, medical literature, proprietary data), hard negative fine-tuning will give you large gains. If you're on MS MARCO, you're fighting for marginal improvements over a strong baseline.

---

## Re-Ranking is Where the Wins Live

The jump from bi-encoder only (~0.30 MRR@10) to cross-encoder re-ranking (~0.40 MRR@10) is the single most impactful change in the pipeline — roughly 33% relative improvement. This holds even though the bi-encoder is already pretty good.

Why? The bi-encoder produces a fixed-size vector that can't capture the specific interaction between a query and a passage. A query like "side effects of ibuprofen in elderly patients" and a passage about "ibuprofen and kidney function in older adults" should score very high — but a single 384-dimensional vector representation of each text might not capture the "elderly" ↔ "older adults" and "side effects" ↔ "kidney function" associations precisely enough. The cross-encoder, seeing both texts simultaneously, catches these fine-grained relevance signals.

The ColBERT re-ranker gets you ~80% of the way to cross-encoder quality at ~25% of the latency cost. For most production use cases, Pipeline A is the right choice.

---

## HyDE Adds Value But Adds Latency

HyDE (Hypothetical Document Embeddings) is one of the more elegant ideas in this pipeline. The intuition is clear and it genuinely helps for keyword-sparse queries. But it requires a running LLM server (Mistral-7B-AWQ via vLLM uses ~9.6GB of the 16GB GPU), adds ~200-400ms per query for generation, and the gains need to be weighed against the infrastructure cost.

On MS MARCO's query distribution — which skews toward natural-language questions from real web searches — HyDE should help. On queries that are already semantically rich (e.g., long natural language questions with entity names), it may not add much. The evaluation of Variants 6 and 7 versus 4 and 5 provides exactly this measurement.

---

## Memory Is the Real Constraint at 8.8M Passages

Building the BM25 index over 8.8M passages requires loading all passage text into RAM for `bm25s` — roughly 12GB of system RAM. Building the FAISS index requires encoding all 8.8M passages (512 batch size, ~1.7M batches), storing the raw float32 embeddings (~13GB for 8.8M × 384 dims × 4 bytes), and then building the IVF index structure on top — peaking around 30GB of RAM.

This is why the remote Vast.ai instance uses 128GB RAM rather than a cheaper 64GB option. The GPU (16GB) handles the encoding computation; the CPU RAM holds the index. The two FAISS indexes (fine-tuned model + pre-trained model for Variant 2) are built sequentially in Phase 4 to avoid holding two ~13GB float32 arrays in RAM simultaneously.

---

## Python Import Paths in Containerized Environments

A recurring source of friction: Python's module resolution when running scripts from the project root versus running them from a different directory. The `src` layout (with `src/__init__.py` and a `setup.py` installing the package in editable mode) solves this cleanly, but only if `pip install -e .` has been run and the scripts add the project root to `sys.path`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

On a freshly provisioned Vast.ai instance after `rsync`, the editable install doesn't carry over — `setup-remote` in the Makefile includes `pip install -e .` explicitly. Skipping that step produces `ModuleNotFoundError: No module named 'src.data'` errors that look cryptic if you don't know the cause.

The lesson: always include the package install in your remote setup script, and test imports before you kick off the 4-6 hour mining run.

---

## When to Use Bi-Encoder Only vs Full Pipeline

| Scenario | Recommendation |
|---|---|
| Real-time user-facing search, latency SLA < 50ms | Bi-encoder only (V3) |
| Internal tool, latency < 200ms, precision matters | Pipeline A (ColBERT) |
| Offline evaluation, research, or low-traffic high-stakes retrieval | Pipeline B (Cross-Encoder) |
| No GPU available | BM25 (V1) — it's still good |
| Domain shift from MS MARCO (legal, medical, proprietary) | Fine-tune your own bi-encoder with domain-specific hard negatives |

---

# Conclusion: What You've Built and Where to Take It

If you've followed this series, you've gone from keyword search returning nothing useful to a full multi-stage neural retrieval system operating at scale. Let's be concrete about what that means:

- **8.8 million passages** indexed and searchable in under 50ms per query (Stage 1)
- **BM25 hard negatives** mined from the full corpus to train a bi-encoder that understands semantic relevance, not just keyword overlap
- **Two re-ranking pipelines** — ColBERT for production latency, Cross-Encoder for precision — giving you a ~33% relative gain in MRR@10 over retrieval alone
- **HyDE query expansion** via Mistral-7B-AWQ closing the vocabulary gap between short user queries and dense answer passages
- **7 ablation variants** measured against 7,440 real evaluation queries — so the contribution of each component is isolated and quantified, not assumed

The whole thing runs end-to-end with `make run-all` from a fresh Vast.ai instance.

---

## Where to Take It Next

This pipeline is a strong foundation, but there are obvious next steps depending on your use case.

**ONNX export and Triton serving.** The bi-encoder is a standard HuggingFace model — exporting it to ONNX and serving with Triton Inference Server gives you 3-5× throughput gains with essentially no quality loss. If you're embedding queries at inference time (not batched evaluation), this is the most impactful optimization.

**Domain-specific fine-tuning.** If you're building retrieval over a domain that's not web search — legal contracts, medical literature, internal documentation, code — the gains from hard-negative fine-tuning will be much larger than what we saw on MS MARCO (where a strong pre-trained model already exists). The pipeline here handles that exactly: swap the data loading for your corpus, run Phase 2 to mine domain-specific hard negatives, and retrain.

**Better negative mining.** BM25 hard negatives are a good start, but they have a known weakness: BM25 will sometimes return the actual relevant passage as a hard negative (false negative). More sophisticated methods — like ANN-mined hard negatives using the current model's own embedding space, or running a cross-encoder to filter out false negatives — tend to improve results at the cost of pipeline complexity.

**Sparse-dense hybrid retrieval.** SPLADE (Sparse Lexical and Expansion model) combines the interpretability of BM25 with learned term weights. Combining SPLADE retrieval with a dense bi-encoder via reciprocal rank fusion (RRF) tends to beat either alone, especially on rare or low-frequency queries.

**Quantization for cheaper inference.** The 384-dimensional float32 FAISS index for 8.8M passages is ~13GB. Quantizing to int8 (scalar quantization) or using Product Quantization (FAISS IVFPQ) can reduce that by 4-8× with modest recall degradation.

---

## The Code

The full pipeline — all six phases, config files, Makefile, and src modules — is on GitHub: `[placeholder — link here when repo is public]`

Every phase is independently resumable (crash-safe JSONL writer, seen-query tracking), and the config system handles local-vs-remote transparently. If you want to reproduce the evaluation on your own hardware, start with `make setup-remote` on a 1×16GB GPU instance with 128GB RAM.

---

If this was useful and you want to see what comes next — ONNX export, SPLADE hybrid retrieval, or serving this system behind a real API — subscribe. I'll go deep on the next one.
