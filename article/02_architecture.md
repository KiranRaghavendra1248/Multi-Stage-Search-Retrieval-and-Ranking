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
