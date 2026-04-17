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
