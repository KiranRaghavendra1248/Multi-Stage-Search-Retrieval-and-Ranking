# Tokenization, Collation, and the Training Data Pipeline

How raw text becomes gradients — from string to embedding to loss.

---

## AutoTokenizer (HuggingFace)

`AutoTokenizer` is a HuggingFace `transformers` construct, not native PyTorch. PyTorch has no built-in text tokenizer — it only handles tensors. Tokenization (text → token IDs) is always an external library.

`AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")` reads the model card metadata and loads the correct tokenizer automatically. For MiniLM it loads a WordPiece tokenizer (same family as BERT).

**What it does:**
```
"what causes inflation"
→ ["what", "causes", "inflation"]      # split into subwords
→ [101, 2054, 5320, 10904, 102]        # map to vocabulary IDs
                                        # 101 = [CLS], 102 = [SEP]
```

**What `tokenizer(texts, return_tensors="pt")` returns:**

A `BatchEncoding` — a dict subclass with:
```python
{
    "input_ids":      torch.LongTensor [B, seq_len],
    "attention_mask": torch.LongTensor [B, seq_len],  # 1=real token, 0=padding
    "token_type_ids": torch.LongTensor [B, seq_len],  # all 0s for single sequences
}
```

These are just integer indices, not embeddings yet. The conversion to vectors happens inside the model's first layer.

---

## Where Token Embeddings Are Created

The embedding lookup happens in the very first layer inside `AutoModel`, before any transformer blocks run.

**What happens when you call `AutoModel(**enc)`:**

```
input_ids [B, seq_len]
    ↓
Embedding layer (inside BERT/MiniLM):
    Token Embedding     → lookup table [vocab_size=30522, 384]  → [B, seq_len, 384]
  + Position Embedding  → lookup table [max_pos=512, 384]       → [B, seq_len, 384]
  + Token Type Embedding→ lookup table [2, 384]                 → [B, seq_len, 384]
  ─────────────────────────────────────────────────────────────────────────────────
    Sum of all three                                            → [B, seq_len, 384]
    LayerNorm + Dropout
    ↓
Transformer blocks (6 layers for MiniLM-L6):
    Self-attention + FFN × 6
    ↓
last_hidden_state [B, seq_len, 384]
```

**The embedding lookup is literally a `nn.Embedding`:**
```python
nn.Embedding(30522, 384)  # vocab_size × hidden_dim
```
`input_ids` are row indices into this weight matrix. Token ID 2054 → row 2054 → a 384-dim learned vector.

Position embedding works the same way: position index 0 → row 0, position index 1 → row 1. This is how the model knows token order (transformers have no inherent sense of sequence order without this).

All three embeddings are summed element-wise. By the time `last_hidden_state` comes back, each position's vector has been contextualized through 6 transformer layers — it's no longer "what is this token" but "what is this token given all surrounding tokens".

---

## Padding

Different sentences have different lengths:
```
"what causes inflation"                                         → 5 tokens
"how does the federal reserve control money supply in the US"   → 14 tokens
```

These can't stack into a `[B, seq_len]` matrix without padding. `padding=True` in the tokenizer call pads all sequences to the length of the longest one in the batch, appending `[PAD]` tokens.

The `attention_mask` marks which tokens are real (1) vs padding (0). This is exactly why `_mean_pool` uses the mask:

```python
mask_expanded = attention_mask.unsqueeze(-1).float()
sum_emb = (token_embeddings * mask_expanded).sum(dim=1)   # zeroes out pad positions
sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
return sum_emb / sum_mask
```

Without masking, the mean pool would average in the near-zero embeddings from padding tokens and corrupt the representation.

---

## The collate_fn

**Why it exists:**

`DataLoader.__getitem__` returns one sample at a time:
```python
{"query": str, "positive": str, "hard_negatives": [str, str, ...]}
```

When 256 such samples are collected into a batch, PyTorch's default collation can't handle variable-length strings or ragged lists. `collate_fn` is a custom function passed to `DataLoader` that tells it how to merge a list of samples into one batch.

**What collate_fn does:**
```python
def collate_fn(batch):
    queries    = [item["query"] for item in batch]           # 256 strings
    positives  = [item["positive"] for item in batch]        # 256 strings
    hard_negs  = [item["hard_negatives"][0] for item in batch]  # 256 strings (K=1)

    query_enc   = tokenizer(queries,   padding=True, truncation=True, max_length=128, return_tensors="pt")
    pos_enc     = tokenizer(positives, padding=True, truncation=True, max_length=128, return_tensors="pt")
    hard_neg_enc= tokenizer(hard_negs, padding=True, truncation=True, max_length=128, return_tensors="pt")

    return {"query_enc": query_enc, "positive_enc": pos_enc, "hard_neg_enc": hard_neg_enc}
```

Output: three `BatchEncoding` dicts, each with `input_ids` and `attention_mask` of shape `[256, seq_len]`.

**Collate vs inference tokenization:**

| | collate_fn (training) | encode() (inference) |
|---|---|---|
| Input | list of 256 raw strings | list of N raw strings |
| Called by | DataLoader automatically | you call manually |
| Output | BatchEncoding tensors `[B, seq_len]` | numpy array `[N, 384]` |
| Goes to | trainer's forward() passes | FAISS index |

---

## The Full Shape Flow (Training)

```
DataLoader.__getitem__():
    one sample → {query: str, positive: str, hard_negatives: [str,...]}

collate_fn (called on 256 samples):
    tokenizer(queries,   padding=True) → query_enc:    {input_ids [256, seq_len], attention_mask [256, seq_len]}
    tokenizer(positives, padding=True) → pos_enc:      {input_ids [256, seq_len], attention_mask [256, seq_len]}
    tokenizer(hard_negs, padding=True) → hard_neg_enc: {input_ids [256, seq_len], attention_mask [256, seq_len]}

trainer forward passes (3 separate passes, same model weights):
    model(query_enc)    → Embedding → 6× Transformer → mean pool → normalize → q_emb  [256, 384]
    model(pos_enc)      → Embedding → 6× Transformer → mean pool → normalize → p_emb  [256, 384]
    model(hard_neg_enc) → Embedding → 6× Transformer → mean pool → normalize → hn_emb [256, 384]

MNRL loss:
    scores_inbatch = q_emb @ p_emb.T / 0.05   → [256, 256]   (diagonal = positives)
    scores_hard    = (q_emb * hn_emb).sum(-1)  → [256, 1]
    logits         = cat([scores_inbatch, scores_hard], dim=1)  → [256, 257]
    targets        = arange(256)               → each query's positive is at column i
    loss           = cross_entropy(logits, targets)

backprop → gradients → update model weights
```

---

## The Full Shape Flow (Inference)

```
encode(["what causes inflation", ...]):
    tokenizer(batch, padding=True) → input_ids [B, seq_len], attention_mask [B, seq_len]
    .to(device)
    forward() → last_hidden_state [B, seq_len, 384]
              → mean pool → [B, 384]
              → L2 normalize → [B, 384]
    .cpu().numpy() → numpy float32 [B, 384]

FAISS.search(query_emb [1, 384]):
    nearest neighbour lookup → top-1000 passage indices + scores
```
