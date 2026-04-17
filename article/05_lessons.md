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
