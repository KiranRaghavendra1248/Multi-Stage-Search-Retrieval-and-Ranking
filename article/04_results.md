# Results: 7 Variants, 7,440 Queries, 8.8 Million Passages

The evaluation (Phase 6) runs all 7 pipeline variants against the full MS MARCO Dev set: 7,440 queries, each evaluated against all 8.8 million passages in the corpus. Using the full corpus is important — many MS MARCO evaluations use a subset for speed, but results on subsets aren't comparable to the published leaderboard. The gold passage for each query comes from `BeIR/msmarco-qrels`, which provides a `query-id → corpus-id` mapping with no train/dev leakage.

Primary metrics: **MRR@10** (Mean Reciprocal Rank at 10 — how high the correct passage ranks on average) and **Recall@100** (fraction of queries where the correct passage appears somewhere in the top 100 results).

---

## The 7 Variants

Each variant was designed to isolate the contribution of one component.

### Variant 1: BM25 Baseline
Pure lexical retrieval using `bm25s` over all 8.8M passages — no neural components at all. This sets the floor.

**Expected: MRR@10 ~0.18, Recall@100 ~0.67, ~15ms/query**

About 82% of the time, the correct answer is not in the top 10 BM25 results. Recall@100 of 0.67 means BM25 doesn't even find the right passage within its top 100 results 33% of the time. The vocabulary gap is real and large.

### Variant 2: Pre-Trained MS MARCO Bi-Encoder (no fine-tuning)
`sentence-transformers/msmarco-MiniLM-L-6-v3` used as-is — already fine-tuned on MS MARCO by Sentence Transformers, no additional training. This is the **benchmark** that our training pipeline needs to beat to justify its existence.

**Expected: MRR@10 ~0.31, Recall@100 ~0.86, ~30ms/query**

Dense retrieval's leap over BM25 is dramatic: MRR@10 goes from 0.18 to 0.31 — a 72% relative improvement — just by switching from keyword matching to pre-trained semantic embeddings. This model already knows MS MARCO's distribution because it was trained on it. The honest question is whether 8-12 hours of additional hard-negative fine-tuning adds anything.

### Variant 3: Our Fine-Tuned Bi-Encoder Only (Stage 1 ablation)
`all-MiniLM-L6-v2` fine-tuned with MNRL loss and BM25 hard negatives, no re-ranker. This is the Stage 1 retrieval quality in isolation.

**Expected: MRR@10 ~0.30+, Recall@100 ~0.85+, ~30ms/query**

This is the most honest number in the whole experiment. Starting from `all-MiniLM-L6-v2` (not the MS MARCO-specific variant) and adding hard negative fine-tuning, we approach — and should slightly exceed — the pre-trained MS MARCO model on MRR@10. The gap between V2 and V3 tells you exactly what our training contributes on top of what's publicly available.

If V3 < V2, the training pipeline has failed to add value and you should just use the pre-trained model. If V3 ≥ V2, the fine-tuning is working.

### Variant 4: Pipeline A — Fine-Tuned Bi-Encoder + ColBERT (no query rewriting)
Stage 1 top-1000 → ColBERT v2 re-ranks to top-50. Uses `lightonai/colbertv2.0` loaded in fp16.

**Expected: MRR@10 ~0.38, Recall@100 ~0.87, ~150ms/query**

MRR@10 jumps roughly 8 points over the bi-encoder alone. Recall@100 barely moves — re-ranking doesn't expand the candidate set, it just orders it better. The latency cost is ~120ms over Stage 1. ColBERT's late interaction (MaxSim over token embeddings) captures more query-document interaction than a single cosine similarity while still being much faster than full cross-attention.

### Variant 5: Pipeline B — Fine-Tuned Bi-Encoder + Cross-Encoder (no query rewriting)
Stage 1 top-1000 → Cross-Encoder re-ranks to top-10. Full cross-attention on every (query, passage) pair.

**Expected: MRR@10 ~0.40, Recall@100 ~0.87, ~600ms/query**

This is the precision ceiling. Two more MRR@10 points over ColBERT, at 4× the latency. Whether 2 points is worth ~450ms per query depends entirely on your use case. For offline evaluation or very low-traffic search, yes. For a user-facing API at scale, Pipeline A is the practical choice.

### Variants 6 & 7: With Query Rewriting
Variants 6 and 7 add the optional pre-retrieval step: pyspellchecker typo correction + WordNet synonym expansion, then HyDE, then the same Pipeline A or B.

**Expected: TBD — depends on whether synonym expansion helps or hurts on MS MARCO's distribution**

The hypothesis: MS MARCO queries are short and often written casually (they come from real Bing searches), so spelling correction might help, but synonym expansion adds noise for domain-specific terms. These are the most uncertain predictions in the experiment.

---

## What the Numbers Prove

The research questions this experiment answers:

| Question | Variants | Expected answer |
|---|---|---|
| How much does dense retrieval gain over BM25? | V1 vs V3 | ~12 MRR@10 points — the vocabulary gap is real |
| Does our hard-negative training beat the public model? | V2 vs V3 | Within 1-2 points either way — honest evaluation of training value |
| How much does re-ranking improve Stage 1? | V3 vs V4, V5 | +8-10 MRR@10 points from re-ranking alone |
| ColBERT vs Cross-Encoder tradeoff | V4 vs V5 | ~2 MRR@10 points, 4× latency cost |
| Does query rewriting help? | V4 vs V6, V5 vs V7 | Unknown — the interesting finding |

The most instructive comparison is V3 vs V5: starting from bi-encoder-only (~0.30) and adding a cross-encoder re-ranker (~0.40) is a 33% relative gain on MRR@10. This is what motivates the two-stage architecture. The bi-encoder provides recall; the cross-encoder provides precision.

The most honest comparison is V2 vs V3: it tells you what weeks of GPU time actually bought you over just downloading the publicly available model.
