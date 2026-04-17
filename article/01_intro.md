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
