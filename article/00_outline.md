# Article Outline

## Proposed Title Options

1. **"Beyond Keyword Search: Building a Multi-Stage Retrieval Pipeline from Scratch on MS MARCO"** *(technical, precise)*
2. **"How I Built a Production-Grade Neural Search System — and What the Numbers Actually Showed"** *(personal, results-forward)*
3. **"BM25 to Cross-Encoder: A Deep Dive into Multi-Stage Document Retrieval"** *(accessible to ML engineers)*
4. **"Hard Negatives, HyDE, and Late Interaction: The Anatomy of a Modern Search Pipeline"** *(buzzword-rich, signals depth)*
5. **"Dense Retrieval Done Right: Fine-Tuning MiniLM with Hard Negatives on 8.8 Million Passages"** *(SEO-friendly, searchable)*

---

## Target Audience

**Primary:** ML engineers who have used HuggingFace and sentence-transformers but have never trained their own retrieval model from scratch. They know what BERT is, probably know about dense retrieval in the abstract, but haven't wired up BM25 hard negative mining → bi-encoder fine-tuning → cross-encoder re-ranking themselves.

**Secondary:** Technical founders and backend engineers building search, RAG, or recommendation systems who want to understand what's under the hood before reaching for a managed vector database.

**Not for:** NLP researchers (too much tutorial-style exposition) or total beginners (assumes familiarity with transformers and PyTorch).

---

## Key Narrative Arc

The article follows a natural problem-escalation structure:

1. **Problem**: Keyword search (BM25) is fast and interpretable but misses semantic relevance. Pure dense retrieval is better but slow at scale. The real world needs both — a cheap first-pass filter plus an expensive precision layer.
2. **Architecture**: Two-stage pipeline: bi-encoder for recall, re-ranker for precision. Plus optional HyDE query expansion to handle the query-answer vocabulary gap.
3. **Training challenge**: Off-the-shelf embeddings don't understand the MS MARCO distribution well enough. You need hard negatives — documents that look relevant but aren't — to force the model to learn semantic relevance beyond keyword overlap.
4. **Results**: The numbers tell an honest story. Our fine-tuning barely beats the pre-trained MS MARCO model, but adding re-ranking (especially cross-encoder) jumps MRR@10 by 10 absolute points.
5. **Lessons**: What the architecture tradeoffs actually feel like when you've run the pipeline end-to-end on 8.8 million passages and 7,440 evaluation queries.

---

## Section Breakdown

| Section | File | Est. Words | Purpose |
|---|---|---|---|
| Introduction | `01_intro.md` | 500 | Hook on why search is hard, introduce MS MARCO, promise the pipeline |
| Architecture Deep-Dive | `02_architecture.md` | 750 | Two-tower vs cross-attention, HyDE, the full pipeline diagram |
| Training | `03_training.md` | 700 | Hard negative mining, MNRL loss, training loop internals |
| Results & Experiments | `04_results.md` | 500 | 7 variants, MRR@10 numbers, what they prove |
| Lessons Learned | `05_lessons.md` | 450 | Practical advice, tradeoffs, real-world friction |
| Conclusion | `06_conclusion.md` | 250 | Summary, next steps, CTA |
| Code Snippets | `07_code_snippets.md` | ~600 | Curated excerpts with explanations |

**Total: ~3,750 words** — appropriate for a long-form Substack deep-dive.

---

## Hook Ideas for the Opening

- Open with the gap: "Your users type 'how do I fix memory leak' but the relevant document says 'address exhaustion mitigation'. BM25 returns nothing useful. A bi-encoder trained on in-domain data would nail it."
- Open with a number: "8.8 million passages. 7,440 evaluation queries. Five weeks of iteration. Here's what I actually learned building a multi-stage neural search pipeline from scratch."
- Open with the failure: Start with BM25's MRR@10 of 0.18 and work forward — what does it mean that 82% of the time your keyword search doesn't have the right answer in the top 10 results?

---

## Call-to-Action Ideas for Substack

- **Code CTA**: "The full pipeline — mining, training, FAISS indexing, and evaluation — is on GitHub. Every phase is reproducible with a single `make` command."
- **Discussion CTA**: "I'm curious how people handle the query-document vocabulary gap in production. Is HyDE worth the LLM overhead? Share your experience in the comments."
- **Subscription CTA**: "If you want the follow-up where I export the bi-encoder to ONNX and serve it with Triton, subscribe and you'll get it as soon as it's out."
