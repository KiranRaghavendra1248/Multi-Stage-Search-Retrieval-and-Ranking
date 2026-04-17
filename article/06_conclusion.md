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
