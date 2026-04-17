# Evaluation Results

```
Variant                                    MRR@10   Recall@100    Latency(ms)
-----------------------------------------------------------------------------
BM25 baseline                              0.1364       0.5181           94.6
Pre-trained MS MARCO bi-encoder            0.2961       0.7669           19.0
Fine-tuned bi-encoder only                 0.1344       0.2728            5.9
Pipeline A: ColBERT                        0.2886       0.5968         1511.5
Pipeline B: Cross-Encoder                  0.2928       0.4917          666.8
Pipeline A + Query Rewriting               0.2499       0.5181         1504.8
Pipeline B + Query Rewriting               0.2536       0.4245          660.2
RRF: BM25 + Fine-tuned                     0.2010       0.6966          111.4
RRF: BM25 + Pretrained                     0.2306       0.8027           81.7
RRF: BM25 + Fine-tuned → ColBERT → Cross-Encoder   0.3848       0.6542         1662.9
RRF: BM25 + Pretrained → ColBERT → Cross-Encoder   0.3921       0.6765         1606.1
```
