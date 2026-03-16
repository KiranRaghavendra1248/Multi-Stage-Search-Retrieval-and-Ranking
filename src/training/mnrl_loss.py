import torch
import torch.nn as nn
import torch.nn.functional as F


class MNRLWithHardNegatives(nn.Module):
    """
    Multiple Negatives Ranking Loss with 1 hard negative per query.

    For each query in the batch:
        - Positives:  its own positive passage (diagonal of in-batch matrix)
        - Negatives:  all OTHER positives in the batch (B-1 easy in-batch negatives)
                    + 1 hard-mined negative
        - Total negatives: B - 1 + 1 = B

    Loss = mean cross-entropy where target is the index of the positive.

    Temperature: fixed at 0.05 (standard for contrastive retrieval losses).
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embs: torch.Tensor,    # [B, D]  L2-normalized
        pos_embs: torch.Tensor,      # [B, D]  L2-normalized
        hard_neg_embs: torch.Tensor, # [B, D]  L2-normalized  (1 hard negative per query)
    ) -> torch.Tensor:
        """
        Returns scalar loss.

        Logits matrix:
            scores_inbatch [B, B]  — query_i vs all positives (diagonal = correct pair)
            scores_hard    [B, 1]  — query_i vs its own hard negative
            logits         [B, B+1] — concatenated

        Target: arange(B) — position of the positive in the first B columns.
        """
        B = query_embs.size(0)

        # In-batch similarity: [B, B]
        scores_inbatch = torch.matmul(query_embs, pos_embs.T) / self.temperature

        # Hard negative similarity: [B, 1]
        scores_hard = (query_embs * hard_neg_embs).sum(dim=-1, keepdim=True) / self.temperature

        # Combined logits: [B, B+1]
        logits = torch.cat([scores_inbatch, scores_hard], dim=1)

        # Target: each query's positive is at column index i
        targets = torch.arange(B, device=query_embs.device)

        return F.cross_entropy(logits, targets)
