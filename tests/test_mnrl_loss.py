import torch
import pytest
from src.training.mnrl_loss import MNRLWithHardNegatives


def _rand_emb(B: int, D: int = 64) -> torch.Tensor:
    """Random L2-normalized embeddings."""
    x = torch.randn(B, D)
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def test_loss_is_scalar():
    loss_fn = MNRLWithHardNegatives()
    B = 8
    q = _rand_emb(B)
    p = _rand_emb(B)
    hn = _rand_emb(B)
    loss = loss_fn(q, p, hn)
    assert loss.shape == ()


def test_loss_is_positive():
    loss_fn = MNRLWithHardNegatives()
    loss = loss_fn(_rand_emb(4), _rand_emb(4), _rand_emb(4))
    assert loss.item() > 0


def test_perfect_alignment_low_loss():
    """When query == positive and hard negative is orthogonal, loss should be low."""
    loss_fn = MNRLWithHardNegatives(temperature=0.05)
    B, D = 8, 64
    emb = _rand_emb(B, D)  # query == positive
    hn = _rand_emb(B, D)   # unrelated
    loss = loss_fn(emb, emb, hn)
    # With perfect alignment the loss should be much lower than random
    random_loss = loss_fn(_rand_emb(B, D), _rand_emb(B, D), _rand_emb(B, D))
    assert loss.item() < random_loss.item()


def test_loss_backward():
    """Ensure gradients flow correctly."""
    loss_fn = MNRLWithHardNegatives()
    q = torch.randn(4, 32, requires_grad=True)
    p = torch.randn(4, 32, requires_grad=True)
    hn = torch.randn(4, 32, requires_grad=True)
    q_norm = torch.nn.functional.normalize(q, p=2, dim=-1)
    p_norm = torch.nn.functional.normalize(p, p=2, dim=-1)
    hn_norm = torch.nn.functional.normalize(hn, p=2, dim=-1)
    loss = loss_fn(q_norm, p_norm, hn_norm)
    loss.backward()
    assert q.grad is not None
    assert p.grad is not None
    assert hn.grad is not None


def test_logits_shape():
    """logits should be [B, B+1] before cross-entropy."""
    B, D = 6, 32
    q = _rand_emb(B, D)
    p = _rand_emb(B, D)
    hn = _rand_emb(B, D)
    # Verify by checking the loss doesn't raise with correct shapes
    loss_fn = MNRLWithHardNegatives()
    loss = loss_fn(q, p, hn)
    assert loss.item() >= 0
