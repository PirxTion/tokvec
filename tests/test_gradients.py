"""Gradient correctness tests for SGNSModel.

Two independent checks:
1. Numerical gradient (finite differences) — model-agnostic ground truth
2. PyTorch oracle — same architecture, same weights, torch.autograd gradients

Both must agree with our analytical NumPy gradients.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from src.model import SGNSModel


# ── Helpers ──────────────────────────────────────────────────────────────

def _loss(model: SGNSModel, centers, contexts, negatives) -> float:
    return model.forward(centers, contexts, negatives)


def numerical_gradient(model: SGNSModel, centers, contexts, negatives, eps=1e-5):
    """Compute numerical gradients via central differences for W and W_prime."""
    grad_W = np.zeros_like(model.W)
    grad_W_prime = np.zeros_like(model.W_prime)

    for i in range(model.W.shape[0]):
        for j in range(model.W.shape[1]):
            model.W[i, j] += eps
            loss_plus = _loss(model, centers, contexts, negatives)
            model.W[i, j] -= 2 * eps
            loss_minus = _loss(model, centers, contexts, negatives)
            model.W[i, j] += eps
            grad_W[i, j] = (loss_plus - loss_minus) / (2 * eps)

    for i in range(model.W_prime.shape[0]):
        for j in range(model.W_prime.shape[1]):
            model.W_prime[i, j] += eps
            loss_plus = _loss(model, centers, contexts, negatives)
            model.W_prime[i, j] -= 2 * eps
            loss_minus = _loss(model, centers, contexts, negatives)
            model.W_prime[i, j] += eps
            grad_W_prime[i, j] = (loss_plus - loss_minus) / (2 * eps)

    return grad_W, grad_W_prime


def _dense_grads(model, grads):
    """Convert sparse grad dict to dense arrays."""
    dense_W = np.zeros_like(model.W)
    for idx, g in grads["W"].items():
        dense_W[idx] += g
    dense_Wp = np.zeros_like(model.W_prime)
    for idx, g in grads["W_prime"].items():
        dense_Wp[idx] += g
    return dense_W, dense_Wp


class TorchSGNS(nn.Module):
    """Minimal PyTorch SGNS — same math as our NumPy model, for oracle testing."""
    def __init__(self, W_np, W_prime_np):
        super().__init__()
        self.W = nn.Embedding.from_pretrained(
            torch.tensor(W_np, dtype=torch.float64), freeze=False
        )
        self.W_prime = nn.Embedding.from_pretrained(
            torch.tensor(W_prime_np, dtype=torch.float64), freeze=False
        )

    def forward(self, centers, contexts, negatives):
        v_c   = self.W(centers)                          # (B, D)
        u_o   = self.W_prime(contexts)                   # (B, D)
        u_neg = self.W_prime(negatives)                  # (B, K, D)

        score_pos = (v_c * u_o).sum(dim=1)               # (B,)
        score_neg = torch.bmm(
            u_neg, v_c.unsqueeze(2)
        ).squeeze(2)                                      # (B, K)

        loss = -torch.log(torch.sigmoid(score_pos) + 1e-10).mean()
        loss -= torch.log(torch.sigmoid(-score_neg) + 1e-10).sum(dim=1).mean()
        return loss


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_model():
    """5-word vocab, 4-dim embeddings — small enough for numerical check."""
    return SGNSModel(vocab_size=5, embed_dim=4, seed=0)


@pytest.fixture
def batch():
    centers   = np.array([0, 1], dtype=np.int32)
    contexts  = np.array([2, 3], dtype=np.int32)
    negatives = np.array([[4, 1], [0, 2]], dtype=np.int32)
    return centers, contexts, negatives


# ── Numerical gradient tests ─────────────────────────────────────────────

def test_gradient_W_numerical(tiny_model, batch):
    centers, contexts, negatives = batch
    _, analytical_grads, _ = tiny_model.gradients(centers, contexts, negatives)
    num_grad_W, _ = numerical_gradient(tiny_model, centers, contexts, negatives)
    dense_grad_W, _ = _dense_grads(tiny_model, analytical_grads)
    np.testing.assert_allclose(
        dense_grad_W, num_grad_W, rtol=1e-4, atol=1e-6,
        err_msg="W gradient mismatch (numerical)"
    )


def test_gradient_W_prime_numerical(tiny_model, batch):
    centers, contexts, negatives = batch
    _, analytical_grads, _ = tiny_model.gradients(centers, contexts, negatives)
    _, num_grad_Wp = numerical_gradient(tiny_model, centers, contexts, negatives)
    _, dense_grad_Wp = _dense_grads(tiny_model, analytical_grads)
    np.testing.assert_allclose(
        dense_grad_Wp, num_grad_Wp, rtol=1e-4, atol=1e-6,
        err_msg="W_prime gradient mismatch (numerical)"
    )


# ── PyTorch oracle tests ────────────────────────────────────────────────

def test_forward_matches_pytorch(tiny_model, batch):
    """NumPy forward loss must match PyTorch forward loss (same weights)."""
    centers, contexts, negatives = batch
    np_loss = tiny_model.forward(centers, contexts, negatives)

    torch_model = TorchSGNS(tiny_model.W.copy(), tiny_model.W_prime.copy())
    torch_loss = torch_model(
        torch.tensor(centers, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(negatives, dtype=torch.long),
    )
    np.testing.assert_allclose(
        np_loss, torch_loss.item(), rtol=1e-6, atol=1e-8,
        err_msg="Forward loss mismatch vs PyTorch"
    )


def test_gradients_match_pytorch(tiny_model, batch):
    """NumPy analytical gradients must match torch.autograd gradients."""
    centers, contexts, negatives = batch

    # NumPy analytical gradients
    _, analytical_grads, _ = tiny_model.gradients(centers, contexts, negatives)
    dense_W, dense_Wp = _dense_grads(tiny_model, analytical_grads)

    # PyTorch autograd gradients
    torch_model = TorchSGNS(tiny_model.W.copy(), tiny_model.W_prime.copy())
    loss = torch_model(
        torch.tensor(centers, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(negatives, dtype=torch.long),
    )
    loss.backward()

    torch_grad_W  = torch_model.W.weight.grad.numpy()
    torch_grad_Wp = torch_model.W_prime.weight.grad.numpy()

    np.testing.assert_allclose(
        dense_W, torch_grad_W, rtol=1e-5, atol=1e-7,
        err_msg="W gradient mismatch vs PyTorch autograd"
    )
    np.testing.assert_allclose(
        dense_Wp, torch_grad_Wp, rtol=1e-5, atol=1e-7,
        err_msg="W_prime gradient mismatch vs PyTorch autograd"
    )


def test_sgd_update_matches_pytorch(tiny_model, batch):
    """One SGD step in NumPy must produce same weights as one step in PyTorch."""
    centers, contexts, negatives = batch
    lr = 0.01

    # Save initial weights
    W_init  = tiny_model.W.copy()
    Wp_init = tiny_model.W_prime.copy()

    # NumPy: one gradient step
    _, grads, _ = tiny_model.gradients(centers, contexts, negatives)
    tiny_model.update(grads, lr)

    # PyTorch: one gradient step
    torch_model = TorchSGNS(W_init.copy(), Wp_init.copy())
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)
    optimizer.zero_grad()
    loss = torch_model(
        torch.tensor(centers, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(negatives, dtype=torch.long),
    )
    loss.backward()
    optimizer.step()

    np.testing.assert_allclose(
        tiny_model.W, torch_model.W.weight.detach().numpy(),
        rtol=1e-5, atol=1e-7,
        err_msg="W weights diverge after one SGD step vs PyTorch"
    )
    np.testing.assert_allclose(
        tiny_model.W_prime, torch_model.W_prime.weight.detach().numpy(),
        rtol=1e-5, atol=1e-7,
        err_msg="W_prime weights diverge after one SGD step vs PyTorch"
    )
