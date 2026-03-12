from __future__ import annotations
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class SGNSModel:
    """Skip-Gram with Negative Sampling model.

    Parameters
    ----------
    vocab_size : int
        V — number of words in the vocabulary.
    embed_dim : int
        D — embedding dimensionality.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    W : np.ndarray, shape (V, D)
        Input (center word) embeddings.
    W_prime : np.ndarray, shape (V, D)
        Output (context/negative) embeddings.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 100, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        scale = 0.5 / embed_dim
        self.W       = rng.uniform(-scale, scale, (vocab_size, embed_dim))
        self.W_prime = np.zeros((vocab_size, embed_dim))
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self._word2idx: dict | None = None
        self._idx2word: dict | None = None

    def forward(
        self,
        centers:   np.ndarray,  # (B,)
        contexts:  np.ndarray,  # (B,)
        negatives: np.ndarray,  # (B, K)
    ) -> float:
        """Compute mean SGNS loss over a mini-batch.

        Loss per sample:
            L = -log σ(v_c · u_o) - Σ_k log σ(-v_c · u_k)
        """
        v_c   = self.W[centers]               # (B, D)
        u_o   = self.W_prime[contexts]        # (B, D)
        u_neg = self.W_prime[negatives]       # (B, K, D)

        score_pos = (v_c * u_o).sum(axis=1)                    # (B,)
        score_neg = np.einsum("bd,bkd->bk", v_c, u_neg)       # (B, K)

        loss = -np.log(sigmoid(score_pos) + 1e-10).mean()
        loss -= np.log(sigmoid(-score_neg) + 1e-10).mean()
        return float(loss)

    def gradients(
        self,
        centers:   np.ndarray,  # (B,)
        contexts:  np.ndarray,  # (B,)
        negatives: np.ndarray,  # (B, K)
    ) -> tuple[float, dict]:
        """Compute loss and analytical gradients.

        Gradient derivation (per sample, then averaged over batch):
            ∂L/∂v_c   = (σ(s_pos) - 1) · u_o + Σ_k σ(s_neg_k) · u_k
            ∂L/∂u_o   = (σ(s_pos) - 1) · v_c
            ∂L/∂u_k   = σ(s_neg_k) · v_c

        Returns
        -------
        loss : float
        grads : dict with keys "W" and "W_prime", each mapping
                word_index -> gradient_vector (D,)
        """
        B, K = negatives.shape
        v_c   = self.W[centers]               # (B, D)
        u_o   = self.W_prime[contexts]        # (B, D)
        u_neg = self.W_prime[negatives]       # (B, K, D)

        score_pos = (v_c * u_o).sum(axis=1)                    # (B,)
        # (v_c[:, None, :] * u_neg).sum(axis=-1)
        score_neg = np.einsum("bd,bkd->bk", v_c, u_neg)       # (B, K)

        sig_pos = sigmoid(score_pos)   # (B,)
        sig_neg = sigmoid(score_neg)   # (B, K)

        sig_neg_inv = sigmoid(-score_neg)  # σ(-s) = 1 - σ(s), kept consistent with forward()
        loss = -np.log(sig_pos + 1e-10).mean() - np.log(sig_neg_inv + 1e-10).mean()

        # Gradient of loss w.r.t. input embeddings v_c:  (B, D)
        # Positive term normalised by B; negative term normalised by B*K
        d_v_c  = (sig_pos - 1)[:, None] * u_o / B                            # (B, D)
        d_v_c += (sig_neg[:, :, None] * u_neg).sum(axis=1) / (B * K)         # (B, D)

        # Gradient w.r.t. output embedding u_o:  (B, D)  [positive term, mean over B]
        d_u_o  = (sig_pos - 1)[:, None] * v_c / B                            # (B, D)

        # Gradient w.r.t. output embeddings u_k:  (B, K, D)  [negative term, mean over B*K]
        d_u_neg = sig_neg[:, :, None] * v_c[:, None, :] / (B * K)            # (B, K, D)

        # Accumulate into sparse dicts keyed by vocab index
        grads_W: dict[int, np.ndarray] = {}
        for b in range(B):
            idx = int(centers[b])
            grads_W[idx] = grads_W.get(idx, np.zeros(self.embed_dim)) + d_v_c[b]

        grads_W_prime: dict[int, np.ndarray] = {}
        for b in range(B):
            idx = int(contexts[b])
            grads_W_prime[idx] = grads_W_prime.get(idx, np.zeros(self.embed_dim)) + d_u_o[b]
            for k in range(K):
                nidx = int(negatives[b, k])
                grads_W_prime[nidx] = grads_W_prime.get(nidx, np.zeros(self.embed_dim)) + d_u_neg[b, k]

        return float(loss), {"W": grads_W, "W_prime": grads_W_prime}

    def update(
        self,
        grads: dict,
        lr: float,
    ) -> None:
        """Apply SGD update in-place using sparse gradient dicts."""
        for idx, g in grads["W"].items():
            self.W[idx] -= lr * g
        for idx, g in grads["W_prime"].items():
            self.W_prime[idx] -= lr * g
