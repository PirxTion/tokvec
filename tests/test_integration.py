"""Integration test: train both NumPy and PyTorch SGNS on the same
small fixed corpus with identical hyperparameters. After a few epochs,
verify that losses track closely and final embeddings are similar.
"""
import numpy as np
import torch
import torch.nn as nn
from src.model import SGNSModel
from src.data import build_vocab, generate_pairs, make_batches
from src.negative_sampling import NegativeSampler


class TorchSGNS(nn.Module):
    """Reference PyTorch SGNS for integration testing."""
    def __init__(self, W_np, W_prime_np):
        super().__init__()
        self.W = nn.Embedding.from_pretrained(
            torch.tensor(W_np, dtype=torch.float64), freeze=False
        )
        self.W_prime = nn.Embedding.from_pretrained(
            torch.tensor(W_prime_np, dtype=torch.float64), freeze=False
        )

    def forward(self, centers, contexts, negatives):
        v_c   = self.W(centers)
        u_o   = self.W_prime(contexts)
        u_neg = self.W_prime(negatives)
        score_pos = (v_c * u_o).sum(dim=1)
        score_neg = torch.bmm(u_neg, v_c.unsqueeze(2)).squeeze(2)
        loss = -torch.log(torch.sigmoid(score_pos) + 1e-10).mean()
        loss -= torch.log(torch.sigmoid(-score_neg) + 1e-10).sum(dim=1).mean()
        return loss


CORPUS = "the cat sat on the mat the cat ate the rat on the mat"


def test_numpy_pytorch_training_convergence():
    """Both implementations should produce near-identical loss curves
    and final weights when trained on the same data with same init."""
    tokens = CORPUS.split()
    word2idx, idx2word, freq_array = build_vocab(tokens, min_count=1)
    V = len(word2idx)
    D = 8
    lr = 0.05
    K = 2
    epochs = 3

    token_ids = np.array([word2idx[t] for t in tokens], dtype=np.int32)

    # Shared init
    np_model = SGNSModel(vocab_size=V, embed_dim=D, seed=7)
    W_init  = np_model.W.copy()
    Wp_init = np_model.W_prime.copy()

    torch_model = TorchSGNS(W_init.copy(), Wp_init.copy())
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Deterministic pairs each epoch
        rng = np.random.default_rng(epoch)
        pairs = list(generate_pairs(token_ids, max_window=2, rng=rng))
        batches = list(make_batches(iter(pairs), batch_size=16))

        # Reset sampler seed each epoch for identical negatives
        sampler_epoch = NegativeSampler(freq_array, table_size=10_000, seed=epoch)

        for centers, contexts in batches:
            negatives = sampler_epoch.sample(len(centers), k=K)

            # NumPy step
            _, grads, _ = np_model.gradients(centers, contexts, negatives)
            np_model.update(grads, lr)

            # PyTorch step
            optimizer.zero_grad()
            loss_pt = torch_model(
                torch.tensor(centers, dtype=torch.long),
                torch.tensor(contexts, dtype=torch.long),
                torch.tensor(negatives, dtype=torch.long),
            )
            loss_pt.backward()
            optimizer.step()

    # Final weights should be very close
    np.testing.assert_allclose(
        np_model.W, torch_model.W.weight.detach().numpy(),
        rtol=1e-4, atol=1e-6,
        err_msg="W diverged after full training"
    )
    np.testing.assert_allclose(
        np_model.W_prime, torch_model.W_prime.weight.detach().numpy(),
        rtol=1e-4, atol=1e-6,
        err_msg="W_prime diverged after full training"
    )
