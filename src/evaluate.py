from __future__ import annotations
import numpy as np


def _unit_normalize(W: np.ndarray) -> np.ndarray:
    """Return L2-normalized copy of W."""
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return W / np.maximum(norms, 1e-10)


def nearest_neighbors(
    W: np.ndarray,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    query: str,
    n: int = 10,
) -> list[str]:
    """Return top-n nearest neighbors of query word by cosine similarity."""
    if query not in word2idx:
        return []
    W_norm = _unit_normalize(W)
    q_vec  = W_norm[word2idx[query]]
    sims   = W_norm @ q_vec                    # (V,)
    sims[word2idx[query]] = -np.inf            # exclude self
    top_ids = np.argpartition(sims, -n)[-n:]
    top_ids = top_ids[np.argsort(sims[top_ids])[::-1]]
    return [idx2word[i] for i in top_ids]


def evaluate_analogies(
    W: np.ndarray,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    analogies: list[tuple[str, str, str, str]],
) -> tuple[int, int]:
    """Evaluate analogies using 3CosAdd. Returns (correct, total) counts.

    For each (a, b, c, d): predict argmax cos(w, v_b - v_a + v_c).
    Excludes a, b, c from candidates.
    """
    W_norm = _unit_normalize(W)
    correct = 0
    total   = 0
    for a, b, c, d in analogies:
        if any(w not in word2idx for w in [a, b, c, d]):
            continue
        query_vec = W_norm[word2idx[b]] - W_norm[word2idx[a]] + W_norm[word2idx[c]]
        sims = W_norm @ query_vec
        for w in [a, b, c]:
            sims[word2idx[w]] = -np.inf
        pred_idx = int(np.argmax(sims))
        if pred_idx == word2idx[d]:
            correct += 1
        total += 1
    return correct, total


def load_google_analogies(path: str) -> dict[str, list[tuple[str, str, str, str]]]:
    """Load Google analogy dataset. Returns dict of category -> list of tuples."""
    categories: dict[str, list] = {}
    current = "misc"
    with open(path) as f:
        for line in f:
            line = line.strip().lower()
            if line.startswith(":"):
                current = line[2:]
            else:
                parts = line.split()
                if len(parts) == 4:
                    categories.setdefault(current, []).append(tuple(parts))
    return categories
