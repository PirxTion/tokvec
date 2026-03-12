from __future__ import annotations
from collections import Counter
import numpy as np


def build_vocab(
    tokens: list[str], min_count: int = 5
) -> tuple[dict[str, int], dict[int, str], np.ndarray]:
    """Build vocabulary from token list.

    Returns:
        word2idx: word -> index mapping
        idx2word: index -> word mapping
        freq_array: normalized frequency for each vocab word (shape: V)
    """
    counts = Counter(tokens)
    vocab = {w: c for w, c in counts.items() if c >= min_count}
    # sort by frequency descending for stable ordering
    sorted_vocab = sorted(vocab.items(), key=lambda x: -x[1])
    word2idx = {w: i for i, (w, _) in enumerate(sorted_vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    freqs = np.array([c for _, c in sorted_vocab], dtype=np.float64)
    freq_array = freqs / freqs.sum()
    return word2idx, idx2word, freq_array
