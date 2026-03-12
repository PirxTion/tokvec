from __future__ import annotations
import numpy as np


class NegativeSampler:
    """Fast O(1) negative sampler using a pre-built unigram^(3/4) noise table."""

    def __init__(
        self,
        freq_array: np.ndarray,
        table_size: int = int(1e8),
        seed: int | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        # Build noise table proportional to freq^(3/4)
        powered = freq_array ** 0.75
        powered /= powered.sum()
        table_counts = (powered * table_size).astype(np.int64)
        # Adjust for rounding so table has exactly table_size entries
        diff = table_size - table_counts.sum()
        table_counts[np.argmax(powered)] += diff
        self._table = np.repeat(np.arange(len(freq_array), dtype=np.int32), table_counts)

    def sample(self, batch_size: int, k: int = 5) -> np.ndarray:
        """Return negative sample indices of shape (batch_size, k)."""
        indices = self.rng.integers(0, len(self._table), size=batch_size * k)
        return self._table[indices].reshape(batch_size, k)
