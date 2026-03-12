from __future__ import annotations
from collections import Counter
import os
import zipfile
from typing import Iterator
import requests
import numpy as np


TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"


def download_text8(data_dir: str = "data") -> str:
    """Download and extract text8. Returns path to text8 file."""
    os.makedirs(data_dir, exist_ok=True)
    text8_path = os.path.join(data_dir, "text8")
    if os.path.exists(text8_path):
        return text8_path
    zip_path = os.path.join(data_dir, "text8.zip")
    if not os.path.exists(zip_path):
        print("Downloading text8...")
        response = requests.get(TEXT8_URL, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Extracting text8...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("text8", data_dir)
    return text8_path


def tokenize(text: str) -> list[str]:
    """Split text on whitespace."""
    return text.lower().split()


def subsample_ids(
    token_ids: np.ndarray,
    freq_array: np.ndarray,
    t: float = 1e-5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Subsample frequent words per Mikolov et al. §2.3.

    P(keep word w) = sqrt(t / f(w))  clipped to [0, 1]

    Note: The paper's full formula is sqrt(t/f) + t/f. We use the simplified
    sqrt(t/f) approximation, which dominates at typical frequencies and is
    the form most implementations use. Be ready to discuss this in review.
    """
    if rng is None:
        rng = np.random.default_rng()
    f = freq_array[token_ids]
    keep_prob = np.sqrt(t / f)
    keep_prob = np.clip(keep_prob, 0.0, 1.0)
    mask = rng.random(len(token_ids)) < keep_prob
    return token_ids[mask]


def generate_pairs(
    token_ids: np.ndarray,
    max_window: int = 5,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield (center_idx, context_idx) pairs using dynamic window size."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(token_ids)
    for i in range(n):
        window = int(rng.integers(1, max_window + 1))
        start = max(0, i - window)
        end = min(n, i + window + 1)
        for j in range(start, end):
            if j != i:
                yield (token_ids[i], token_ids[j])


def make_batches(
    pairs: Iterator[tuple[int, int]],
    batch_size: int = 512,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Collect pairs into mini-batches."""
    centers, contexts = [], []
    for center, context in pairs:
        centers.append(center)
        contexts.append(context)
        if len(centers) == batch_size:
            yield np.array(centers, dtype=np.int32), np.array(contexts, dtype=np.int32)
            centers, contexts = [], []
    if centers:
        yield np.array(centers, dtype=np.int32), np.array(contexts, dtype=np.int32)


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
