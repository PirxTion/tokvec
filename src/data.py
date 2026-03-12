from __future__ import annotations
from collections import Counter
import os
import zipfile
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
    return text.split()


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
