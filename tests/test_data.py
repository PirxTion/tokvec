import os
import numpy as np
import pytest
from src.data import build_vocab, download_text8, tokenize, subsample_ids


def test_build_vocab_basic():
    tokens = ["a", "b", "a", "c", "a", "b"]
    word2idx, idx2word, freq_array = build_vocab(tokens, min_count=2)
    # "a" (3), "b" (2) pass; "c" (1) filtered out
    assert "a" in word2idx
    assert "b" in word2idx
    assert "c" not in word2idx
    assert len(word2idx) == 2
    assert idx2word[word2idx["a"]] == "a"
    assert abs(freq_array.sum() - 1.0) < 1e-6  # normalized frequencies
    # "a" (count 3) should have higher frequency than "b" (count 2)
    assert freq_array[word2idx["a"]] > freq_array[word2idx["b"]]


def test_build_vocab_min_count():
    tokens = ["x"] * 10 + ["y"] * 3 + ["z"] * 1
    word2idx, _, freq_array = build_vocab(tokens, min_count=3)
    assert "x" in word2idx
    assert "y" in word2idx
    assert "z" not in word2idx
    assert freq_array.shape == (2,)


def test_tokenize():
    tokens = tokenize("hello world hello")
    assert tokens == ["hello", "world", "hello"]


def test_subsample_removes_frequent_words():
    # freq_array: word 0 is very frequent (0.99), word 1 is rare (0.01)
    freq_array = np.array([0.99, 0.01])
    token_ids = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 100)
    rng = np.random.default_rng(42)
    kept = subsample_ids(token_ids, freq_array, t=1e-5, rng=rng)
    freq_0 = (kept == 0).sum() / len(kept)
    freq_1 = (kept == 1).sum() / len(kept)
    # frequent word (0) should be heavily discarded; rare word (1) mostly kept
    assert freq_0 < freq_1


def test_download_text8_creates_file(tmp_path):
    if os.environ.get("SKIP_DOWNLOAD"):
        pytest.skip("SKIP_DOWNLOAD is set")
    path = download_text8(data_dir=str(tmp_path))
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read(100)
    assert len(content) > 0
