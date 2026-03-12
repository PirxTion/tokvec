import numpy as np
from src.data import build_vocab


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
