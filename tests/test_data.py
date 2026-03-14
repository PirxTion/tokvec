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

from src.data import generate_pairs, make_batches


def test_generate_pairs_center_context():
    # Use a sequence where we can verify positional window distance.
    # token_ids positions: 0→10, 1→20, 2→30, 3→40, 4→50
    # With max_window=1, center at position i pairs only with positions i±1.
    token_ids = np.array([10, 20, 30, 40, 50])
    rng = np.random.default_rng(0)
    pairs = list(generate_pairs(token_ids, max_window=1, rng=rng))
    # Build a set of (center_val, context_val) to check positional distance
    # Each pair must come from adjacent positions (distance 1 in the sequence)
    adjacency = {
        (token_ids[i], token_ids[j])
        for i in range(len(token_ids))
        for j in range(len(token_ids))
        if abs(i - j) <= 1 and i != j
    }
    for center, context in pairs:
        assert (center, context) in adjacency, (
            f"Pair ({center}, {context}) exceeds window=1"
        )
    # Center tokens from non-edge positions must appear
    centers = {p[0] for p in pairs}
    assert 20 in centers  # position 1
    assert 30 in centers  # position 2
    assert 40 in centers


def test_make_batches_shapes():
    pairs = [(i, i + 1) for i in range(100)]
    batches = list(make_batches(iter(pairs), batch_size=32))
    assert len(batches) > 0
    centers, contexts = batches[0]
    assert centers.shape == contexts.shape
    assert centers.shape[0] <= 32
