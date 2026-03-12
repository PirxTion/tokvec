import numpy as np
from src.negative_sampling import NegativeSampler


def test_sample_shape():
    freq_array = np.array([0.5, 0.3, 0.2])
    sampler = NegativeSampler(freq_array, table_size=10_000, seed=0)
    negatives = sampler.sample(batch_size=8, k=5)
    assert negatives.shape == (8, 5)


def test_sample_distribution_follows_unigram_power():
    # word 0 has freq 0.9, word 1 has freq 0.1
    # after ^(3/4): 0.9^0.75 ≈ 0.924, 0.1^0.75 ≈ 0.178
    # so word 0 should appear ~5x more than word 1
    freq_array = np.array([0.9, 0.1])
    sampler = NegativeSampler(freq_array, table_size=1_000_000, seed=42)
    samples = sampler.sample(batch_size=10_000, k=1).flatten()
    ratio = (samples == 0).sum() / (samples == 1).sum()
    # should be roughly 0.924/0.178 ≈ 5.2
    assert 3.0 < ratio < 8.0


def test_sample_valid_indices():
    freq_array = np.array([0.2, 0.5, 0.3])
    sampler = NegativeSampler(freq_array, table_size=10_000, seed=1)
    negatives = sampler.sample(batch_size=100, k=5)
    assert negatives.min() >= 0
    assert negatives.max() < 3
