"""Benchmark: dict-based vs np.add.at gradient accumulation.

Run with:  uv run pytest -m benchmark tests/test_benchmark_gradients.py -v -s
"""

import timeit
from itertools import product

import numpy as np
import pytest

pytestmark = pytest.mark.benchmark

# ---------------------------------------------------------------------------
# Two standalone accumulation implementations
# ---------------------------------------------------------------------------

def accumulate_dense(centers, contexts, negatives, d_v_c, d_u_o, d_u_neg, V, D):
    """Current approach: np.add.at on dense (V, D) arrays."""
    grad_W = np.zeros((V, D))
    grad_Wp = np.zeros((V, D))
    np.add.at(grad_W, centers, d_v_c)
    np.add.at(grad_Wp, contexts, d_u_o)
    np.add.at(grad_Wp, negatives.ravel(), d_u_neg.reshape(-1, D))
    return grad_W, grad_Wp


def accumulate_dict(centers, contexts, negatives, d_v_c, d_u_o, d_u_neg, V, D):
    """Old approach: Python loop into sparse dicts."""
    B, K = negatives.shape
    grads_W = {}
    for b in range(B):
        idx = int(centers[b])
        grads_W[idx] = grads_W.get(idx, np.zeros(D)) + d_v_c[b]
    grads_W_prime = {}
    for b in range(B):
        idx = int(contexts[b])
        grads_W_prime[idx] = grads_W_prime.get(idx, np.zeros(D)) + d_u_o[b]
        for k in range(K):
            nidx = int(negatives[b, k])
            grads_W_prime[nidx] = grads_W_prime.get(nidx, np.zeros(D)) + d_u_neg[b, k]
    return grads_W, grads_W_prime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(V, B, K, D, rng):
    """Generate random indices and gradient arrays."""
    centers = rng.integers(0, V, size=B)
    contexts = rng.integers(0, V, size=B)
    negatives = rng.integers(0, V, size=(B, K))
    d_v_c = rng.standard_normal((B, D))
    d_u_o = rng.standard_normal((B, D))
    d_u_neg = rng.standard_normal((B, K, D))
    return centers, contexts, negatives, d_v_c, d_u_o, d_u_neg


def _dict_to_dense(grads_dict, V, D):
    """Convert a sparse dict of gradients to a dense (V, D) array."""
    arr = np.zeros((V, D))
    for idx, grad in grads_dict.items():
        arr[idx] += grad
    return arr


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------

def test_accumulation_equivalence():
    """Both approaches must produce identical gradient arrays."""
    rng = np.random.default_rng(42)
    V, B, K, D = 1_000, 64, 5, 100
    data = _make_data(V, B, K, D, rng)

    grad_W_dense, grad_Wp_dense = accumulate_dense(*data, V, D)
    grads_W_dict, grads_Wp_dict = accumulate_dict(*data, V, D)

    grad_W_from_dict = _dict_to_dense(grads_W_dict, V, D)
    grad_Wp_from_dict = _dict_to_dense(grads_Wp_dict, V, D)

    np.testing.assert_allclose(grad_W_dense, grad_W_from_dict, atol=1e-12)
    np.testing.assert_allclose(grad_Wp_dense, grad_Wp_from_dict, atol=1e-12)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

_V_VALUES = [10_000, 50_000]
_B_VALUES = [64, 512]
_K_VALUES = [5, 15]
_D = 100
_PARAMS = list(product(_V_VALUES, _B_VALUES, _K_VALUES))


@pytest.mark.slow
@pytest.mark.parametrize("V,B,K", _PARAMS, ids=[f"V{v}_B{b}_K{k}" for v, b, k in _PARAMS])
def test_benchmark(V, B, K):
    """Time both accumulation strategies and print comparison."""
    rng = np.random.default_rng(0)
    data = _make_data(V, B, K, _D, rng)

    # Adaptive iteration count: more iterations for smaller workloads
    base_ops = B * (1 + K)
    if base_ops < 5_000:
        number, repeat = 200, 5
    elif base_ops < 50_000:
        number, repeat = 50, 5
    else:
        number, repeat = 20, 5

    t_dense = timeit.repeat(
        lambda: accumulate_dense(*data, V, _D), number=number, repeat=repeat
    )
    t_dict = timeit.repeat(
        lambda: accumulate_dict(*data, V, _D), number=number, repeat=repeat
    )

    med_dense = sorted(t_dense)[len(t_dense) // 2] / number * 1e3  # ms
    med_dict = sorted(t_dict)[len(t_dict) // 2] / number * 1e3

    speedup = med_dict / med_dense if med_dense > 0 else float("inf")
    winner = "dense" if speedup > 1 else "dict"

    print(
        f"\n  V={V:>6}, B={B:>4}, K={K:>2}  |  "
        f"dense: {med_dense:7.3f} ms  dict: {med_dict:7.3f} ms  |  "
        f"ratio(dict/dense): {speedup:.2f}x  winner: {winner}"
    )
