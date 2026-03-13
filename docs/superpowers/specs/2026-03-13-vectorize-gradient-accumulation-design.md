# Vectorize Gradient Accumulation

Date: 2026-03-13
Status: approved

## Problem

The `gradients()` method in `src/model.py` uses nested Python loops (O(B*K) iterations) to accumulate sparse gradients into dicts. This is the main code-level performance weakness compared to reference implementations (MarkoKolarski, albsd) which use `np.add.at`.

## Design

### Approach

Replace sparse dict-based gradient accumulation with `np.add.at` on full (V, D) arrays. Switch the gradient return format from `dict[int, np.ndarray]` to dense `np.ndarray` of shape (V, D).

### Changes

#### 1. `src/model.py` — `gradients()` method

**Before (lines 134-146):** Python loops accumulating into `dict[int, np.ndarray]`

**After:** Three `np.add.at` calls on zero-initialized (V, D) arrays:
```python
grad_W = np.zeros_like(self.W)
grad_Wp = np.zeros_like(self.W_prime)
np.add.at(grad_W, centers, d_v_c)
np.add.at(grad_Wp, contexts, d_u_o)
np.add.at(grad_Wp, negatives.ravel(), d_u_neg.reshape(-1, self.embed_dim))
```

Return type: `grads` changes from `dict[str, dict[int, np.ndarray]]` to `dict[str, np.ndarray]` — i.e. `grads["W"]` is now an `np.ndarray (V, D)` instead of a `dict[int, np.ndarray]`.

#### 2. `src/model.py` — `update()` method

**Before:** Dict iteration with per-index updates. Docstring says "sparse gradient dicts".

**After:** Dense vectorized SGD. Update docstring to "dense gradient arrays":
```python
self.W -= lr * grads["W"]
self.W_prime -= lr * grads["W_prime"]
```

#### 3. `train.py` — grad_norm computation (line 164-166)

**Before:** `sum(np.dot(g, g) for g in grads["W"].values())`

**After:** `np.sum(grads["W"] ** 2) + np.sum(grads["W_prime"] ** 2)` under `np.sqrt`.

#### 4. `tests/test_gradients.py` — `_dense_grads` helper

Remove `_dense_grads()` entirely. Replace its 3 call sites with direct access to `grads["W"]` and `grads["W_prime"]`.

### Not changed (but must verify)

- `forward()` — unaffected
- `sigmoid()` — unaffected
- Gradient math (d_v_c, d_u_o, d_u_neg computation) — identical
- `tests/test_integration.py` — calls `gradients()` + `update()` in tandem; no code changes needed but must verify it still passes

### Trade-offs

- **Memory:** ~112 MB extra per step (two V×D float64 arrays: 2 × 70k × 100 × 8 bytes). Negligible relative to the permanent W and W_prime matrices which already consume the same amount.
- **Performance:** Eliminates O(B*K) Python iterations per step. The dense `self.W -= lr * grad_W` touches all V rows even though most are zero, but this single vectorized op is negligible.
- **Correctness:** `np.add.at` correctly handles duplicate indices, same as the dict `.get()` approach.

### Verification

All existing tests must pass:

- `test_gradient_W_numerical` — remove `_dense_grads`, compare `grads["W"]` directly
- `test_gradient_W_prime_numerical` — remove `_dense_grads`, compare `grads["W_prime"]` directly
- `test_gradients_match_pytorch` — same removal
- `test_sgd_update_matches_pytorch` — works transparently (no `_dense_grads` used)
- `test_forward_matches_pytorch` — unaffected
- `test_numpy_pytorch_training_convergence` (integration) — works transparently, must verify

Note: the dense grad_norm formula produces identical results to the old sparse version because `np.zeros_like` initialization guarantees untouched rows are zero.
