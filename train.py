"""word2vec SGNS training entrypoint.

Usage:
    uv run python train.py [--epochs 5] [--dim 100] [--lr 0.025] \
                           [--batch 512] [--neg 5] [--window 5] \
                           [--min-count 5] [--subsample 1e-5] \
                           [--save embeddings.npy]
"""
from __future__ import annotations
import argparse
import time
import numpy as np

from src.data import (
    download_text8, tokenize, build_vocab,
    subsample_ids, generate_pairs, make_batches,
)
from src.model import SGNSModel
from src.negative_sampling import NegativeSampler


def linear_lr(lr0: float, step: int, total_steps: int) -> float:
    return max(lr0 * (1.0 - step / total_steps), lr0 * 1e-4)


def train(args: argparse.Namespace) -> SGNSModel:
    # ── Data ────────────────────────────────────────────────────────────────
    text8_path = download_text8()
    with open(text8_path) as f:
        raw = f.read()
    tokens = tokenize(raw)
    word2idx, idx2word, freq_array = build_vocab(tokens, min_count=args.min_count)
    V = len(word2idx)
    print(f"Vocab size: {V:,}")

    token_ids = np.array(
        [word2idx[t] for t in tokens if t in word2idx], dtype=np.int32
    )

    # ── Model + sampler ─────────────────────────────────────────────────────
    model   = SGNSModel(vocab_size=V, embed_dim=args.dim, seed=42)
    # Attach vocab to model so __main__ block can access them after train() returns
    model._word2idx = word2idx
    model._idx2word = idx2word
    sampler = NegativeSampler(freq_array, seed=42)
    rng     = np.random.default_rng(42)

    # Estimate total steps for LR schedule
    approx_pairs_per_epoch = int(len(token_ids) * args.window * 2 * 0.75)
    total_steps = (approx_pairs_per_epoch // args.batch) * args.epochs

    step = 0
    for epoch in range(1, args.epochs + 1):
        subsampled = subsample_ids(token_ids, freq_array, t=args.subsample, rng=rng)
        pairs_iter  = generate_pairs(subsampled, max_window=args.window, rng=rng)
        batches     = make_batches(pairs_iter, batch_size=args.batch)

        epoch_loss = 0.0
        n_batches  = 0
        t0 = time.time()

        for centers, contexts in batches:
            negatives = sampler.sample(len(centers), k=args.neg)
            lr = linear_lr(args.lr, step, total_steps)
            loss, grads = model.gradients(centers, contexts, negatives)
            model.update(grads, lr)

            epoch_loss += loss
            n_batches  += 1
            step       += 1

            if step % 10_000 == 0:
                elapsed = time.time() - t0
                print(
                    f"  step {step:>7,} | loss {loss:.4f} "
                    f"| lr {lr:.5f} | {elapsed:.0f}s elapsed"
                )

        print(
            f"Epoch {epoch}/{args.epochs} done — "
            f"avg loss: {epoch_loss / max(n_batches, 1):.4f}"
        )

    if args.save:
        np.save(args.save, model.W)
        print(f"Embeddings saved to {args.save}")

    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--dim",        type=int,   default=100)
    p.add_argument("--lr",         type=float, default=0.025)
    p.add_argument("--batch",      type=int,   default=512)
    p.add_argument("--neg",        type=int,   default=5)
    p.add_argument("--window",     type=int,   default=5)
    p.add_argument("--min-count",  type=int,   default=5,   dest="min_count")
    p.add_argument("--subsample",  type=float, default=1e-5)
    p.add_argument("--save",       type=str,   default="embeddings.npy")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = train(args)

    # Quick nearest-neighbor preview after training
    from src.evaluate import nearest_neighbors
    for word in ["king", "paris", "computer", "good"]:
        if word in model._word2idx:
            nn = nearest_neighbors(model.W, model._word2idx, model._idx2word, word, n=5)
            print(f"\nNearest to '{word}': {nn}")
