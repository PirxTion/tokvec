"""word2vec SGNS training entrypoint.

Usage:
    uv run python train.py [--epochs 5] [--dim 100] [--lr 0.025] \
                           [--batch 512] [--neg 5] [--window 5] \
                           [--min-count 5] [--subsample 1e-5] \
                           [--save embeddings.npy] \
                           [--log-dir logs] [--save-steps 50000] \
                           [--eval-steps 50000]
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from datetime import datetime
import numpy as np

from src.data import (
    download_text8, tokenize, build_vocab,
    subsample_ids, generate_pairs, make_batches,
)
from src.model import SGNSModel
from src.negative_sampling import NegativeSampler


ANALOGIES_URL = (
    "https://raw.githubusercontent.com/nicholas-leonard/word2vec/"
    "master/questions-words.txt"
)


class TeeLogger:
    """Write to both stdout and a log file."""

    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, "w")

    def write(self, msg: str) -> int:
        self.terminal.write(msg)
        self.log_file.write(msg)
        self.log_file.flush()
        return len(msg)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


def download_analogies(data_dir: str = "data") -> str:
    import requests
    path = os.path.join(data_dir, "questions-words.txt")
    if not os.path.exists(path):
        print("Downloading analogy dataset...")
        r = requests.get(ANALOGIES_URL)
        r.raise_for_status()
        with open(path, "w") as f:
            f.write(r.text)
    return path


def linear_lr(lr0: float, step: int, total_steps: int) -> float:
    return max(lr0 * (1.0 - step / total_steps), lr0 * 1e-4)


def run_evaluation(
    model: SGNSModel,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    label: str = "final",
    analogies_path: str | None = None,
) -> float | None:
    """Run nearest-neighbors and (optionally) analogy evaluation. Returns overall analogy accuracy."""
    from src.evaluate import nearest_neighbors, evaluate_analogies, load_google_analogies

    probe_words = ["king", "paris", "computer", "good", "man"]
    print(f"\n── Nearest Neighbors ({label}) ──────────────────")
    for word in probe_words:
        if word in word2idx:
            nn = nearest_neighbors(model.W, word2idx, idx2word, word, n=8)
            print(f"  {word:12s} → {', '.join(nn)}")

    if analogies_path is not None:
        categories = load_google_analogies(analogies_path)
        print(f"\n── Analogy Accuracy ({label}) ─────────────────")
        all_correct, all_total = 0, 0
        for cat, pairs in categories.items():
            correct_i, total_i = evaluate_analogies(model.W, word2idx, idx2word, pairs)
            acc = correct_i / total_i if total_i > 0 else 0.0
            print(f"  {cat:35s}  {acc*100:5.1f}%  ({total_i} pairs)")
            all_correct += correct_i
            all_total   += total_i
        overall = all_correct / all_total if all_total > 0 else 0.0
        print(f"\n  Overall: {overall*100:.1f}%  ({all_correct}/{all_total})")
        return overall
    return None


def train(args: argparse.Namespace, run_dir: str, metrics_path: str | None = None) -> SGNSModel:
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
    model._word2idx = word2idx
    model._idx2word = idx2word
    sampler = NegativeSampler(freq_array, seed=42)
    rng     = np.random.default_rng(42)

    # Download analogies once if eval_steps is set
    analogies_path = download_analogies() if args.eval_steps else None

    # ── Metrics CSV ──────────────────────────────────────────────────────────
    csv_file = None
    csv_writer = None
    if metrics_path:
        csv_file = open(metrics_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "epoch", "loss", "lr", "grad_norm", "elapsed", "analogy_acc"])

    # Estimate total steps for LR schedule
    approx_pairs_per_epoch = int(len(token_ids) * args.window * 2 * 0.75)
    total_steps = (approx_pairs_per_epoch // args.batch) * args.epochs

    step = 0
    t_start = time.time()
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
            loss, grads, stats = model.gradients(centers, contexts, negatives)
            model.update(grads, lr * len(centers))

            epoch_loss += loss
            n_batches  += 1
            step       += 1

            if step % args.log_steps == 0:
                elapsed = time.time() - t0
                grad_norm = np.sqrt(
                    sum(np.dot(g, g) for g in grads["W"].values()) +
                    sum(np.dot(g, g) for g in grads["W_prime"].values())
                )
                print(
                    f"  step {step:>7,} | loss {loss:.4f} "
                    f"| lr {lr:.5f} | gnorm {grad_norm:.3f} | {elapsed:.0f}s elapsed"
                    f"\n    pos_score {stats['score_pos_mean']:+.4f} "
                    f"neg_score {stats['score_neg_mean']:+.4f} "
                    f"| σ(pos) {stats['sig_pos_mean']:.4f} "
                    f"σ(neg) {stats['sig_neg_mean']:.4f}"
                )
                if csv_writer:
                    csv_writer.writerow([step, epoch, f"{loss:.6f}", f"{lr:.6f}",
                                         f"{grad_norm:.6f}", f"{time.time()-t_start:.1f}", ""])
                    csv_file.flush()

            # ── Periodic checkpoint ──────────────────────────────────
            if args.save_steps and step % args.save_steps == 0:
                ckpt_path = os.path.join(run_dir, f"checkpoint_step{step:07d}.npy")
                np.save(ckpt_path, model.W)
                print(f"  Checkpoint saved: {ckpt_path}")

            # ── Periodic evaluation ──────────────────────────────────
            if args.eval_steps and step % args.eval_steps == 0:
                acc = run_evaluation(model, word2idx, idx2word, f"step {step:,}", analogies_path)
                if csv_writer and acc is not None:
                    csv_writer.writerow([step, epoch, "", "", "", f"{time.time()-t_start:.1f}",
                                         f"{acc:.6f}"])
                    csv_file.flush()

        print(
            f"Epoch {epoch}/{args.epochs} done — "
            f"avg loss: {epoch_loss / max(n_batches, 1):.4f}"
        )

    if args.save:
        save_path = os.path.join(run_dir, args.save)
        np.save(save_path, model.W)
        print(f"Embeddings saved to {save_path}")

    if csv_file:
        csv_file.close()

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
    p.add_argument("--log-steps",  type=int,   default=10000, dest="log_steps",
                   help="Print training metrics every N steps")
    p.add_argument("--log-dir",    type=str,   default="logs", dest="log_dir")
    p.add_argument("--save-steps", type=int,   default=0,   dest="save_steps",
                   help="Save checkpoint every N steps (0 = off)")
    p.add_argument("--eval-steps", type=int,   default=0,   dest="eval_steps",
                   help="Run evaluation every N steps (0 = off)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Set up run directory ─────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "train.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    metrics_path = os.path.join(run_dir, "metrics.csv")

    print(f"Run started: {timestamp}")
    print(f"Run directory: {run_dir}")
    print(f"Args: {vars(args)}\n")

    try:
        model = train(args, run_dir=run_dir, metrics_path=metrics_path)

        # ── Final evaluation ─────────────────────────────────────────────
        analogies_path = download_analogies()
        run_evaluation(
            model, model._word2idx, model._idx2word,
            label="final",
            analogies_path=analogies_path,
        )
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\nAll outputs saved to {run_dir}")
