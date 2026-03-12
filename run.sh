#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ── Colors ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step() { printf "\n${GREEN}▶ %s${NC}\n" "$1"; }
warn() { printf "${YELLOW}⚠ %s${NC}\n" "$1"; }
fail() { printf "${RED}✗ %s${NC}\n" "$1"; exit 1; }

# ── 1. Check for uv ─────────────────────────────────────────────────────────
step "Checking for uv..."
if ! command -v uv &>/dev/null; then
    warn "uv not found — installing via official installer"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv &>/dev/null || fail "uv installation failed"
fi
echo "uv $(uv --version)"

# ── 2. Install / sync dependencies ──────────────────────────────────────────
step "Syncing dependencies (creates .venv if needed)..."
uv sync --dev

# ── 3. Run tests ────────────────────────────────────────────────────────────
step "Running tests..."
if uv run pytest tests/ -v; then
    echo -e "${GREEN}All tests passed.${NC}"
else
    fail "Tests failed — aborting."
fi

# ── 4. Train ────────────────────────────────────────────────────────────────
step "Training word2vec (skip-gram + negative sampling)..."

EPOCHS="${EPOCHS:-5}"
DIM="${DIM:-100}"
LR="${LR:-0.025}"
BATCH="${BATCH:-512}"
NEG="${NEG:-5}"
WINDOW="${WINDOW:-5}"
MIN_COUNT="${MIN_COUNT:-5}"
EVAL_STEPS="${EVAL_STEPS:-50000}"
SAVE_STEPS="${SAVE_STEPS:-50000}"
LOG_STEPS="${LOG_STEPS:-10000}"

echo "  epochs=$EPOCHS  dim=$DIM  lr=$LR  batch=$BATCH  neg=$NEG  window=$WINDOW"

uv run python train.py \
    --epochs "$EPOCHS" \
    --dim "$DIM" \
    --lr "$LR" \
    --batch "$BATCH" \
    --neg "$NEG" \
    --window "$WINDOW" \
    --min-count "$MIN_COUNT" \
    --eval-steps "$EVAL_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --log-steps "$LOG_STEPS" \
    --save embeddings.npy \
    --log-dir logs

# ── 5. Visualize ────────────────────────────────────────────────────────────
step "Generating training plots..."

# Find the most recent run directory
RUN_DIR="$(ls -dt logs/*/  2>/dev/null | head -1)"
if [ -z "$RUN_DIR" ]; then
    warn "No run directory found in logs/ — skipping visualization."
else
    METRICS_CSV="${RUN_DIR}metrics.csv"
    if [ -f "$METRICS_CSV" ]; then
        echo "  Using $METRICS_CSV"
        uv run python visualize.py "$METRICS_CSV" --save "${RUN_DIR}training_plot.png"
        echo -e "${GREEN}Plot saved to ${RUN_DIR}training_plot.png${NC}"
    else
        warn "No metrics.csv found in $RUN_DIR — skipping visualization."
    fi
fi

# ── Done ────────────────────────────────────────────────────────────────────
step "All done!"
RUN_DIR="${RUN_DIR:-logs/}"
echo "  Run output: ${RUN_DIR}"
