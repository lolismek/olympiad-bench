#!/usr/bin/env bash
# Run a model against IPhO 2018 E2 with live log flushing + a tail-able trace.
#
# Usage:
#   eval/run_model.sh <model> [seeds] [randomize] [randomize_strength] [seed_offset]
#   defaults: seeds=1, randomize=false, randomize_strength=1.0, seed_offset=0
#
# Examples:
#   eval/run_model.sh anthropic/claude-sonnet-4-6
#   eval/run_model.sh anthropic/claude-opus-4-7 3
#   eval/run_model.sh anthropic/claude-opus-4-7 5 true        # randomized truth
#   eval/run_model.sh anthropic/claude-opus-4-7 5 true 0.5    # half-strength
#   eval/run_model.sh anthropic/claude-sonnet-4-6 1 true 1.0 3  # only seed-3
#
# Live inspection while running:
#   tail -f logs/trace.jsonl                  # one JSON line per tool call
#   inspect view --log-dir logs/              # rich web UI (refreshes ~10s)

set -euo pipefail

MODEL="${1:?usage: run_model.sh <model> [seeds] [randomize] [randomize_strength] [seed_offset]}"
SEEDS="${2:-1}"
RANDOMIZE="${3:-false}"
STRENGTH="${4:-1.0}"
SEED_OFFSET="${5:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs
# Rotate the trace file so runs don't mix.
STAMP="$(date +%Y%m%dT%H%M%S)"
TRACE_FILE="logs/trace-${STAMP}.jsonl"
ln -sf "$(basename "$TRACE_FILE")" logs/trace.jsonl

export IPHO_TRACE_FILE="${PWD}/${TRACE_FILE}"

exec inspect eval eval/tasks/ipho_2018_E2.py@ipho_2018_E2 \
    --model "$MODEL" \
    -T "seeds=${SEEDS}" \
    -T "randomize=${RANDOMIZE}" \
    -T "randomize_strength=${STRENGTH}" \
    -T "seed_offset=${SEED_OFFSET}" \
    --log-dir logs/ \
    --log-shared 5
