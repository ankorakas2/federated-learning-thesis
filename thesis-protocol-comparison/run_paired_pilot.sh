#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/federated-learning-thesis"
PROJECT="$ROOT/thesis-protocol-comparison"
LOGDIR="$ROOT/results/paired-system-pilot"

mkdir -p "$LOGDIR"

export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
export RAY_DEDUP_LOGS=0
export THESIS_RESULTS_DIR="$ROOT/results"

run_protocol () {
    protocol="$1"
    repetition="$2"

    pkill -f 'flower-superlink.*--control-api-address 127.0.0.1:39093' 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true

    flwr federation simulation-config @none/default local --num-supernodes 5

    flwr run . \
      --run-config "protocol=\"$protocol\" num-server-rounds=1" \
      --stream 2>&1 | tee \
      "$LOGDIR/${protocol}-rep-${repetition}.log"
}

cd "$PROJECT"

for rep in 1 2 3 4 5
do
    echo "========== REPETITION $rep =========="

    # Alternate order to reduce order/warm-cache bias
    if (( rep % 2 == 1 )); then
        run_protocol plain "$rep"
        run_protocol secagg "$rep"
    else
        run_protocol secagg "$rep"
        run_protocol plain "$rep"
    fi
done

echo "All paired repetitions completed."
