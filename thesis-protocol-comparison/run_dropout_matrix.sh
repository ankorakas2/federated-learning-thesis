#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/federated-learning-thesis"
PROJECT="$ROOT/thesis-protocol-comparison"
EXP_ROOT="$ROOT/results/dropout-final-v1"

NUM_CLIENTS=100
NUM_ROUNDS=10

SEEDS=(42 123 2026)
DROPOUT_RATES=(0.05 0.10)

export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
export RAY_DEDUP_LOGS=0

cd "$PROJECT"

mkdir -p "$EXP_ROOT"

# Preserve the exact experiment source/configuration
if [[ ! -f "$EXP_ROOT/metadata_complete.txt" ]]; then

    python --version > "$EXP_ROOT/python-version.txt"

    python - <<'PY' > "$EXP_ROOT/software-versions.txt"
import flwr
import ray
import torch
import torchvision

print("Flower:", flwr.__version__)
print("Ray:", ray.__version__)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
PY

    git status --short \
        > "$EXP_ROOT/git-status.txt" 2>&1 || true

    git diff \
        > "$EXP_ROOT/source-uncommitted.diff" 2>&1 || true

    tar -czf "$EXP_ROOT/source-code.tar.gz" \
        thesis_compare \
        pyproject.toml \
        run_dropout_matrix.sh

    sha256sum \
        thesis_compare/*.py \
        pyproject.toml \
        run_dropout_matrix.sh \
        > "$EXP_ROOT/source-sha256.txt"

    touch "$EXP_ROOT/metadata_complete.txt"
fi


run_one () {

    dropout_rate="$1"
    rep="$2"
    seed="$3"
    protocol="$4"

    RUN_DIR="$EXP_ROOT/dropout-${dropout_rate}/rep-${rep}/${protocol}"

    mkdir -p "$RUN_DIR"

    if [[ -f "$RUN_DIR/.done" ]]; then
        echo
        echo "SKIPPING completed run:"
        echo " dropout=$dropout_rate | rep=$rep | seed=$seed | protocol=$protocol"
        return
    fi

    echo
    echo "============================================================"
    echo " RQ3 FINAL RUN"
    echo " clients=$NUM_CLIENTS"
    echo " dropout=$dropout_rate"
    echo " repetition=$rep"
    echo " seed=$seed"
    echo " protocol=$protocol"
    echo " rounds=$NUM_ROUNDS"
    echo "============================================================"

    pkill -f 'flower-superlink.*--control-api-address 127.0.0.1:39093' \
        2>/dev/null || true

    ray stop --force 2>/dev/null || true

    export THESIS_RESULTS_DIR="$RUN_DIR/results"
    export THESIS_COMM_LOG="$RUN_DIR/communication.jsonl"

    rm -f "$THESIS_COMM_LOG"

    flwr federation simulation-config \
        @none/default local \
        --num-supernodes "$NUM_CLIENTS"

    if [[ "$protocol" == "plain" ]]; then

        flwr run . \
          --run-config \
          "protocol=\"plain\" num-clients=$NUM_CLIENTS num-server-rounds=$NUM_ROUNDS seed=$seed dropout-rate=$dropout_rate" \
          --stream 2>&1 | tee "$RUN_DIR/run.log"

    else

        flwr run . \
          --run-config \
          "protocol=\"secagg\" num-clients=$NUM_CLIENTS num-server-rounds=$NUM_ROUNDS seed=$seed dropout-rate=$dropout_rate num-shares=5 reconstruction-threshold=3 max-weight=2000" \
          --stream 2>&1 | tee "$RUN_DIR/run.log"

    fi

    grep -q '\[SUMMARY\]' "$RUN_DIR/run.log"

    if [[ "$protocol" == "secagg" ]]; then

        completed_rounds=$(
            grep -c 'Secure aggregation completed\.'                 "$RUN_DIR/run.log" || true
        )

        halted_rounds=$(
            grep -c 'Secure aggregation halted\.'                 "$RUN_DIR/run.log" || true
        )

        {
            echo "completed_rounds=$completed_rounds"
            echo "halted_rounds=$halted_rounds"
        } > "$RUN_DIR/secagg-round-status.txt"

        echo "SecAgg+ completed rounds: $completed_rounds"
        echo "SecAgg+ halted rounds: $halted_rounds"

        if (( completed_rounds + halted_rounds != NUM_ROUNDS )); then
            echo "ERROR: SecAgg+ round accounting mismatch"
            exit 1
        fi
    fi

    touch "$RUN_DIR/.done"

    echo
    echo "COMPLETED:"
    echo " dropout=$dropout_rate | rep=$rep | seed=$seed | protocol=$protocol"
}


for dropout_rate in "${DROPOUT_RATES[@]}"
do
    for rep in 1 2 3
    do

        seed="${SEEDS[$((rep - 1))]}"

        # Counterbalanced protocol order
        if (( rep % 2 == 1 )); then
            run_one "$dropout_rate" "$rep" "$seed" plain
            run_one "$dropout_rate" "$rep" "$seed" secagg
        else
            run_one "$dropout_rate" "$rep" "$seed" secagg
            run_one "$dropout_rate" "$rep" "$seed" plain
        fi

    done
done

echo
echo "============================================================"
echo "ALL RQ3 DROPOUT EXPERIMENTS COMPLETED"
echo "============================================================"
