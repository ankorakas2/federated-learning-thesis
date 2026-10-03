#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/federated-learning-thesis"
PROJECT="$ROOT/thesis-protocol-comparison"
EXP_ROOT="$ROOT/results/system-scaling-final-v1"

mkdir -p "$EXP_ROOT"

export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
export RAY_DEDUP_LOGS=0

SEEDS=(42 123 2026)
NUM_ROUNDS=10

cd "$PROJECT"

# Reproducibility metadata
if [[ ! -f "$EXP_ROOT/metadata_complete.txt" ]]; then
    {
        echo "===== DATE ====="
        date -Is
        echo
        echo "===== GIT COMMIT ====="
        git rev-parse HEAD 2>/dev/null || echo "NO_GIT_COMMIT"
        echo
        echo "===== PYTHON ====="
        python --version
        echo
        echo "===== VERSIONS ====="
        python - <<'PY'
import flwr, ray, torch, torchvision
print("Flower:", flwr.__version__)
print("Ray:", ray.__version__)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
PY
        echo
        echo "===== SYSTEM ====="
        uname -a
        echo
        lscpu
        echo
        free -h
    } > "$EXP_ROOT/environment-metadata.txt"

    uv pip freeze | sort > "$EXP_ROOT/environment-freeze.txt"

    # Preserve exact source state used for the final experiments
    git status --short > "$EXP_ROOT/git-status.txt" 2>&1 || true
    git diff > "$EXP_ROOT/source-uncommitted.diff" 2>&1 || true

    tar -czf "$EXP_ROOT/source-code.tar.gz"         thesis_compare         pyproject.toml         run_system_matrix.sh         run_paired_pilot.sh         LICENSE

    sha256sum         thesis_compare/*.py         pyproject.toml         run_system_matrix.sh         > "$EXP_ROOT/source-sha256.txt"

    touch "$EXP_ROOT/metadata_complete.txt"
fi

run_one () {
    clients="$1"
    rep="$2"
    protocol="$3"
    seed="$4"

    RUN_ROOT="$EXP_ROOT/clients-${clients}/rep-${rep}/${protocol}"
    DONE_FILE="$RUN_ROOT/.done"

    mkdir -p "$RUN_ROOT"

    if [[ -f "$DONE_FILE" ]]; then
        echo "SKIP: clients=$clients rep=$rep protocol=$protocol already completed."
        return
    fi

    echo
    echo "============================================================"
    echo " clients=$clients | repetition=$rep | seed=$seed | protocol=$protocol "
    echo "============================================================"

    pkill -f 'flower-superlink.*--control-api-address 127.0.0.1:39093' \
        2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true

    export THESIS_RESULTS_DIR="$RUN_ROOT/results"
    export THESIS_COMM_LOG="$RUN_ROOT/communication.jsonl"

    rm -f "$THESIS_COMM_LOG"
    mkdir -p "$THESIS_RESULTS_DIR"

    flwr federation simulation-config \
        @none/default local \
        --num-supernodes "$clients"

    if [[ "$protocol" == "plain" ]]; then
        flwr run . \
          --run-config \
          "protocol=\"plain\" num-clients=$clients num-server-rounds=$NUM_ROUNDS seed=$seed" \
          --stream 2>&1 | tee "$RUN_ROOT/flower.log"
    else
        flwr run . \
          --run-config \
          "protocol=\"secagg\" num-clients=$clients num-server-rounds=$NUM_ROUNDS seed=$seed num-shares=5 reconstruction-threshold=3 max-weight=2000" \
          --stream 2>&1 | tee "$RUN_ROOT/flower.log"
    fi

    # Store path to the generated summary for easy analysis later
    SUMMARY=$(find "$THESIS_RESULTS_DIR" \
        -name "run_summary.json" \
        -type f | head -1 || true)

    if [[ -n "$SUMMARY" ]]; then
        echo "$SUMMARY" > "$RUN_ROOT/summary-path.txt"
    fi

    touch "$DONE_FILE"

    echo "COMPLETED: clients=$clients rep=$rep protocol=$protocol"

    pkill -f 'flower-superlink.*--control-api-address 127.0.0.1:39093' \
        2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
}

for clients in 20 50 100
do
    for rep in 1 2 3
    do
        seed="${SEEDS[$((rep - 1))]}"

        # Alternate protocol order between repetitions
        if (( rep % 2 == 1 )); then
            run_one "$clients" "$rep" plain "$seed"
            run_one "$clients" "$rep" secagg "$seed"
        else
            run_one "$clients" "$rep" secagg "$seed"
            run_one "$clients" "$rep" plain "$seed"
        fi
    done
done

echo
echo "============================================================"
echo " ALL SYSTEM-SCALING EXPERIMENTS COMPLETED "
echo "============================================================"
