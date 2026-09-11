"""Instrumentation utilities for Federated Learning thesis experiments."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from flwr.app import ArrayRecord


def array_record_nbytes(arrays: ArrayRecord) -> int:
    """Return raw model payload size in bytes."""
    return sum(len(array.data) for array in arrays.values())


def create_run_directory(protocol: str, seed: int) -> Path:
    """Create a unique results directory for one experiment run."""

    default_root = (
        Path.home()
        / "federated-learning-thesis"
        / "results"
    )

    results_root = Path(
        os.environ.get("THESIS_RESULTS_DIR", default_root)
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_dir = results_root / f"{timestamp}_{protocol}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    """Append one row to a CSV file."""

    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def metric_history_to_dict(history: dict) -> dict:
    """Convert Flower MetricRecord history to JSON-compatible dictionaries."""

    output = {}

    for round_number, metric_record in history.items():
        output[str(round_number)] = {
            key: value
            for key, value in metric_record.items()
        }

    return output


def save_summary(path: Path, summary: dict) -> None:
    """Save experiment summary as JSON."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
