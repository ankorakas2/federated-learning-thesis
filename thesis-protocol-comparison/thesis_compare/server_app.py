"""Instrumented Plain FedAvg versus SecAgg+ comparison ServerApp."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD, Key
from flwr.server.workflow.default_workflows import (
    default_evaluate_workflow,
    default_fit_workflow,
)

from thesis_compare.task import get_weights, make_net


def aggregate_fit_metrics(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)

    return {
        "train_loss": sum(
            num_examples * values["train_loss"]
            for num_examples, values in metrics
        )
        / total_examples,
        "local_train_seconds": sum(
            num_examples * values["local_train_seconds"]
            for num_examples, values in metrics
        )
        / total_examples,
    }


def aggregate_evaluate_metrics(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)

    return {
        "accuracy": sum(
            num_examples * values["accuracy"]
            for num_examples, values in metrics
        )
        / total_examples
    }


def create_results_dir(protocol, seed):
    root = Path(
        os.environ.get(
            "THESIS_RESULTS_DIR",
            Path.home() / "federated-learning-thesis" / "results",
        )
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    path = root / f"{timestamp}_comparison_{protocol}_seed{seed}"
    path.mkdir(parents=True, exist_ok=True)

    return path


def history_values_by_round(entries):
    return {int(round_number): value for round_number, value in entries}


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    protocol = str(context.run_config["protocol"])

    if protocol not in {"plain", "secagg"}:
        raise ValueError("protocol must be 'plain' or 'secagg'")

    seed = int(context.run_config["seed"])
    num_clients = int(context.run_config["num-clients"])
    num_rounds = int(context.run_config["num-server-rounds"])

    parameters = ndarrays_to_parameters(
        get_weights(make_net(seed))
    )

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        initial_parameters=parameters,
    )

    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    results_dir = create_results_dir(protocol, seed)

    round_timings = {}

    if protocol == "plain":
        base_fit_workflow = default_fit_workflow
    else:
        base_fit_workflow = SecAggPlusWorkflow(
            num_shares=int(context.run_config["num-shares"]),
            reconstruction_threshold=int(
                context.run_config["reconstruction-threshold"]
            ),
            max_weight=float(context.run_config["max-weight"]),
        )

    def timed_fit_workflow(grid, legacy_context):
        current_round = int(
            legacy_context.state.config_records[
                MAIN_CONFIGS_RECORD
            ][Key.CURRENT_ROUND]
        )

        start = perf_counter()
        base_fit_workflow(grid, legacy_context)
        elapsed = perf_counter() - start

        round_timings.setdefault(current_round, {})
        round_timings[current_round]["fit_seconds"] = elapsed

        print(
            f"[ROUND_TIMING] round={current_round} "
            f"fit_seconds={elapsed:.4f}",
            flush=True,
        )

    def timed_evaluate_workflow(grid, legacy_context):
        current_round = int(
            legacy_context.state.config_records[
                MAIN_CONFIGS_RECORD
            ][Key.CURRENT_ROUND]
        )

        start = perf_counter()
        default_evaluate_workflow(grid, legacy_context)
        elapsed = perf_counter() - start

        round_timings.setdefault(current_round, {})
        round_timings[current_round]["evaluate_seconds"] = elapsed

        print(
            f"[ROUND_TIMING] round={current_round} "
            f"evaluate_seconds={elapsed:.4f}",
            flush=True,
        )

    workflow = DefaultWorkflow(
        fit_workflow=timed_fit_workflow,
        evaluate_workflow=timed_evaluate_workflow,
    )

    print(f"[HARNESS] protocol={protocol}", flush=True)
    print(f"[HARNESS] clients={num_clients}", flush=True)
    print(f"[HARNESS] rounds={num_rounds}", flush=True)

    experiment_start = perf_counter()

    workflow(grid, legacy_context)

    wall_seconds = perf_counter() - experiment_start

    history = legacy_context.history

    losses = history_values_by_round(
        history.losses_distributed
    )

    train_losses = history_values_by_round(
        history.metrics_distributed_fit.get("train_loss", [])
    )

    local_train_times = history_values_by_round(
        history.metrics_distributed_fit.get(
            "local_train_seconds", []
        )
    )

    accuracies = history_values_by_round(
        history.metrics_distributed.get("accuracy", [])
    )

    rounds_csv = results_dir / "rounds.csv"

    with rounds_csv.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "protocol",
            "round",
            "fit_seconds",
            "evaluate_seconds",
            "measured_round_seconds",
            "local_train_seconds",
            "train_loss",
            "eval_loss",
            "accuracy",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for round_number in range(1, num_rounds + 1):
            timing = round_timings.get(round_number, {})

            fit_seconds = float(
                timing.get("fit_seconds", 0.0)
            )

            evaluate_seconds = float(
                timing.get("evaluate_seconds", 0.0)
            )

            writer.writerow(
                {
                    "protocol": protocol,
                    "round": round_number,
                    "fit_seconds": fit_seconds,
                    "evaluate_seconds": evaluate_seconds,
                    "measured_round_seconds": (
                        fit_seconds + evaluate_seconds
                    ),
                    "local_train_seconds": local_train_times.get(
                        round_number
                    ),
                    "train_loss": train_losses.get(
                        round_number
                    ),
                    "eval_loss": losses.get(
                        round_number
                    ),
                    "accuracy": accuracies.get(
                        round_number
                    ),
                }
            )

    summary = {
        "protocol": protocol,
        "seed": seed,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "wall_seconds": wall_seconds,
        "local_epochs": int(
            context.run_config["local-epochs"]
        ),
        "learning_rate": float(
            context.run_config["learning-rate"]
        ),
        "batch_size": int(
            context.run_config["batch-size"]
        ),
    }

    with (results_dir / "run_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print(
        f"[HARNESS] wall_seconds={wall_seconds:.4f}",
        flush=True,
    )

    print(
        f"[HARNESS] results={results_dir}",
        flush=True,
    )
