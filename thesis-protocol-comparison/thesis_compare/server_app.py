"""Fully instrumented Plain FedAvg versus SecAgg+ ServerApp."""

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

from thesis_compare.measurement import (
    HostMonitor,
    summarize_message_trace,
)
from thesis_compare.task import (
    get_weights,
    make_net,
)


def aggregate_fit_metrics(metrics):
    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    train_loss = (
        sum(
            num_examples * values["train_loss"]
            for num_examples, values in metrics
        )
        / total_examples
    )

    local_times = [
        float(values["local_train_seconds"])
        for _, values in metrics
    ]

    return {
        "train_loss": train_loss,
        "local_train_seconds_mean": (
            sum(local_times) / len(local_times)
        ),
        "local_train_seconds_min": min(local_times),
        "local_train_seconds_max": max(local_times),
        "raw_model_download_bytes_total": sum(
            int(values["raw_model_download_bytes"])
            for _, values in metrics
        ),
        "raw_model_upload_bytes_total": sum(
            int(values["raw_model_upload_bytes"])
            for _, values in metrics
        ),
    }


def aggregate_evaluate_metrics(metrics):
    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

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
            Path.home()
            / "federated-learning-thesis"
            / "results",
        )
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    path = root / (
        f"{timestamp}_comparison_"
        f"{protocol}_seed{seed}"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def history_values_by_round(entries):
    return {
        int(round_number): value
        for round_number, value in entries
    }


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    protocol = str(
        context.run_config["protocol"]
    )

    if protocol not in {"plain", "secagg"}:
        raise ValueError(
            "protocol must be 'plain' or 'secagg'"
        )

    seed = int(
        context.run_config["seed"]
    )

    num_clients = int(
        context.run_config["num-clients"]
    )

    num_rounds = int(
        context.run_config["num-server-rounds"]
    )

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
        config=ServerConfig(
            num_rounds=num_rounds
        ),
        strategy=strategy,
    )

    results_dir = create_results_dir(
        protocol,
        seed,
    )

    round_timings = {}

    if protocol == "plain":
        base_fit_workflow = default_fit_workflow
    else:
        base_fit_workflow = SecAggPlusWorkflow(
            num_shares=int(
                context.run_config["num-shares"]
            ),
            reconstruction_threshold=int(
                context.run_config[
                    "reconstruction-threshold"
                ]
            ),
            max_weight=float(
                context.run_config["max-weight"]
            ),
        )

    def timed_fit_workflow(
        grid,
        legacy_context,
    ):
        current_round = int(
            legacy_context.state.config_records[
                MAIN_CONFIGS_RECORD
            ][Key.CURRENT_ROUND]
        )

        start = perf_counter()

        base_fit_workflow(
            grid,
            legacy_context,
        )

        elapsed = (
            perf_counter() - start
        )

        round_timings.setdefault(
            current_round,
            {},
        )

        round_timings[current_round][
            "fit_seconds"
        ] = elapsed

        print(
            f"[ROUND_TIMING] round={current_round} "
            f"fit_seconds={elapsed:.4f}",
            flush=True,
        )

    def timed_evaluate_workflow(
        grid,
        legacy_context,
    ):
        current_round = int(
            legacy_context.state.config_records[
                MAIN_CONFIGS_RECORD
            ][Key.CURRENT_ROUND]
        )

        start = perf_counter()

        default_evaluate_workflow(
            grid,
            legacy_context,
        )

        elapsed = (
            perf_counter() - start
        )

        round_timings.setdefault(
            current_round,
            {},
        )

        round_timings[current_round][
            "evaluate_seconds"
        ] = elapsed

        print(
            f"[ROUND_TIMING] round={current_round} "
            f"evaluate_seconds={elapsed:.4f}",
            flush=True,
        )

    workflow = DefaultWorkflow(
        fit_workflow=timed_fit_workflow,
        evaluate_workflow=timed_evaluate_workflow,
    )

    print(
        f"[HARNESS] protocol={protocol}",
        flush=True,
    )
    print(
        f"[HARNESS] clients={num_clients}",
        flush=True,
    )
    print(
        f"[HARNESS] rounds={num_rounds}",
        flush=True,
    )

    monitor = HostMonitor(interval=0.5)
    monitor.start()

    experiment_start = perf_counter()

    workflow(
        grid,
        legacy_context,
    )

    wall_seconds = (
        perf_counter() - experiment_start
    )

    monitor.stop()
    host_metrics = monitor.summary()

    history = legacy_context.history

    losses = history_values_by_round(
        history.losses_distributed
    )

    train_losses = history_values_by_round(
        history.metrics_distributed_fit.get(
            "train_loss",
            [],
        )
    )

    local_mean = history_values_by_round(
        history.metrics_distributed_fit.get(
            "local_train_seconds_mean",
            [],
        )
    )

    local_min = history_values_by_round(
        history.metrics_distributed_fit.get(
            "local_train_seconds_min",
            [],
        )
    )

    local_max = history_values_by_round(
        history.metrics_distributed_fit.get(
            "local_train_seconds_max",
            [],
        )
    )

    raw_download = history_values_by_round(
        history.metrics_distributed_fit.get(
            "raw_model_download_bytes_total",
            [],
        )
    )

    raw_upload = history_values_by_round(
        history.metrics_distributed_fit.get(
            "raw_model_upload_bytes_total",
            [],
        )
    )

    accuracies = history_values_by_round(
        history.metrics_distributed.get(
            "accuracy",
            [],
        )
    )

    comm_log = os.environ.get(
        "THESIS_COMM_LOG",
        "",
    )

    comm = summarize_message_trace(
        comm_log
    )

    rounds_csv = (
        results_dir / "rounds.csv"
    )

    with rounds_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "protocol",
            "clients",
            "round",
            "fit_seconds",
            "evaluate_seconds",
            "measured_round_seconds",
            "local_train_seconds_min",
            "local_train_seconds_mean",
            "local_train_seconds_max",
            "train_loss",
            "eval_loss",
            "accuracy",
            "raw_model_download_bytes_total",
            "raw_model_upload_bytes_total",
            "serialized_server_to_client_bytes",
            "serialized_client_to_server_bytes",
            "serialized_total_bytes",
            "serialized_train_bytes",
            "serialized_evaluate_bytes",
            "message_exchanges",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for round_number in range(
            1,
            num_rounds + 1,
        ):
            timing = round_timings.get(
                round_number,
                {},
            )

            fit_seconds = float(
                timing.get(
                    "fit_seconds",
                    0.0,
                )
            )

            evaluate_seconds = float(
                timing.get(
                    "evaluate_seconds",
                    0.0,
                )
            )

            comm_round = (
                comm["per_round"].get(
                    round_number,
                    {},
                )
            )

            writer.writerow(
                {
                    "protocol": protocol,
                    "clients": num_clients,
                    "round": round_number,
                    "fit_seconds": fit_seconds,
                    "evaluate_seconds": evaluate_seconds,
                    "measured_round_seconds": (
                        fit_seconds
                        + evaluate_seconds
                    ),
                    "local_train_seconds_min": (
                        local_min.get(round_number)
                    ),
                    "local_train_seconds_mean": (
                        local_mean.get(round_number)
                    ),
                    "local_train_seconds_max": (
                        local_max.get(round_number)
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
                    "raw_model_download_bytes_total": (
                        raw_download.get(round_number)
                    ),
                    "raw_model_upload_bytes_total": (
                        raw_upload.get(round_number)
                    ),
                    "serialized_server_to_client_bytes": (
                        comm_round.get(
                            "server_to_client_bytes",
                            0,
                        )
                    ),
                    "serialized_client_to_server_bytes": (
                        comm_round.get(
                            "client_to_server_bytes",
                            0,
                        )
                    ),
                    "serialized_total_bytes": (
                        comm_round.get(
                            "total_bytes",
                            0,
                        )
                    ),
                    "serialized_train_bytes": (
                        comm_round.get(
                            "train_bytes",
                            0,
                        )
                    ),
                    "serialized_evaluate_bytes": (
                        comm_round.get(
                            "evaluate_bytes",
                            0,
                        )
                    ),
                    "message_exchanges": (
                        comm_round.get(
                            "message_exchanges",
                            0,
                        )
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
            context.run_config[
                "local-epochs"
            ]
        ),
        "learning_rate": float(
            context.run_config[
                "learning-rate"
            ]
        ),
        "batch_size": int(
            context.run_config[
                "batch-size"
            ]
        ),
        "num_shares": (
            int(
                context.run_config[
                    "num-shares"
                ]
            )
            if protocol == "secagg"
            else None
        ),
        "reconstruction_threshold": (
            int(
                context.run_config[
                    "reconstruction-threshold"
                ]
            )
            if protocol == "secagg"
            else None
        ),
        "serialized_communication": (
            comm["overall"]
        ),
        "host_metrics": host_metrics,
        "communication_metric_definition": (
            "Serialized Flower protobuf application-message "
            "bytes measured at the ClientApp boundary; "
            "not TCP/Ethernet traffic."
        ),
    }

    with (
        results_dir
        / "run_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        f"[HARNESS] wall_seconds="
        f"{wall_seconds:.4f}",
        flush=True,
    )

    print(
        f"[COMM] total_serialized_bytes="
        f"{comm['overall']['total_bytes']}",
        flush=True,
    )

    print(
        f"[HOST] peak_memory_percent="
        f"{host_metrics.get('host_memory_peak_percent')}",
        flush=True,
    )

    print(
        f"[HARNESS] results="
        f"{results_dir}",
        flush=True,
    )
