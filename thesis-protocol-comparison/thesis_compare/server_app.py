"""Unified Plain FedAvg versus SecAgg+ ServerApp."""

import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

from flwr.common import (
    Context,
    Metrics,
    ndarrays_to_parameters,
)
from flwr.server import (
    Grid,
    LegacyContext,
    ServerApp,
    ServerConfig,
)
from flwr.server.strategy import FedAvg
from flwr.server.workflow import (
    DefaultWorkflow,
    SecAggPlusWorkflow,
)

from thesis_compare.task import (
    get_weights,
    make_net,
)


def aggregate_fit_metrics(metrics):
    """Weighted aggregation of local training metrics."""

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

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
    """Weighted evaluation accuracy."""

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
    """Create results directory."""

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


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run one side of the controlled protocol comparison."""

    protocol = str(
        context.run_config["protocol"]
    )

    if protocol not in {
        "plain",
        "secagg",
    }:
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
        get_weights(
            make_net(seed)
        )
    )

    # All clients participate.
    # This removes client-sampling as a confounder
    # in the first Plain-vs-SecAgg+ comparison.
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

    if protocol == "plain":
        workflow = DefaultWorkflow()

    else:
        workflow = DefaultWorkflow(
            fit_workflow=SecAggPlusWorkflow(
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
        )

    results_dir = create_results_dir(
        protocol,
        seed,
    )

    print(
        f"[HARNESS] protocol={protocol}",
        flush=True,
    )

    print(
        f"[HARNESS] clients={num_clients}",
        flush=True,
    )

    start = perf_counter()

    workflow(
        grid,
        legacy_context,
    )

    wall_seconds = (
        perf_counter() - start
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

    with (
        results_dir / "run_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        f"[HARNESS] wall_seconds={wall_seconds:.4f}",
        flush=True,
    )

    print(
        f"[HARNESS] results={results_dir}",
        flush=True,
    )
