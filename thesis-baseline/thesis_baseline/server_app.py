"""Federated Learning thesis ServerApp."""

from time import perf_counter

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from thesis_baseline.instrumentation import (
    array_record_nbytes,
    create_run_directory,
    metric_history_to_dict,
    save_summary,
)
from thesis_baseline.strategy import (
    InstrumentedDeterministicFedAvg,
)
from thesis_baseline.task import (
    Net,
    load_centralized_dataset,
    set_seed,
    test,
)


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run instrumented deterministic FedAvg."""

    fraction_train = float(
        context.run_config["fraction-train"]
    )

    fraction_evaluate = float(
        context.run_config["fraction-evaluate"]
    )

    num_rounds = int(
        context.run_config["num-server-rounds"]
    )

    lr = float(
        context.run_config["learning-rate"]
    )

    seed = int(
        context.run_config["seed"]
    )

    set_seed(seed)

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    run_dir = create_run_directory(
        protocol="plain-fedavg",
        seed=seed,
    )

    rounds_csv = run_dir / "rounds.csv"

    print(
        f"[RESULTS] run_dir={run_dir}",
        flush=True,
    )

    strategy = InstrumentedDeterministicFedAvg(
        sampling_seed=seed,
        rounds_csv=rounds_csv,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=4,
    )

    experiment_start = perf_counter()

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    total_strategy_seconds = (
        perf_counter() - experiment_start
    )

    summary = {
        "protocol": "plain-fedavg",
        "seed": seed,
        "num_rounds": num_rounds,
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "learning_rate": lr,
        "total_strategy_seconds": total_strategy_seconds,
        "final_model_bytes": array_record_nbytes(
            result.arrays
        ),
        "train_metrics": metric_history_to_dict(
            result.train_metrics_clientapp
        ),
        "client_evaluate_metrics": metric_history_to_dict(
            result.evaluate_metrics_clientapp
        ),
        "server_evaluate_metrics": metric_history_to_dict(
            result.evaluate_metrics_serverapp
        ),
    }

    save_summary(
        run_dir / "summary.json",
        summary,
    )

    print(
        f"[RESULTS] total_strategy_seconds="
        f"{total_strategy_seconds:.4f}",
        flush=True,
    )

    print(
        f"[RESULTS] saved={run_dir}",
        flush=True,
    )

    if context.run_config["save-model"]:
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(
            state_dict,
            run_dir / "final_model.pt",
        )


def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
) -> MetricRecord:
    """Evaluate global model on centralized CIFAR-10 test set."""

    model = Net()
    model.load_state_dict(
        arrays.to_torch_state_dict()
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    test_dataloader = load_centralized_dataset()

    test_loss, test_acc = test(
        model,
        test_dataloader,
        device,
    )

    return MetricRecord(
        {
            "accuracy": test_acc,
            "loss": test_loss,
        }
    )
