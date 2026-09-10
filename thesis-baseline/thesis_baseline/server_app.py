"""Federated Learning thesis baseline using Flower and PyTorch."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from thesis_baseline.strategy import DeterministicFedAvg

from thesis_baseline.task import Net, load_centralized_dataset, set_seed, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    seed = int(context.run_config["seed"])
    set_seed(seed)

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = DeterministicFedAvg(
	sampling_seed=seed,
        fraction_train=fraction_train, 
        fraction_evaluate=fraction_evaluate, 
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=4,)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
