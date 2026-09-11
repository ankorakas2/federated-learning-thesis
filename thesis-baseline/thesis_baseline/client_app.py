"""Federated Learning thesis baseline using Flower and PyTorch."""

import torch
from time import perf_counter
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from thesis_baseline.task import Net, load_data
from thesis_baseline.task import test as test_fn
from thesis_baseline.task import train as train_fn


app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on deterministic logical-client data."""

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Flower's runtime node/partition identity
    runtime_partition_id = int(context.node_config["partition-id"])

    # Stable logical client selected by our deterministic strategy
    logical_client_id = int(
        msg.content["config"]["logical-client-id"]
    )

    partition_id = logical_client_id
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    seed = int(context.run_config["seed"])
    server_round = int(msg.content["config"]["server-round"])

    print(
        f"[TRAIN] round={server_round} "
        f"node_id={context.node_id} "
        f"logical_client_id={logical_client_id} "
        f"runtime_partition_id={runtime_partition_id}",
        flush=True,
    )

    trainloader, _ = load_data(
        partition_id,
        num_partitions,
        batch_size,
        seed,
    )

    train_start = perf_counter()

    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    local_train_seconds = perf_counter() - train_start

    model_record = ArrayRecord(model.state_dict())

    metrics = {
        "train_loss": train_loss,
        "local_train_seconds": local_train_seconds,
        "num-examples": len(trainloader.dataset),
    }

    metric_record = MetricRecord(metrics)

    content = RecordDict(
        {
            "arrays": model_record,
            "metrics": metric_record,
        }
    )

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local validation data."""

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    seed = int(context.run_config["seed"])

    _, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size,
        seed,
    )

    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
