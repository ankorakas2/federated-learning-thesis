"""Instrumented deterministic FedAvg strategy."""

import random
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from time import perf_counter, sleep

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    RecordDict,
)
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from thesis_baseline.instrumentation import (
    append_csv_row,
    array_record_nbytes,
)


ROUND_FIELDS = [
    "round",
    "logical_client_ids",
    "selected_clients",
    "train_successes",
    "train_failures",
    "evaluate_successes",
    "evaluate_failures",
    "model_bytes_per_client",
    "download_model_bytes",
    "upload_model_bytes",
    "total_training_model_payload_bytes",
    "mean_local_train_seconds",
    "max_local_train_seconds",
    "train_phase_seconds",
    "round_through_evaluation_seconds",
]


class InstrumentedDeterministicFedAvg(FedAvg):
    """FedAvg with deterministic clients and experimental instrumentation."""

    def __init__(
        self,
        sampling_seed: int,
        rounds_csv: Path,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sampling_seed = sampling_seed
        self.rounds_csv = rounds_csv

        self.round_start_times = {}
        self.round_rows = {}

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Select deterministic logical clients and start round timing."""

        if self.fraction_train == 0.0:
            return []

        while True:
            all_node_ids = sorted(grid.get_node_ids())

            if len(all_node_ids) >= self.min_available_nodes:
                break

            sleep(1)

        num_nodes = int(
            len(all_node_ids) * self.fraction_train
        )

        sample_size = max(
            num_nodes,
            self.min_train_nodes,
        )

        if sample_size > len(all_node_ids):
            raise RuntimeError(
                f"Requested {sample_size} clients, "
                f"but only {len(all_node_ids)} are available."
            )

        rng = random.Random(
            self.sampling_seed + server_round
        )

        logical_client_ids = sorted(
            rng.sample(
                range(len(all_node_ids)),
                sample_size,
            )
        )

        selected_node_ids = [
            all_node_ids[logical_id]
            for logical_id in logical_client_ids
        ]

        model_bytes = array_record_nbytes(arrays)

        self.round_start_times[server_round] = perf_counter()

        self.round_rows[server_round] = {
            "round": server_round,
            "logical_client_ids": ",".join(
                str(client_id)
                for client_id in logical_client_ids
            ),
            "selected_clients": len(logical_client_ids),
            "model_bytes_per_client": model_bytes,
            "download_model_bytes": (
                model_bytes * len(logical_client_ids)
            ),
        }

        print(
            f"[SCHEDULE] round={server_round} "
            f"logical_client_ids={logical_client_ids}",
            flush=True,
        )

        messages = []

        for node_id, logical_client_id in zip(
            selected_node_ids,
            logical_client_ids,
            strict=True,
        ):
            client_config = ConfigRecord(
                {
                    key: value
                    for key, value in config.items()
                }
            )

            client_config["server-round"] = server_round
            client_config["logical-client-id"] = logical_client_id

            record = RecordDict(
                {
                    self.arrayrecord_key: arrays,
                    self.configrecord_key: client_config,
                }
            )

            messages.append(
                Message(
                    content=record,
                    message_type=MessageType.TRAIN,
                    dst_node_id=node_id,
                )
            )

        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ):
        """Aggregate training replies and measure training overhead."""

        replies = list(replies)

        valid_replies = [
            reply
            for reply in replies
            if not reply.has_error()
        ]

        failures = len(replies) - len(valid_replies)

        upload_bytes = 0
        local_train_times = []

        for reply in valid_replies:
            for arrays in reply.content.array_records.values():
                upload_bytes += array_record_nbytes(arrays)

            for metrics in reply.content.metric_records.values():
                if "local_train_seconds" in metrics:
                    local_train_times.append(
                        float(metrics["local_train_seconds"])
                    )

        arrays, metrics = super().aggregate_train(
            server_round,
            replies,
        )

        elapsed = (
            perf_counter()
            - self.round_start_times[server_round]
        )

        row = self.round_rows[server_round]

        row["train_successes"] = len(valid_replies)
        row["train_failures"] = failures
        row["upload_model_bytes"] = upload_bytes
        row["total_training_model_payload_bytes"] = (
            row["download_model_bytes"]
            + upload_bytes
        )

        row["mean_local_train_seconds"] = (
            mean(local_train_times)
            if local_train_times
            else 0.0
        )

        row["max_local_train_seconds"] = (
            max(local_train_times)
            if local_train_times
            else 0.0
        )

        row["train_phase_seconds"] = elapsed

        print(
            f"[INSTRUMENT] round={server_round} "
            f"train_seconds={elapsed:.4f} "
            f"download_bytes={row['download_model_bytes']} "
            f"upload_bytes={upload_bytes}",
            flush=True,
        )

        return arrays, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ):
        """Aggregate evaluation and finish round instrumentation."""

        replies = list(replies)

        valid_replies = [
            reply
            for reply in replies
            if not reply.has_error()
        ]

        failures = len(replies) - len(valid_replies)

        metrics = super().aggregate_evaluate(
            server_round,
            replies,
        )

        total_round_time = (
            perf_counter()
            - self.round_start_times[server_round]
        )

        row = self.round_rows[server_round]

        row["evaluate_successes"] = len(valid_replies)
        row["evaluate_failures"] = failures
        row["round_through_evaluation_seconds"] = total_round_time

        append_csv_row(
            self.rounds_csv,
            row,
            ROUND_FIELDS,
        )

        print(
            f"[ROUND_METRICS] round={server_round} "
            f"round_seconds={total_round_time:.4f}",
            flush=True,
        )

        return metrics
