"""Measurement utilities for thesis experiments."""

import fcntl
import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

import psutil
from flwr.common.serde import message_to_proto


def message_nbytes(message) -> int:
    """Serialized Flower Message size in protobuf bytes."""
    return message_to_proto(message).ByteSize()


def append_message_trace(
    path: str,
    *,
    protocol: str,
    partition_id: int,
    message_type: str,
    group_id: str,
    incoming_bytes: int,
    outgoing_bytes: int,
) -> None:
    """Append one client-boundary message measurement safely."""

    if not path:
        return

    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "protocol": protocol,
        "partition_id": partition_id,
        "message_type": message_type,
        "group_id": group_id,
        "server_to_client_bytes": incoming_bytes,
        "client_to_server_bytes": outgoing_bytes,
    }

    with trace_path.open("a", encoding="utf-8") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        file.write(json.dumps(record) + "\n")
        file.flush()
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def summarize_message_trace(path: str) -> dict:
    """Summarize serialized Flower application-message traffic."""

    overall = {
        "server_to_client_bytes": 0,
        "client_to_server_bytes": 0,
        "total_bytes": 0,
        "message_exchanges": 0,
    }

    per_round = defaultdict(
        lambda: {
            "server_to_client_bytes": 0,
            "client_to_server_bytes": 0,
            "total_bytes": 0,
            "message_exchanges": 0,
            "train_bytes": 0,
            "evaluate_bytes": 0,
        }
    )

    if not path or not Path(path).exists():
        return {
            "overall": overall,
            "per_round": {},
        }

    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            down = int(record["server_to_client_bytes"])
            up = int(record["client_to_server_bytes"])
            total = down + up

            overall["server_to_client_bytes"] += down
            overall["client_to_server_bytes"] += up
            overall["total_bytes"] += total
            overall["message_exchanges"] += 1

            group_id = str(record.get("group_id", ""))

            try:
                round_number = int(group_id)
            except (TypeError, ValueError):
                continue

            round_data = per_round[round_number]

            round_data["server_to_client_bytes"] += down
            round_data["client_to_server_bytes"] += up
            round_data["total_bytes"] += total
            round_data["message_exchanges"] += 1

            message_type = str(
                record.get("message_type", "")
            ).lower()

            if "train" in message_type:
                round_data["train_bytes"] += total
            elif "evaluate" in message_type:
                round_data["evaluate_bytes"] += total

    return {
        "overall": overall,
        "per_round": dict(per_round),
    }


class HostMonitor:
    """Sample host-level CPU, RAM and swap usage."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def _sample(self):
        psutil.cpu_percent(interval=None)

        while not self._stop_event.is_set():
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()

            self.samples.append(
                {
                    "cpu_percent": psutil.cpu_percent(
                        interval=None
                    ),
                    "memory_used_bytes": int(vm.used),
                    "memory_percent": float(vm.percent),
                    "swap_used_bytes": int(swap.used),
                }
            )

            time.sleep(self.interval)

    def start(self):
        self._thread = threading.Thread(
            target=self._sample,
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        if not self.samples:
            return {}

        cpu = [x["cpu_percent"] for x in self.samples]
        memory = [
            x["memory_used_bytes"]
            for x in self.samples
        ]
        memory_pct = [
            x["memory_percent"]
            for x in self.samples
        ]
        swap = [
            x["swap_used_bytes"]
            for x in self.samples
        ]

        return {
            "sample_count": len(self.samples),
            "host_cpu_mean_percent": sum(cpu) / len(cpu),
            "host_cpu_peak_percent": max(cpu),
            "host_memory_peak_bytes": max(memory),
            "host_memory_peak_percent": max(memory_pct),
            "host_swap_peak_bytes": max(swap),
        }
