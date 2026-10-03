from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.home() / "federated-learning-thesis"

CORE = ROOT / "results/system-scaling-final-v1"
DROP = ROOT / "results/dropout-final-v1"
OUT = ROOT / "analysis/final_results"

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

core_runs = pd.read_csv(CORE / "all_runs.csv")
core_rounds = pd.read_csv(CORE / "all_rounds.csv")

drop_runs = pd.read_csv(DROP / "dropout_all_runs.csv")
drop_rounds = pd.read_csv(DROP / "dropout_all_rounds.csv")
drop_status = pd.read_csv(DROP / "dropout_round_status.csv")


def save_figure(fig, filename):
    fig.tight_layout()

    fig.savefig(
        OUT / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{filename}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# RQ1 + RQ2 SUMMARY
# ============================================================

rq12 = (
    core_runs
    .groupby(["clients", "protocol"], as_index=False)
    .agg(
        runs=("seed", "count"),

        accuracy_mean=("final_accuracy", "mean"),
        accuracy_sd=("final_accuracy", "std"),

        eval_loss_mean=("final_eval_loss", "mean"),
        eval_loss_sd=("final_eval_loss", "std"),

        wall_seconds_mean=("wall_seconds", "mean"),
        wall_seconds_sd=("wall_seconds", "std"),

        communication_bytes_mean=(
            "serialized_total_bytes",
            "mean",
        ),
        communication_bytes_sd=(
            "serialized_total_bytes",
            "std",
        ),

        cpu_mean_percent=(
            "host_cpu_mean_percent",
            "mean",
        ),

        memory_peak_percent_mean=(
            "host_memory_peak_percent",
            "mean",
        ),
    )
)

rq12["communication_MB_mean"] = (
    rq12["communication_bytes_mean"] / 1_000_000
)

rq12["communication_MB_sd"] = (
    rq12["communication_bytes_sd"] / 1_000_000
)

rq12.to_csv(
    OUT / "table_rq1_rq2_summary.csv",
    index=False,
)


# ============================================================
# PAIRED PLAIN vs SECAGG+
# ============================================================

paired = []

for clients in [20, 50, 100]:

    for seed in [42, 123, 2026]:

        plain = core_runs[
            (core_runs["clients"] == clients)
            & (core_runs["seed"] == seed)
            & (core_runs["protocol"] == "plain")
        ].iloc[0]

        secagg = core_runs[
            (core_runs["clients"] == clients)
            & (core_runs["seed"] == seed)
            & (core_runs["protocol"] == "secagg")
        ].iloc[0]

        paired.append(
            {
                "clients": clients,
                "seed": seed,

                "accuracy_difference_pp":
                    (
                        secagg["final_accuracy"]
                        - plain["final_accuracy"]
                    ) * 100,

                "runtime_overhead_percent":
                    (
                        secagg["wall_seconds"]
                        / plain["wall_seconds"]
                        - 1
                    ) * 100,

                "communication_overhead_percent":
                    (
                        secagg["serialized_total_bytes"]
                        / plain["serialized_total_bytes"]
                        - 1
                    ) * 100,
            }
        )

paired = pd.DataFrame(paired)

paired.to_csv(
    OUT / "table_rq2_paired_runs.csv",
    index=False,
)

paired_summary = (
    paired
    .groupby("clients", as_index=False)
    .agg(
        accuracy_difference_pp_mean=(
            "accuracy_difference_pp",
            "mean",
        ),
        accuracy_difference_pp_sd=(
            "accuracy_difference_pp",
            "std",
        ),

        runtime_overhead_percent_mean=(
            "runtime_overhead_percent",
            "mean",
        ),
        runtime_overhead_percent_sd=(
            "runtime_overhead_percent",
            "std",
        ),

        communication_overhead_percent_mean=(
            "communication_overhead_percent",
            "mean",
        ),
        communication_overhead_percent_sd=(
            "communication_overhead_percent",
            "std",
        ),
    )
)

paired_summary.to_csv(
    OUT / "table_rq2_paired_summary.csv",
    index=False,
)


# ============================================================
# RQ3 SUMMARY
# ============================================================

rq3 = (
    drop_runs
    .groupby(["dropout_rate", "protocol"], as_index=False)
    .agg(
        runs=("seed", "count"),

        accuracy_mean=("final_accuracy", "mean"),
        accuracy_sd=("final_accuracy", "std"),

        eval_loss_mean=("final_eval_loss", "mean"),
        eval_loss_sd=("final_eval_loss", "std"),

        wall_seconds_mean=("wall_seconds", "mean"),
        wall_seconds_sd=("wall_seconds", "std"),

        communication_bytes_mean=(
            "serialized_total_bytes",
            "mean",
        ),
        communication_bytes_sd=(
            "serialized_total_bytes",
            "std",
        ),

        completed_rounds=("completed_rounds", "sum"),
        halted_rounds=("halted_rounds", "sum"),
    )
)

rq3["round_completion_percent"] = (
    rq3["completed_rounds"]
    / (
        rq3["completed_rounds"]
        + rq3["halted_rounds"]
    )
    * 100
)

rq3["communication_MB_mean"] = (
    rq3["communication_bytes_mean"] / 1_000_000
)

rq3.to_csv(
    OUT / "table_rq3_summary.csv",
    index=False,
)

drop_status.to_csv(
    OUT / "table_rq3_per_seed_status.csv",
    index=False,
)


# ============================================================
# FIGURES — RQ1 LEARNING CURVES
# ============================================================

for clients in [20, 50, 100]:

    subset = core_rounds[
        core_rounds["clients"] == clients
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for protocol in ["plain", "secagg"]:

        data = subset[
            subset["protocol"] == protocol
        ]

        grouped = (
            data
            .groupby("round")["accuracy"]
            .agg(["mean", "std"])
            .reset_index()
        )

        label = (
            "Plain FedAvg"
            if protocol == "plain"
            else "SecAgg+"
        )

        ax.plot(
            grouped["round"],
            grouped["mean"],
            marker="o",
            label=label,
        )

        ax.fill_between(
            grouped["round"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15,
        )

    ax.set_title(
        f"RQ1 — Learning Curve ({clients} Clients)"
    )

    ax.set_xlabel(
        "Communication Round (Γύρος επικοινωνίας)"
    )

    ax.set_ylabel(
        "Accuracy (Ακρίβεια)"
    )

    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.25)
    ax.legend()

    save_figure(
        fig,
        f"fig_rq1_learning_curve_{clients}",
    )


# ============================================================
# FINAL ACCURACY vs CLIENT COUNT
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

for protocol in ["plain", "secagg"]:

    data = rq12[
        rq12["protocol"] == protocol
    ].sort_values("clients")

    label = (
        "Plain FedAvg"
        if protocol == "plain"
        else "SecAgg+"
    )

    ax.errorbar(
        data["clients"],
        data["accuracy_mean"],
        yerr=data["accuracy_sd"],
        marker="o",
        capsize=4,
        label=label,
    )

ax.set_title(
    "RQ1 — Final Accuracy vs Number of Clients"
)

ax.set_xlabel("Clients (Πελάτες)")
ax.set_ylabel("Final Accuracy (Τελική ακρίβεια)")
ax.set_xticks([20, 50, 100])
ax.grid(alpha=0.25)
ax.legend()

save_figure(
    fig,
    "fig_rq1_final_accuracy_vs_clients",
)


# ============================================================
# COMMUNICATION vs CLIENT COUNT
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

for protocol in ["plain", "secagg"]:

    data = rq12[
        rq12["protocol"] == protocol
    ].sort_values("clients")

    label = (
        "Plain FedAvg"
        if protocol == "plain"
        else "SecAgg+"
    )

    ax.errorbar(
        data["clients"],
        data["communication_MB_mean"],
        yerr=data["communication_MB_sd"],
        marker="o",
        capsize=4,
        label=label,
    )

ax.set_title(
    "RQ2 — Serialized Communication vs Number of Clients"
)

ax.set_xlabel("Clients (Πελάτες)")
ax.set_ylabel(
    "Serialized Communication (MB)"
)

ax.set_xticks([20, 50, 100])
ax.grid(alpha=0.25)
ax.legend()

save_figure(
    fig,
    "fig_rq2_communication_vs_clients",
)


# ============================================================
# RUNTIME vs CLIENT COUNT
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

for protocol in ["plain", "secagg"]:

    data = rq12[
        rq12["protocol"] == protocol
    ].sort_values("clients")

    label = (
        "Plain FedAvg"
        if protocol == "plain"
        else "SecAgg+"
    )

    ax.errorbar(
        data["clients"],
        data["wall_seconds_mean"],
        yerr=data["wall_seconds_sd"],
        marker="o",
        capsize=4,
        label=label,
    )

ax.set_title(
    "RQ2 — Wall-clock Runtime vs Number of Clients"
)

ax.set_xlabel("Clients (Πελάτες)")
ax.set_ylabel("Wall-clock Time (s)")
ax.set_xticks([20, 50, 100])
ax.grid(alpha=0.25)
ax.legend()

save_figure(
    fig,
    "fig_rq2_runtime_vs_clients",
)


# ============================================================
# RQ3 ROUND COMPLETION
# ============================================================

completion = (
    drop_status
    .groupby(["dropout_rate", "protocol"], as_index=False)
    .agg(
        completion_mean=(
            "round_completion_rate",
            "mean",
        ),
        completion_sd=(
            "round_completion_rate",
            "std",
        ),
    )
)

fig, ax = plt.subplots(figsize=(8, 5))

for protocol in ["plain", "secagg"]:

    data = completion[
        completion["protocol"] == protocol
    ].sort_values("dropout_rate")

    label = (
        "Plain FedAvg"
        if protocol == "plain"
        else "SecAgg+"
    )

    ax.errorbar(
        data["dropout_rate"] * 100,
        data["completion_mean"] * 100,
        yerr=data["completion_sd"] * 100,
        marker="o",
        capsize=4,
        label=label,
    )

ax.set_title(
    "RQ3 — Round Completion under Client Dropout"
)

ax.set_xlabel(
    "Dropout Rate (Ποσοστό αποσύνδεσης, %)"
)

ax.set_ylabel(
    "Round Completion Rate (%)"
)

ax.set_xticks([0, 5, 10])
ax.set_ylim(0, 105)
ax.grid(alpha=0.25)
ax.legend()

save_figure(
    fig,
    "fig_rq3_round_completion",
)


# ============================================================
# RQ3 ACCURACY
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

for protocol in ["plain", "secagg"]:

    data = rq3[
        rq3["protocol"] == protocol
    ].sort_values("dropout_rate")

    label = (
        "Plain FedAvg"
        if protocol == "plain"
        else "SecAgg+"
    )

    ax.errorbar(
        data["dropout_rate"] * 100,
        data["accuracy_mean"],
        yerr=data["accuracy_sd"],
        marker="o",
        capsize=4,
        label=label,
    )

ax.set_title(
    "RQ3 — Final Accuracy under Client Dropout"
)

ax.set_xlabel(
    "Dropout Rate (Ποσοστό αποσύνδεσης, %)"
)

ax.set_ylabel(
    "Final Observed Accuracy"
)

ax.set_xticks([0, 5, 10])
ax.grid(alpha=0.25)
ax.legend()

save_figure(
    fig,
    "fig_rq3_accuracy_vs_dropout",
)


# ============================================================
# TEXT REPORT
# ============================================================

with (
    OUT / "analysis_summary.txt"
).open(
    "w",
    encoding="utf-8",
) as f:

    f.write("RQ1 / RQ2 SUMMARY\n")
    f.write("=================\n\n")
    f.write(rq12.to_string(index=False))

    f.write("\n\nPAIRED OVERHEAD SUMMARY\n")
    f.write("=======================\n\n")
    f.write(
        paired_summary.to_string(index=False)
    )

    f.write("\n\nRQ3 SUMMARY\n")
    f.write("===========\n\n")
    f.write(rq3.to_string(index=False))

    f.write("\n\nRQ3 PER-SEED STATUS\n")
    f.write("===================\n\n")
    f.write(
        drop_status.to_string(index=False)
    )


print()
print("==============================================")
print("FINAL ANALYSIS PACKAGE CREATED")
print("==============================================")
print()

print("Output directory:")
print(OUT)

print()

for path in sorted(OUT.iterdir()):
    print(path.name)
