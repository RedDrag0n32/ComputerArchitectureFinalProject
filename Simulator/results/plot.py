import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Clean conversion from x% to just x
def clean_percent(value):
    if isinstance(value, str):
        return float(value.replace("%", "").strip())
    return float(value)

def clean_perf(value):
    if isinstance(value, str):
        if "N/A" in value:
            return 0.0
        return float(value.replace("%/bit", "").strip())
    return float(value)
    

def load_results(csv_file):
    file = pd.read_csv(csv_file)

    file["accuracy"] = file["accuracy"].apply(clean_percent)
    file["btb_hit_rate"] = file["btb_hit_rate"].apply(clean_percent)

    return file

# We make one function per figure required

def plot_accuracy_bar(file):
    workloads = file["workload"].unique()
    predictors = file["predictor"].unique()

    x = np.arange(len(workloads))
    width = .25

    plt.figure(figsize=(8, 5))

    for i, predictor in enumerate(predictors):
        values = [
            file[(file["workload"] == w) & (file["predictor"] == predictor)]["accuracy"].values[0]
            for w in workloads
        ]
        plt.bar(x + i * width, values, width, label=predictor)

    plt.xlabel("Workload")
    plt.ylabel("Prediction Accuracy (%)")
    plt.title("Prediction Accuracy by Workload")
    plt.xticks(x + width, workloads)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig1_accuracy.png")
    plt.close()


def plot_cpi_bar(file):
    workloads = file["workload"].unique()
    predictors = file["predictor"].unique()

    x = np.arange(len(workloads))
    width = .25

    plt.figure(figsize=(8, 5))

    for i, predictor in enumerate(predictors):
        values = [
            file[(file["workload"] == w) & (file["predictor"] == predictor)]["actual_cpi"].values[0]
            for w in workloads
        ]
        
        plt.bar(x + i * width, values, width, label=predictor)

    plt.axhline(y=1.0, linestyle="--", label="Ideal CPI = 1")

    plt.xlabel("Workload")
    plt.ylabel("Actual CPI")
    plt.title("CPI Comparison by Workload")
    plt.xticks(x + width, workloads)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig2_cpi.png")
    plt.close()

def plot_performance_per_bit(file):
    file["performance_per_bit_clean"] = file["performance_per_bit"].apply(clean_perf)

    plt.figure(figsize=(8, 5))
    labels = file["workload"] + " - " + file["predictor"]

    plt.bar(labels, file["performance_per_bit_clean"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Performance per Bit (%/bit)")
    plt.title("Performance per Bit Comparison")
    plt.tight_layout()
    plt.savefig("fig5_performance_per_bit.png")
    plt.close()

def plot_btb_hit_rate(csv_file="BTBResults.csv"):
    file = pd.read_csv(csv_file)

    file["btb_entries"] = file["btb_entries"].astype(int)
    file["btb_hit_rate"] = file["btb_hit_rate"].apply(clean_percent)

    plt.figure(figsize=(8, 5))

    for workload in file["workload"].unique():
        subset = file[file["workload"] == workload]
        subset = subset.sort_values("btb_entries")

        plt.plot(
            subset["btb_entries"],
            subset["btb_hit_rate"],
            marker="o",
            label = workload
        )

    plt.xscale("log", base=2)
    plt.xlabel("BTB Entries")
    plt.ylabel("BTB Hit Rate (%)")
    plt.title("BTB Hit Rate vs. BTB Size")
    plt.xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig6_btb_hit_rate.png", dpi=300)
    plt.show()

def plot_storage_vs_accuracy(csv_file="TwoBitStorageExperiment.csv"):
    file = pd.read_csv(csv_file)

    file["storage_bits"] = file["storage_bits"].astype(int)
    file["accuracy"] = file["accuracy"].apply(clean_percent)

    plt.figure(figsize=(8, 5))

    for workload in file["workload"].unique():
        subset = file[file["workload"] == workload].sort_values("storage_bits")

        plt.plot(
            subset["storage_bits"],
            subset["accuracy"],
            marker="o",
            label=workload
        )

    plt.xscale("log", base=2)
    plt.xlabel("2-bit Predictor Storage Bits")
    plt.ylabel("Prediction Accuracy (%)")
    plt.title("Storage vs. Accuracy for 2-bit Predictor")
    plt.xticks([4, 8, 32, 128, 512, 2048])
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig3_storage_vs_accuracy.png", dpi=300)
    plt.show()

def plot_gshare_history_vs_accuracy(csv_file="GShareHistoryBits.csv"):
    file = pd.read_csv(csv_file)

    file["history_bits"] = file["history_bits"].astype(int)
    file["accuracy"] = file["accuracy"].apply(clean_percent)

    plt.figure(figsize=(8, 5))

    for workload in file["workload"].unique():
        subset = file[file["workload"] == workload].sort_values("history_bits")

        plt.plot(
            subset["history_bits"],
            subset["accuracy"],
            marker="o",
            label=workload
        )

    plt.annotate(
        "Sweet spot: H=6",
        xy=(6, 77.27),
        xytext=(8, 70),
        arrowprops=dict(arrowstyle="->")
    )

    plt.xlabel("Gshare History Length (bits)")
    plt.ylabel("Prediction Accuracy (%)")
    plt.title("Gshare History Length vs. Accuracy")
    plt.xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig4_gshare_history.png", dpi=300)
    plt.show()

def main():
    file = load_results("results.csv")

    plot_accuracy_bar(file)
    plot_cpi_bar(file)
    plot_storage_vs_accuracy()
    plot_gshare_history_vs_accuracy()
    plot_performance_per_bit(file)
    plot_btb_hit_rate()


if __name__ == "__main__":
    main()