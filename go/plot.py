import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pandas import read_json
from argparse import ArgumentParser


def plot_recall_vs_time(dataset, PROJECT_ROOT):
    recall_file = PROJECT_ROOT / f"result/{dataset}/pir.jsonl"

    mean_recall = []
    std_recall = []
    mean_time = []
    std_time = []

    df = read_json(recall_file, lines=True)
    for beamw in np.arange(2, 32, 2):
        df_beamw = df[df["beamw"] == beamw]
        if df_beamw.empty:
            continue
        df_recall = df_beamw["recall@10"]
        df_time = df_beamw["time"] * 1000  # convert to milliseconds
        mean_recall.append(df_recall.mean())
        std_recall.append(df_recall.std())
        mean_time.append(df_time.mean())
        std_time.append(df_time.std())

    fig, ax = plt.subplots(figsize=(3, 3), layout="constrained")
    ax.errorbar(
        mean_time,
        mean_recall,
        xerr=std_time,
        yerr=std_recall,
        marker="o",
        label="SPRINT",
    )
    ax.set_xlabel("Total Server Time Per Query (ms)")
    ax.set_ylabel("10-Recall@10")
    ax.set_title(f"{dataset.title()}: Recall vs Time")
    ax.grid(True)
    ax.legend()

    img_file = PROJECT_ROOT / f"img/{dataset}/time.pdf"
    img_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_file)
    plt.close(fig)


def plot_batch_recall_vs_time(dataset, batchsize, PROJECT_ROOT):
    recall_file = PROJECT_ROOT / f"result/{dataset}/pir.jsonl"
    batch_recall_file = PROJECT_ROOT / f"result/{dataset}/batch.jsonl"

    mean_recall = []
    std_recall = []
    mean_time = []
    std_time = []

    df = read_json(recall_file, lines=True)
    for beamw in np.arange(2, 20, 2):
        df_beamw = df[df["beamw"] == beamw]
        if df_beamw.empty:
            continue
        df_recall = df_beamw["recall@10"]
        df_time = df_beamw["time"] * 1000  # convert to milliseconds
        mean_recall.append(df_recall.mean())
        std_recall.append(df_recall.std())
        mean_time.append(df_time.mean())
        std_time.append(df_time.std())

    batch_mean_recall = []
    batch_std_recall = []
    batch_mean_time = []
    batch_std_time = []

    df = read_json(batch_recall_file, lines=True)
    df = df[df["batchsize"] == batchsize]
    for beamw in np.arange(2, 300, 2):
        df_beamw = df[df["beamw"] == beamw]
        if df_beamw.empty:
            continue
        df_recall = df_beamw["recall@10"]
        df_time = df_beamw["time"] * 1000  # convert to milliseconds
        batch_mean_recall.append(df_recall.mean())
        batch_std_recall.append(df_recall.std())
        batch_mean_time.append(df_time.mean() / batchsize)
        batch_std_time.append(df_time.std() / batchsize)

    fig, ax = plt.subplots(figsize=(3, 3), layout="constrained")

    ax.errorbar(
        batch_mean_time,
        batch_mean_recall,
        xerr=batch_std_time,
        yerr=batch_std_recall,
        marker="^",
        label="Batch",
        color="C2",
    )
    ax.errorbar(
        mean_time,
        mean_recall,
        xerr=std_time,
        yerr=std_recall,
        marker="o",
        label="Single",
        color="C0",
    )
    ax.set_xlabel("Total Server Time Per Query (ms)")
    ax.set_ylabel("10-Recall@10")
    ax.set_title(f"{dataset.title()}: batch size = {batchsize}")
    ax.grid(True)
    ax.legend()

    img_file = PROJECT_ROOT / f"img/{dataset}/batch_{batchsize}.pdf"
    img_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_file)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="movielens", type=str, help="Dataset name")
    parser.add_argument(
        "--batchsize", default=10, type=int, help="Batch size for search"
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parent.parent

    plot_recall_vs_time(args.dataset, PROJECT_ROOT)
    plot_batch_recall_vs_time(args.dataset, args.batchsize, PROJECT_ROOT)
