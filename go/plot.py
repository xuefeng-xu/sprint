import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pandas import read_json
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="movielens", type=str, help="Dataset name")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parent.parent
    recall_file = PROJECT_ROOT / f"result/{args.dataset}/pir.jsonl"

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
    ax.set_title(f"{args.dataset.title()}: Recall vs Time")
    ax.grid(True)
    ax.legend()

    img_file = PROJECT_ROOT / f"img/{args.dataset}/time.pdf"
    img_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_file)
    plt.close(fig)
