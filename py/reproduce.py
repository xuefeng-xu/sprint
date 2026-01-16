import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pandas import read_json
from argparse import ArgumentParser
from py.search import simulate_search
from py.tiptoe import simulate_tiptoe


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="movielens", type=str, help="Dataset name")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parent.parent
    recall_file = PROJECT_ROOT / f"result/{args.dataset}/recall.jsonl"
    if recall_file.exists():
        df = read_json(recall_file, lines=True)

    #######################################################################################

    # Heatmap
    N0 = [10, 30, 100, 300]
    BRANCH = [2, 4, 8, 16]
    BEAMW = [3, 10, 30]
    cached = 1

    for n0 in N0:
        for branch in BRANCH:
            for beamw in BEAMW:
                if "df" not in locals() or (
                    "df" in locals()
                    and df[
                        (df["n0"] == n0)
                        & (df["branch"] == branch)
                        & (df["beamw"] == beamw)
                        & (df["cached"] == cached)
                    ].empty
                ):
                    simulate_search(
                        dataset=args.dataset,
                        n0=n0,
                        branch=branch,
                        beamw=beamw,
                        cached=cached,
                    )

    df = read_json(recall_file, lines=True)
    recall_data = np.zeros((len(BEAMW), len(N0), len(BRANCH)))
    retrieve_data = np.zeros((len(BEAMW), len(N0), len(BRANCH)))

    fig_hm, ax_hm = plt.subplots(2, len(BEAMW), figsize=(9.5, 6), layout="constrained")
    for i, beamw in enumerate(BEAMW):
        for j, n0 in enumerate(N0):
            for k, branch in enumerate(BRANCH):
                df_cell = df[
                    (df["n0"] == n0)
                    & (df["branch"] == branch)
                    & (df["beamw"] == beamw)
                    & (df["cached"] == cached)
                ]
                recall_data[i, j, k] = df_cell["recall@10"].values[0][-1]
                retrieve_data[i, j, k] = df_cell["retrieve"].values[0][-1] * 100

    vmin_recall, vmax_recall = recall_data.min(), recall_data.max()
    vmin_retrieve, vmax_retrieve = retrieve_data.min(), retrieve_data.max()
    thre_recall = (vmin_recall + vmax_recall) / 2
    thre_retrieve = (vmin_retrieve + vmax_retrieve) / 2

    for i, beamw in enumerate(BEAMW):
        ax_hm[0, i].imshow(
            recall_data[i], vmin=vmin_recall, vmax=vmax_recall, cmap="YlGn"
        )
        ax_hm[1, i].imshow(
            retrieve_data[i], vmin=vmin_retrieve, vmax=vmax_retrieve, cmap="YlGn_r"
        )

        ax_hm[0, i].set_xticks(range(len(BRANCH)))
        ax_hm[0, i].set_yticks(range(len(N0)))
        ax_hm[0, i].set_xticklabels(BRANCH)
        ax_hm[0, i].set_yticklabels(N0)
        ax_hm[0, i].set_xlabel("Branching Factor")
        ax_hm[0, i].set_ylabel("n0")
        ax_hm[0, i].set_title(f"10-Recall@10 (Beam Width={beamw})")

        ax_hm[1, i].set_xticks(range(len(BRANCH)))
        ax_hm[1, i].set_yticks(range(len(N0)))
        ax_hm[1, i].set_xticklabels(BRANCH)
        ax_hm[1, i].set_yticklabels(N0)
        ax_hm[1, i].set_xlabel("Branching Factor")
        ax_hm[1, i].set_ylabel("n0")
        ax_hm[1, i].set_title(f"Retrieve (%) (Beam Width={beamw})")

        for j in range(len(N0)):
            for k in range(len(BRANCH)):
                ax_hm[0, i].text(
                    k,
                    j,
                    f"{recall_data[i, j, k]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if recall_data[i, j, k] > thre_recall else "black",
                )
                ax_hm[1, i].text(
                    k,
                    j,
                    f"{retrieve_data[i, j, k]:.2f}%",
                    ha="center",
                    va="center",
                    color=(
                        "white" if retrieve_data[i, j, k] < thre_retrieve else "black"
                    ),
                )

    # Add colorbars for the two rows
    cbar_recall = fig_hm.colorbar(
        ax_hm[0, 0].images[0], ax=ax_hm[0, :], shrink=0.8, location="right"
    )
    cbar_recall.set_label("10-Recall@10 (higher is better)")

    cbar_retrieve = fig_hm.colorbar(
        ax_hm[1, 0].images[0], ax=ax_hm[1, :], shrink=0.8, location="right"
    )
    cbar_retrieve.set_label("Retrieve % (lower is better)")

    hm_file = PROJECT_ROOT / f"img/{args.dataset}/heatmap.pdf"
    hm_file.parent.mkdir(parents=True, exist_ok=True)
    fig_hm.savefig(hm_file)
    plt.close(fig_hm)

    #######################################################################################

    # Beam width analysis
    N0 = [10, 30, 100]
    BRANCH = [8, 4, 2]
    BEAMW = [3, 10, 20, 40, 60]
    cached = 1

    for n0 in N0:
        for branch in BRANCH:
            for beamw in BEAMW:
                if df[
                    (df["n0"] == n0)
                    & (df["branch"] == branch)
                    & (df["beamw"] == beamw)
                    & (df["cached"] == cached)
                ].empty:
                    simulate_search(
                        dataset=args.dataset,
                        n0=n0,
                        branch=branch,
                        beamw=beamw,
                        cached=cached,
                    )

    df = read_json(recall_file, lines=True)
    fig_bw, ax_bw = plt.subplots(1, len(N0), figsize=(9.5, 3), layout="constrained")
    for i, n0 in enumerate(N0):
        for ls, branch in zip(["-o", "--v", "-.d"], BRANCH):
            df_cell = df[
                (df["n0"] == n0)
                & (df["branch"] == branch)
                & (df["beamw"].isin(BEAMW))
                & (df["cached"] == cached)
            ].sort_values(by="beamw")

            recall_curve = df_cell["recall@10"].apply(lambda x: x[-1]).values
            ax_bw[i].plot(
                BEAMW,
                recall_curve,
                ls,
                label=f"branch={branch}",
            )

        ax_bw[i].set_xlabel("Beam Width")
        ax_bw[i].set_ylabel("10-Recall@10")
        ax_bw[i].set_title(f"n0={n0}")
        ax_bw[i].grid(True)
        ax_bw[i].legend()

    bw_file = PROJECT_ROOT / f"img/{args.dataset}/beamwidth.pdf"
    bw_file.parent.mkdir(parents=True, exist_ok=True)
    fig_bw.savefig(bw_file)
    plt.close(fig_bw)

    #######################################################################################

    # Progressive recall
    N0 = [10, 30, 100]
    branch = 2
    BEAMW = [30, 10, 3]
    cached = 1

    # Tiptoe
    tiptoe_recall = PROJECT_ROOT / f"result/{args.dataset}/tiptoe.jsonl"
    if tiptoe_recall.exists():
        df_tiptoe = read_json(tiptoe_recall, lines=True)

    if "df_tiptoe" not in locals() or (
        "df_tiptoe" in locals() and df_tiptoe[(df_tiptoe["beamw"] == 1)].empty
    ):
        simulate_tiptoe(args.dataset, beamw=1)

    df_tiptoe = read_json(tiptoe_recall, lines=True)
    tiptoe_recall_10 = df_tiptoe[(df_tiptoe["beamw"] == 1)]["recall@10"].values[0]

    fig_pr, ax_pr = plt.subplots(1, len(N0), figsize=(9.5, 3), layout="constrained")
    for i, n0 in enumerate(N0):
        for ls, beamw in zip(["-o", "--v", "-.d"], BEAMW):
            df_cell = df[
                (df["n0"] == n0)
                & (df["branch"] == branch)
                & (df["beamw"] == beamw)
                & (df["cached"] == cached)
            ]

            recall_curve = df_cell["recall@10"].values[0]
            ax_pr[i].plot(
                recall_curve,
                ls,
                label=f"BW={beamw}",
            )

        ax_pr[i].axhline(
            tiptoe_recall_10,
            linestyle="--",
            color="red",
            label="Tiptoe",
        )

        ax_pr[i].set_xlabel("Search Rounds")
        ax_pr[i].set_ylabel("10-Recall@10")
        ax_pr[i].set_title(f"n0={n0}, branch={branch}")
        ax_pr[i].grid(True)
        ax_pr[i].legend()

    pr_file = PROJECT_ROOT / f"img/{args.dataset}/progress.pdf"
    pr_file.parent.mkdir(parents=True, exist_ok=True)
    fig_pr.savefig(pr_file)
    plt.close(fig_pr)

    #######################################################################################

    # Cache depth analysis
    n0 = 10
    branch = 2
    BEAMW = [3, 10, 30]
    CACHED = [3, 2, 1]

    for cached in CACHED:
        for beamw in BEAMW:
            if df[
                (df["n0"] == n0)
                & (df["branch"] == branch)
                & (df["beamw"] == beamw)
                & (df["cached"] == cached)
            ].empty:
                simulate_search(
                    dataset=args.dataset,
                    n0=n0,
                    branch=branch,
                    beamw=beamw,
                    cached=cached,
                )

    df = read_json(recall_file, lines=True)
    fig_cd, ax_cd = plt.subplots(1, len(BEAMW), figsize=(9.5, 3), layout="constrained")
    for i, beamw in enumerate(BEAMW):
        for ls, cached in zip(["-o", "--v", "-.d"], CACHED):
            df_cell = df[
                (df["n0"] == n0)
                & (df["branch"] == branch)
                & (df["beamw"] == beamw)
                & (df["cached"] == cached)
            ]

            recall_curve = df_cell["recall@10"].values[0]
            ax_cd[i].plot(
                recall_curve,
                ls,
                label=f"cache={cached}",
            )

        ax_cd[i].set_xlabel("Search Rounds")
        ax_cd[i].set_ylabel("10-Recall@10")
        ax_cd[i].set_title(f"n0={n0}, branch={branch}, BW={beamw}")
        ax_cd[i].grid(True)
        ax_cd[i].legend()

    cd_file = PROJECT_ROOT / f"img/{args.dataset}/cache.pdf"
    cd_file.parent.mkdir(parents=True, exist_ok=True)
    fig_cd.savefig(cd_file)
    plt.close(fig_cd)
