import json
import numpy as np
from pathlib import Path
from py.index import get_common_parser, build_index
from py.utils import (
    load_base,
    load_query,
    load_groundtruth,
    mean_n_recall_at_k,
    TopK,
)


def save_results(
    dataset, n0, branch, beamw, cached, recall_1, recall_10, recall_100, retrieve
):
    PROJECT_ROOT = Path(__file__).parent.parent
    file = PROJECT_ROOT / f"result/{dataset}/recall.jsonl"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)

    res = {
        "n0": n0,
        "branch": branch,
        "beamw": beamw,
        "cached": cached,
        "recall@1": recall_1,
        "recall@10": recall_10,
        "recall@100": recall_100,
        "retrieve": retrieve,
    }

    with open(file, "a") as f:
        f.write(json.dumps(res) + "\n")


def search_per_query(xq, xb, indices, beamw=1, cached=1, k=100):
    topk_cand = TopK(k)

    result = []
    for i, indices_i in enumerate(indices):
        indices_i = np.asarray(indices_i)
        xb_lvl_i = xb[indices_i]

        if i < cached:
            # Use cached full level search
            dist_i = xb_lvl_i.dot(xq)
            idx_i = np.argsort(dist_i)[::-1]
            dist_i = dist_i[idx_i]

        else:
            # Beam search
            branch = len(indices_i) // len(indices[i - 1])

            # Indices of candidates
            idx_i = [np.arange(j * branch, (j + 1) * branch) for j in cand_i]
            idx_i = np.asarray(idx_i).flatten()

            xb_i = xb_lvl_i[idx_i]
            dist_i = xb_i.dot(xq)

            idx_sort = np.argsort(dist_i)[::-1]
            dist_i = dist_i[idx_sort]
            idx_i = idx_i[idx_sort]

        # Store all candidates
        for dist, idx in zip(dist_i, idx_i):
            # Map back to original indices
            topk_cand.insert(dist, indices_i[idx])

        # Save results for evaluation
        if i >= cached - 1:
            result.append(topk_cand.get_topk_indices())

        # Skip if at leaf level
        if i < len(indices) - 1:
            # Candidates to search at next level
            if idx_i.size > beamw:
                cand_i = idx_i[:beamw]
            else:
                cand_i = idx_i

    return result


def search_all_queries(xq, xb, indices, beamw=1, cached=1, k=100):
    height = len(indices)
    n_rounds = height - cached + 1

    results = [[] for _ in range(n_rounds)]
    for xq_i in xq:
        res = search_per_query(xq_i, xb, indices, beamw, cached, k)
        for round in range(n_rounds):
            results[round].append(res[round])

    return results


def retrieve_percent(indices, beamw=1, cached=1):
    percent = []
    count = 0
    n = 0
    for i in range(len(indices)):
        n_nodes = len(indices[i])
        n += n_nodes

        if i < cached:
            count += n_nodes
        else:
            percent.append(count)
            n_node_prev = len(indices[i - 1])
            branch = n_nodes // n_node_prev
            count += branch * min(beamw, n_node_prev)
    percent.append(count)
    return (np.asarray(percent) / n).tolist()


def simulate_search(dataset, n0, branch, beamw, cached):
    xb = load_base(dataset)
    nb, d = xb.shape
    xq = load_query(dataset)
    nq, _ = xq.shape
    gt = load_groundtruth(dataset)

    indices = build_index(dataset, xb, n0, branch)
    results = search_all_queries(xq, xb, indices, beamw, cached)

    height = len(indices)
    recall_1, recall_10, recall_100 = [], [], []
    for round in range(height - cached + 1):
        results_i = results[round]
        recall_1.append(mean_n_recall_at_k(results_i, gt, n=1, k=1))
        recall_10.append(mean_n_recall_at_k(results_i, gt, n=10, k=10))
        recall_100.append(mean_n_recall_at_k(results_i, gt, n=100, k=100))

    retrieve = retrieve_percent(indices, beamw, cached)
    save_results(
        dataset,
        n0,
        branch,
        beamw,
        cached,
        recall_1,
        recall_10,
        recall_100,
        retrieve,
    )

    return nb, nq, d, recall_1[-1], recall_10[-1], recall_100[-1], retrieve[-1]


if __name__ == "__main__":
    parser = get_common_parser()
    parser.add_argument(
        "--beamw",
        default=1,
        type=int,
        help="Beam width for search",
    )
    parser.add_argument(
        "--cached",
        default=1,
        type=int,
        help="Cached index depth",
    )
    args = parser.parse_args()

    nb, nq, d, recall_1, recall_10, recall_100, retrieve = simulate_search(
        args.dataset,
        args.n0,
        args.branch,
        args.beamw,
        args.cached,
    )

    print(f"{args.dataset}: dim={d}, nb={nb}, nq={nq}")
    print(
        f"n0={args.n0}, branch={args.branch}, beamw={args.beamw}, cached={args.cached}"
    )
    print(
        f"Recall@1={recall_1:.2f}, Recall@10={recall_10:.2f}, Recall@100={recall_100:.2f}, Retrieve={retrieve * 100:.2f}%"
    )
