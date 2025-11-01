import numpy as np
from pathlib import Path
from index import get_common_parser
from hbkt import hier_balanced_kmeans
from utils import (
    load_base,
    load_query,
    load_groundtruth,
    save_indices,
    load_indices,
    mean_recall_at_k,
    TopK,
)


def save_results(dataset, n0, branch, k, beamw, recall):
    PROJECT_ROOT = Path(__file__).parent.parent
    file = PROJECT_ROOT / f"result/{dataset}/recall.txt"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w") as f:
            f.write("n0,branch,k,beamw,recall\n")

    with file.open("a") as f:
        f.write(f"{n0},{branch},{k},{beamw},{recall}\n")


def search_per_query(xq, xb, indices, beamw=1, k=100):
    topk_cand = TopK(k)
    height = len(indices) - 1

    for i, indices_i in enumerate(indices):
        indices_i = np.asarray(indices_i)
        xb_i = xb[indices_i]

        if i == 0:
            dist_i = xb_i.dot(xq)
            idx_i = np.argsort(dist_i)[::-1]
            dist_i = dist_i[idx_i]

        else:
            branch = len(indices_i) // len(indices[i - 1])

            idx_i = [np.arange(j * branch, (j + 1) * branch) for j in cand_i]
            idx_i = np.asarray(idx_i).flatten()

            xb_i = xb_i[idx_i]
            dist_i = xb_i.dot(xq)
            idx_sort = np.argsort(dist_i)[::-1]
            dist_i = dist_i[idx_sort]
            idx_i = idx_i[idx_sort]

        # Store all candidates
        for dist, idx in zip(dist_i, idx_i):
            # Map back to original indices
            topk_cand.insert(dist, indices_i[idx])

        if i == height:
            continue

        # Candidates to search at next level
        if idx_i.size > beamw:
            cand_i = idx_i[:beamw]
        else:
            cand_i = idx_i

    return topk_cand.get_topk_indices()


def search_all_queries(xq, xb, indices, beamw=1, k=100):
    results = []
    for xq_i in xq:
        res = search_per_query(xq_i, xb, indices, beamw, k)
        res = np.asarray(res)
        results.append(res)

    return results


if __name__ == "__main__":
    parser = get_common_parser()
    parser.add_argument(
        "--k",
        default=100,
        type=int,
        help="Number of nearest neighbors",
    )
    parser.add_argument(
        "--beamw",
        default=1,
        type=int,
        help="Beamwidth for search",
    )
    args = parser.parse_args()

    xb = load_base(args.dataset)
    nb, d = xb.shape
    xq = load_query(args.dataset)
    nq, _ = xq.shape
    gt = load_groundtruth(args.dataset, args.k)

    print(f"{args.dataset}: nb {nb}, nq {nq}, dim {d}")
    print(f"n0 {args.n0}, branch {args.branch}, beamw {args.beamw}")

    try:
        indices = load_indices(args.dataset, args.n0, args.branch)
    except FileNotFoundError:
        indices = hier_balanced_kmeans(xb, args.n0, args.branch)
        if args.saveidx:
            save_indices(indices, args.dataset, args.n0, args.branch)

    results = search_all_queries(xq, xb, indices, args.beamw, args.k)
    recall = mean_recall_at_k(results, gt, args.k)
    print(f"Recall @ {args.k}: {recall}")

    save_results(args.dataset, args.n0, args.branch, args.k, args.beamw, recall)
