import json
import math
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from k_means_constrained import KMeansConstrained
from py.utils import (
    load_base,
    load_query,
    load_groundtruth,
    mean_n_recall_at_k,
    TopK,
    NpEncoder,
)


def balanced_clustering(x):
    n, _ = x.shape
    n_clusters = math.ceil(math.sqrt(n))

    kmeans = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=n // n_clusters,
        size_max=math.ceil(n / n_clusters),
        random_state=0,
        n_jobs=-1,  # Adjust to reduce memory usage
    )
    kmeans.fit(x)

    centroids = kmeans.cluster_centers_
    bincount = np.bincount(kmeans.labels_)
    maxcount = np.max(bincount)

    indices = []
    for i in range(n_clusters):
        mask = kmeans.labels_ == i

        if bincount[i] < maxcount:
            diff = int(maxcount - bincount[i])

            # Find the closest vectors from other clusters
            dist_other_i = x[~mask].dot(centroids[i])
            supp_idx = np.argpartition(dist_other_i, -diff)[-diff:]

            # Add them to this cluster to make it balanced
            supp_idx = np.flatnonzero(~mask)[supp_idx]
            mask[supp_idx] = True

        indices.append(np.flatnonzero(mask))

    return centroids, indices


def indices_path(dataset):
    PROJECT_ROOT = Path(__file__).parent.parent
    indices_dir = PROJECT_ROOT / f"dataset/{dataset}/indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    return indices_dir / f"tiptoe.json"


def load_indices(dataset):
    indices_file = indices_path(dataset)

    if not indices_file.exists():
        raise FileNotFoundError(f"Indices file not found: {indices_file}")

    with open(indices_file, "r") as f:
        indices = json.load(f)
    return indices["centroids"], indices["indices"]


def save_indices(centroids, indices, dataset):
    indices_file = indices_path(dataset)
    indices_file.parent.mkdir(parents=True, exist_ok=True)

    with open(indices_file, "w") as f:
        json.dump({"centroids": centroids, "indices": indices}, f, cls=NpEncoder)


def build_index(dataset, xb):
    try:
        centroids, indices = load_indices(dataset)
    except FileNotFoundError:
        centroids, indices = balanced_clustering(xb)
        save_indices(centroids, indices, dataset)
    return np.asarray(centroids), indices


def search_per_query(xq, xb, centroids, indices, beamw=1, k=100):
    topk_cand = TopK(k)

    # Find closest centroids
    dist_cent = centroids.dot(xq)
    beamw = min(beamw, len(centroids))
    cent_idx = np.argpartition(dist_cent, -beamw)[-beamw:]

    for c_idx in cent_idx:
        dist_vec = xb[indices[c_idx]].dot(xq)
        for dist, idx in zip(dist_vec, indices[c_idx]):
            topk_cand.insert(dist, idx)

    return topk_cand.get_topk_indices()


def search_all_queries(xq, xb, centroids, indices, beamw=1, k=100):
    results = []
    for xq_i in xq:
        res = search_per_query(xq_i, xb, centroids, indices, beamw, k)
        results.append(res)
    return results


def save_results(dataset, beamw, recall_1, recall_10, recall_100, retrieve):
    PROJECT_ROOT = Path(__file__).parent.parent
    file = PROJECT_ROOT / f"result/{dataset}/tiptoe.jsonl"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)

    res = {
        "beamw": beamw,
        "recall@1": recall_1,
        "recall@10": recall_10,
        "recall@100": recall_100,
        "retrieve": retrieve,
    }

    with open(file, "a") as f:
        f.write(json.dumps(res) + "\n")


def simulate_tiptoe(dataset, beamw=1):
    xb = load_base(dataset)
    nb, d = xb.shape
    xq = load_query(dataset)
    nq, _ = xq.shape
    gt = load_groundtruth(dataset)

    centroids, indices = build_index(dataset, xb)
    results = search_all_queries(xq, xb, centroids, indices, beamw)

    recall_1 = mean_n_recall_at_k(results, gt, n=1, k=1)
    recall_10 = mean_n_recall_at_k(results, gt, n=10, k=10)
    recall_100 = mean_n_recall_at_k(results, gt, n=100, k=100)

    retrieve = len(indices[0]) * beamw / nb

    save_results(
        dataset,
        beamw,
        recall_1,
        recall_10,
        recall_100,
        retrieve,
    )

    return nb, nq, d, recall_1, recall_10, recall_100


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="movielens", type=str, help="Dataset name")
    parser.add_argument(
        "--beamw",
        default=1,
        type=int,
        help="Beam width for search",
    )
    args = parser.parse_args()

    nb, nq, d, recall_1, recall_10, recall_100 = simulate_tiptoe(
        args.dataset,
        args.beamw,
    )

    print(f"{args.dataset}: dim={d}, nb={nb}, nq={nq}")
    print(
        f"beamw={args.beamw}: Recall@1={recall_1:.2f}, Recall@10={recall_10:.2f}, Recall@100={recall_100:.2f}"
    )
