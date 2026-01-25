import numpy as np
from argparse import ArgumentParser
from faiss.contrib.vecs_io import fvecs_write
from py.utils.dataloader import load_base, load_query, dataset_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="movielens", type=str, help="Dataset name")
    args = parser.parse_args()

    xq = load_query(args.dataset)
    xb = load_base(args.dataset)

    nb, d = xb.shape
    nq, _ = xq.shape

    # Preprocess: index metric is Euclidean (L2), wanted metric is Inner Product (IP)
    # Reference: Table 2 in https://arxiv.org/pdf/2401.08281v4

    # Query: x -> [x, 0]
    xq_ip = np.hstack([xq, np.zeros((nq, 1))])

    # Base: y -> [y, sqrt(alpha - ||y||^2)]
    l2normsq = np.sum(xb**2, axis=1)
    alpha = np.max(l2normsq) + 1
    xb_ip = np.hstack([xb, np.sqrt(alpha - l2normsq).reshape(-1, 1)])

    # Pacmann seems require even dimensions, so pad if necessary
    if d % 2 == 1:
        xq_ip = np.hstack([xq_ip, np.zeros((nq, 1))])
        xb_ip = np.hstack([xb_ip, np.zeros((nb, 1))])

    dataset_path = dataset_path(args.dataset)
    fvecs_write(dataset_path / f"ip_{args.dataset}_query.fvecs", xq_ip)
    fvecs_write(dataset_path / f"ip_{args.dataset}_base.fvecs", xb_ip)
