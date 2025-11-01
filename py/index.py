from argparse import ArgumentParser
from hbkt import hier_balanced_kmeans
from utils import load_base, save_indices


def get_common_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="movielens", type=str, help="Dataset name")
    parser.add_argument("--n0", default=100, type=int, help="Number of the root node")
    parser.add_argument("--branch", default=2, type=int, help="Tree branching factor")
    parser.add_argument(
        "--saveidx",
        default=1,
        type=int,
        choices=[0, 1],
        help="Whether to save the indices",
    )
    return parser


if __name__ == "__main__":
    parser = get_common_parser()
    args = parser.parse_args()

    xb = load_base(args.dataset)
    nb, d = xb.shape
    print(f"{args.dataset}: nb {nb}, dim {d}")

    indices = hier_balanced_kmeans(xb, args.n0, args.branch)

    height = len(indices) - 1
    for i in range(height + 1):
        n_nodes = len(indices[i])
        if i < height:
            print(f"Level {i}: {n_nodes} nodes, branch {len(indices[i+1]) // n_nodes}")
        else:
            print(f"Level {i}: {n_nodes} nodes")

    if args.saveidx:
        save_indices(indices, args.dataset, args.n0, args.branch)
