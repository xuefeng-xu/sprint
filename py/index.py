from argparse import ArgumentParser
from py.utils import load_base, save_indices, load_indices, hier_balanced_kmeans


def get_common_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="movielens", type=str, help="Dataset name")
    parser.add_argument("--n0", default=100, type=int, help="Number of the root node")
    parser.add_argument("--branch", default=2, type=int, help="Tree branching factor")
    return parser


def build_index(dataset, xb, n0, branch):
    try:
        indices = load_indices(dataset, n0, branch)
    except FileNotFoundError:
        indices = hier_balanced_kmeans(xb, n0, branch)
        save_indices(indices, dataset, n0, branch)
    return indices


if __name__ == "__main__":
    parser = get_common_parser()
    args = parser.parse_args()

    xb = load_base(args.dataset)
    nb, d = xb.shape
    print(f"{args.dataset}: dim={d}, nb={nb}")

    indices = build_index(args.dataset, xb, args.n0, args.branch)

    height = len(indices) - 1
    total_nodes = 0
    for i in range(height + 1):
        n_nodes = len(indices[i])
        total_nodes += n_nodes
        if i == 0:
            print(f"Level {i}: #nodes={n_nodes}")
        else:
            print(
                f"Level {i}: #nodes={n_nodes}, branch={n_nodes // len(indices[i - 1])}"
            )
    print(f"Total #nodes: {total_nodes}")
