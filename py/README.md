## Hierarchical IVF

### Build Index

```bash
python index.py --dataset movielens --n0 100 --branch 4
```

| Parameter | Description | Values |
|---|---|---|
| `dataset` | Dataset name | `movielens` |
| `n0` | Number of the root node | e.g., `100` |
| `branch` | Tree branching factor | e.g., `2` |
| `saveidx` | Whether to save the index | `0` or `1` |

### Search

```bash
python search.py --dataset movielens --beamwidth 3 --k 100
```

| Parameter | Description | Values |
|---|---|---|
| `dataset` | Dataset name | `movielens` |
| `n0` | Number of the root node | e.g., `100` |
| `branch` | Tree branching factor | e.g., `2` |
| `saveidx` | Whether to save the index | `0` or `1` |
| `k` | Number of nearest neighbors | e.g., `100` |
| `beamw` | Beamwidth for search | e.g., `1` |
