## Hierarchical Balanced Tree Indexing

### Reproduction

To reproduce results for the `movielens` dataset:

```bash
python reproduce.py --dataset movielens
```

Plots are saved in `../img/{dataset}/*.pdf`

---

### Build Index

```bash
python index.py --dataset movielens --n0 100 --branch 4
```

Indices are saved in `../dataset/{dataset}/indices/n0_{n0}_branch_{branch}.json`

| Parameter | Description | Values |
|---|---|---|
| `dataset` | Dataset name | String (e.g., `movielens`) |
| `n0` | Number of the root node | Integer (e.g., `100`) |
| `branch` | Tree branching factor | Integer (e.g., `2`) |

---

### SPRINT Simulation

```bash
python search.py --dataset movielens --beamwidth 3
```

Results are saved in `../result/{dataset}/recall.jsonl`

| Parameter | Description | Values |
|---|---|---|
| `dataset` | Dataset name | String (e.g., `movielens`) |
| `n0` | Number of the root node | Integer (e.g., `100`) |
| `branch` | Tree branching factor | Integer (e.g., `2`) |
| `beamw` | Beam width for search | Integer (e.g., `1`) |
| `cached` | Cached index depth | Integer (e.g., `1`) |

---

### Tiptoe Simulation

```bash
python tiptoe.py --dataset movielens
```
| Parameter | Description | Values |
|---|---|---|
| `dataset` | Dataset name | String (e.g., `movielens`) |
| `beamw` | Beam width for search | Integer (e.g., `1`) |

Indices are saved in `../dataset/{dataset}/indices/tiptoe.json`

Results are saved in `../result/{dataset}/tiptoe.jsonl`
