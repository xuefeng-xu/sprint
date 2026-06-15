## Private Vector Retrieval

### Reproduction

To reproduce results for the `movielens` dataset:

```bash
sh reproduce.sh movielens 10
```

Plots are saved in `../img/{dataset}/{time,batch_10}.pdf`

---

### SPRINT with PIR

```bash
go run search.go -dataset movielens
```

| Parameter | Description | Values |
|---|---|---|
| `dataset` | Dataset name | String (e.g., `movielens`) |
| `n0` | Number of the root node | Integer (e.g., `100`) |
| `branch` | Tree branching factor | Integer (e.g., `2`) |
| `beamw` | Beam width for search | Integer (e.g., `1`) |
| `cached` | Cached index depth | Integer (e.g., `1`) |

Results are saved in `../result/{dataset}/pir.jsonl`

---

### Dataset Preprocessing

We use Inner Product as the similarity metric. However, [Pacmann](https://github.com/wuwuz/Pacmann) uses Euclidean distance, therefore we need to preprocess the dataset before running Pacmann.

```bash
python preprocess.py --dataset movielens
```

Preprocessed datasets are saved as `ip_{dataset}_base.fvecs` and `ip_{dataset}_query.fvecs` in `../dataset/{dataset}/`

---

### Batch Retrieval

```bash
go run batch/main.go -dataset movielens -batchsize 10
```

| Parameter | Description | Values |
|---|---|---|
| `dataset` | Dataset name | String (e.g., `movielens`) |
| `n0` | Number of the root node | Integer (e.g., `100`) |
| `branch` | Tree branching factor | Integer (e.g., `2`) |
| `beamw` | Beam width for search | Integer (e.g., `1`) |
| `cached` | Cached index depth | Integer (e.g., `1`) |
| `batchsize` | Batch size for retrieval | Integer (e.g., `10`) |

Results are saved in `../result/{dataset}/batch.jsonl`
