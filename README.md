# SPRINT: Scalable Private Retrieval with Indexed Neighbor Trees

## Installation

#### 1. Clone this repository:

```bash
git clone https://github.com/xuefeng-xu/sprint.git && cd sprint
```

#### 2. Python Setup:

2.i. Create a Python environment:

```bash
conda create -n sprint python=3.11
conda activate sprint
```

2.ii. Install dependencies:

```bash
pip install -e .
```

#### 3. Go Setup:

3.i. Prerequisites:

Install [Go](https://golang.org/dl/) (1.22.1 or higher)

3.ii. Download [pianopir](https://github.com/wuwuz/Pacmann/tree/5dc056d75ceaeeedba8cbbd6b0b1c0a00a78b674/pianopir) folder and put it at `./go/pianopir`

3.iii. Download dependencies:

```bash
go mod tidy
```

## System Requirements

We recommend using a Linux system with at least 128GB of memory to run experiments.

## Reproduction

For Python experiments, see [py/README.md](py/README.md).

For Go experiments, see [go/README.md](go/README.md).

Datasets are automatically downloaded from https://huggingface.co/datasets/vecdb/vector and stored in `./dataset/{dataset}/*`.

Plots are saved in `./img/*.pdf`
