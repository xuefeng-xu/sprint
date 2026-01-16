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

## System Requirements

We recommend using a Linux system with at least 128GB of memory to run experiments.

## Reproduction

For Python experiments, see [py/README.md](py/README.md).

Datasets are automatically downloaded from https://huggingface.co/datasets/vecdb/vector

Plots are saved in `./img/*.pdf`
