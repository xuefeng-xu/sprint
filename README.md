# SPRINT

## Installation

#### 1. Clone this repository:

```bash
git clone https://github.com/xuefeng-xu/sprint.git && cd sprint
```

#### 2. Python Setup

2.i. Create a Python environment:

```bash
conda create -n sprint python=3.11
conda activate sprint
```

2.ii. Install dependencies:

```bash
pip install -e .
```

#### 3. C++ Setup

3.i. Prerequisites:

- C++17
- [Bazel](https://bazel.build/install)

3.ii. Build the project:

```bash
bazel build //cpp:all
```

## Reproduction

Dataset http://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html

Plots are saved under `./img/*.pdf`
