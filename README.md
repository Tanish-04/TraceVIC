# TraceVIC: Causal Reasoning over Code Evolution for Identifying Vulnerability-Inducing Commits

> **TraceVIC** identifies the commit that introduced a vulnerability in C/C++
> projects, given the commit that patched it, by reasoning over how the affected
> code evolved across its commit history.

This repository contains the complete implementation accompanying the paper:
graph construction, the two-phase training pipeline, and all baseline methods.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Contributions](#key-contributions)
3. [Repository Structure](#repository-structure)
4. [Dataset](#dataset)
5. [Environment Setup](#environment-setup)
6. [End-to-End Pipeline](#end-to-end-pipeline)
7. [The Three Arms](#the-three-arms)
8. [Baselines](#baselines)
9. [Configuration Reference](#configuration-reference)
10. [Acknowledgments](#acknowledgments)

---

## Overview

Traditional SZZ-based approaches trace vulnerability-inducing commits (VICs) with
a line-level `git blame` from the fixing commit. That works for simple cases but
struggles when the vulnerability's history spans many commits over long time
horizons.

TraceVIC encodes both the **structural** (CFG, DFG) and **temporal** relationships
between code changes into a heterogeneous graph, then learns to rank candidate
commits with a two-phase architecture:

```
Fixing Commit
      │
      ▼
┌─────────────────────────────────────────┐
│  Phase 1: Root-Cause Line Identification│
│  ─────────────────────────────────────  │
│  GAT encoder over a heterogeneous code  │
│  graph (CFG + DFG + LINEMAP + temporal) │
│  + RankNet pairwise scoring head        │
│  → surfaces the deletion lines most     │
│    likely to be the root cause          │
└────────────────┬────────────────────────┘
                 │ top-k deletion lines
                 ▼
┌─────────────────────────────────────────┐
│  Phase 2: VIC Ranking                   │
│  ─────────────────────────────────────  │
│  Commit Transformer + dual-query        │
│  cross-attention over the pooled        │
│  candidate commits of those lines       │
│  → ranks them to find the VIC           │
└─────────────────────────────────────────┘
```

Temporal edges encode commit ordering between code changes, letting the model
reason about *when* a line was introduced relative to other changes — a signal
purely structural approaches miss.

---

## Key Contributions

- **Temporal code graphs** — a graph representation augmenting CFG/DFG edges with
  forward and backward temporal edges over each line's commit history.
- **Two-phase architecture** — Phase 1 narrows the search space to the most
  suspicious deletion lines; Phase 2 ranks the commits those lines blame to.
- **Comprehensive evaluation** — 755 Linux kernel CVEs with ground-truth
  inducing commits, under 5-fold cross-validation, against **eight** SZZ
  variants and NeuralSZZ.

---

## Repository Structure

```text
TraceVIC/
├── README.md                        ← this file
├── environment.yml                  ← conda environment specification
│
├── graph_construction/              ← Stage 1: build heterogeneous code graphs
│   ├── README.md                    ←   detailed 7-step pipeline documentation
│   ├── config.yaml                  ←   projects, tool paths, settings
│   ├── extract_source_files.py      ←   Steps 1 & 5: check out before/after sources
│   ├── generate_joern.py            ←   Steps 2 & 6: Joern CPG generation
│   ├── generate_graphs.py           ←   Steps 3 & 7: heterogeneous graph assembly
│   ├── history_commits_retrieval.py ←   Step 4: V-SZZ / B-SZZ history tracing
│   ├── create_commits_json.py       ←   Step 4b: per-case commit chains
│   ├── sem_szz_analysis.py          ←   addition-only fixes: anchor-line derivation
│   ├── gen_finalgraph/ parse_patch/ parse_joerndot/ mapping/ project/ app/
│   └── data/                        ←   dataset (distributed via OSF, see below)
│
├── training_pipeline/               ← Stages 2–4: pre-build, train, evaluate
│   ├── README.md                    ←   training documentation
│   ├── training_config.yaml         ←   every hyper-parameter, path and switch
│   ├── build_graphs.py              ←   Stage 2: PyG-compatible graph pre-building
│   ├── main_kfold.py                ←   Stage 3: k-fold CV training (Phase 1 → 2)
│   ├── run_evaluation.py            ←   Stage 4: P/R/F1 tables, external datasets
│   ├── data_processing/ models/ training/
│   └── temporal_graph/              ←   pre-built graphs (regenerate, see Stage 2)
│
└── baselines/
    ├── README.md                    ←   comparability argument across all arms
    ├── SZZ/                         ←   B-SZZ, AG-SZZ, MA-SZZ, V-SZZ + results
    ├── SZZ_remaining/               ←   R-SZZ, L-SZZ, RA-SZZ, TSE-SZZ drivers + results
    └── neuralszz/                   ←   NeuralSZZ (HAN + RankNet)
```

> Each subdirectory has its own `README.md` with detailed instructions.

---

## Dataset

**755 verified Linux kernel CVEs** with ground-truth vulnerability-inducing
commits (787 GT commits in total), plus **79 cases** across four additional
projects used in the generalizability evaluation.

Each test case contains:

| File | Description |
|---|---|
| `info.json` | CVE id, fixing commit SHA, ground-truth inducing commit SHA(s) |
| `<fix_sha>/graph.json` | heterogeneous code-change graph for the fixing commit |
| `<fix_sha>/graph_vszz.json` | V-SZZ history chains per deletion line |
| `commits.json` | the case's full historical commit chain |
| `<history_sha>/graph.json` | graph for each history commit |

> **⬇ Download the dataset (OSF):** **https://osf.io/qfkjm**

Extract it so that `graph_construction/data/` contains one directory per project.
It is ~3.5 GB and is **not** tracked in git.

Everything downstream of `data/` is regenerated locally rather than downloaded —
see [Stage 2](#stage-2-graph-pre-building).

### Supported projects

| Project | Language | Role | Cases |
|---|---|---|---|
| Linux kernel | C | primary evaluation | 755 |
| FFmpeg | C | generalizability | 20 |
| OpenSSL | C | generalizability | 20 |
| ImageMagick | C | generalizability | 20 |
| PHP-SRC | C | generalizability | 19 |

---

## Environment Setup

| Requirement | Details |
|---|---|
| OS | Linux (tested on Ubuntu 22.04) |
| GPU | CUDA-compatible, ≥16 GB VRAM (Phase 1 peaks ~13–15 GB) |
| Java | **21** — required by Joern and GumTree |
| Tools | [Joern](https://joern.io/), [GumTree](https://github.com/GumTreeDiff/gumtree), [Bear](https://github.com/rizsotto/Bear) |

```bash
git clone <repo-url> TraceVIC
cd TraceVIC

conda env create -f environment.yml
conda activate tracevic

python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"
python -c "import torch_geometric; print('PyG', torch_geometric.__version__)"
```

The environment ships its own JDK 21, so `conda activate tracevic` puts a
suitable `java` on `PATH`. A UTF-8 locale is required or Joern aborts at startup:

```bash
export LANG=C.UTF-8 LC_ALL=C.UTF-8
```

> Tool installation and the `GRAPHGEN_*` environment variables are documented in
> [`graph_construction/README.md`](graph_construction/README.md#environment-setup).

---

## End-to-End Pipeline

```
Stage 1                 Stage 2               Stage 3              Stage 4
Graph Construction  →   Graph Pre-Building →  Training         →   Evaluation
(run once, or          (build_graphs.py)     (main_kfold.py)      (run_evaluation.py)
 download the data)
```

**If you downloaded the dataset from OSF, skip Stage 1 entirely** and start at
Stage 2.

### Stage 1: Graph Construction

The most time-intensive stage; only needed if building the dataset from scratch.

```bash
cd graph_construction

python extract_source_files.py     --mode fix     --projects linux   # Step 1
python generate_joern.py                          --projects linux   # Step 2
python generate_graphs.py          --mode fix     --projects linux   # Step 3
python history_commits_retrieval.py               --projects linux   # Step 4
python create_commits_json.py                     --projects linux   # Step 4b
python extract_source_files.py     --mode history --projects linux   # Step 5
python generate_joern.py                          --projects linux   # Step 6
python generate_graphs.py          --mode history --projects linux   # Step 7
```

> Full details, prerequisites and troubleshooting:
> [`graph_construction/README.md`](graph_construction/README.md#pipeline-steps)

### Stage 2: Graph Pre-Building

Converts the raw JSON graphs into PyG-compatible per-deletion-line structures.
**Required after downloading the dataset** — these are not distributed, because
they rebuild in well under a minute.

```bash
cd training_pipeline

python build_graphs.py --mode all                # full_graph + no_temporal (~30 s)
python build_graphs.py --mode bszz_candidates    # the B-SZZ candidate arm (~10 s)
```

### Stage 3: Training

5-fold cross-validation; within each fold Phase 1 trains, is evaluated, its
embeddings are cached, and Phase 2 trains on top.

```bash
cd training_pipeline
python main_kfold.py --graph-mode full_graph --save-dir checkpoints_GAT_Temporal
```

See [The Three Arms](#the-three-arms) for the exact commands.

### Stage 4: Evaluation

```bash
cd training_pipeline

python run_evaluation.py                                  # P/R/F1 tables, all arms
python run_evaluation.py --arm gat_temporal --k-lines 3   # one arm

# generalizability: a trained model on the non-Linux projects
python run_evaluation.py --arm gat_temporal \
    --data-root    ../graph_construction/data/FFmpeg ../graph_construction/data/OpenSSL \
    --prebuilt-dir temporal_graph_gen/FFmpeg          temporal_graph_gen/OpenSSL
```

> **Checkpoints are not distributed** (~39 GB). `run_evaluation.py` reads each
> fold's saved weights, so train an arm first. Point it at a non-default
> directory with `--ckpt-dir gat_temporal=<dir>`.

---

## The Three Arms

These are the three configurations reported in the paper. **Copy them verbatim** —
the ablation needs three flags set together, and omitting one silently trains a
configuration that is not in the paper.

**Arm 1 — GAT + temporal (main method)**
```bash
python main_kfold.py --graph-mode full_graph --save-dir checkpoints_755_GAT_Temporal
```

**Arm 2 — GAT − temporal (ablation)**
```bash
python main_kfold.py --graph-mode no_temporal \
    --no-model-use-temporal-pe \
    --save-dir checkpoints_755_GAT_NoTemporal
```

**Arm 3 — B-SZZ candidates (ablation)**
```bash
python build_graphs.py --mode bszz_candidates          # once
python main_kfold.py --graph-mode bszz_candidates --save-dir checkpoints_755_BSZZ_Candidates
```

| arm | what it isolates |
|---|---|
| GAT + temporal | the full method |
| GAT − temporal | does the *representation* of history help? Same candidates; temporal edges and positional encodings removed. |
| B-SZZ candidates | does the *full chain* help versus only the latest blame hop? Same model; different candidate set. |

---

## Baselines

All baselines are scored on the **same 755 cases**, against the **same 787
ground-truth commits**, with the **same micro-averaged metric** and `sha[:12]`
prefix matching as TraceVIC — see
[`baselines/README.md`](baselines/README.md#why-the-numbers-are-comparable).

| Method | Where | Notes |
|---|---|---|
| B-SZZ, AG-SZZ, MA-SZZ, V-SZZ | [`baselines/SZZ/`](baselines/SZZ/README.md) | implemented here; results shipped |
| R-SZZ, L-SZZ, RA-SZZ, TSE-SZZ | [`baselines/SZZ_remaining/`](baselines/SZZ_remaining/README.md) | run via an external replication package; drivers + results shipped |
| NeuralSZZ | [`baselines/neuralszz/`](baselines/neuralszz/README.md) | HAN + RankNet line ranking, then SZZ on the top-k lines |

```bash
cd baselines/SZZ
python evaluate.py --method b --time all755      # b | ag | ma | v
```

---

## Configuration Reference

### Graph construction — `graph_construction/config.yaml`

All paths are relative to `graph_construction/`.

| Key | Default | Description |
|---|---|---|
| `pipeline.repos_dir` | `repositories` | directory holding every clone |
| `projects.<name>.root` | `repositories/<name>` | that project's cloned git working tree — e.g. `repositories/linux`. Used as the base for resolving `compile_commands.json` include paths when Clang parses a source file. |
| `projects.<name>.compile_db` | `repositories/<name>/compile_commands.json` | compile database for that project |
| `vszz.pyszz_path` | `../baselines/SZZ` | SZZ implementations imported by Step 4 |
| `gumtree.java_home` | `/usr/lib/jvm/java-21-openjdk-amd64` | JDK 21 home — **no env-var override, no fallback** |

Environment variables: `GRAPHGEN_LIBCLANG`, `GRAPHGEN_PROJECT`,
`GRAPHGEN_COMPILE_DB` (all optional overrides).

### Training — `training_pipeline/training_config.yaml`

Every key can also be set on the command line; flags default to `None`, so
omitting one leaves the YAML value in effect (`python main_kfold.py --help`).

| Key | Default | Description |
|---|---|---|
| `paths.data_root` | `../graph_construction/data/linux` | dataset root (`--data-root`) |
| `paths.graph_mode` | `full_graph` | `full_graph` / `no_temporal` / `bszz_candidates` |
| `paths.save_dir` | `checkpoints_GAT_Temporal` | checkpoint output (`--save-dir`) |
| `defaults.n_folds` | `5` | CV folds — **also read by `run_evaluation.py`** |
| `defaults.seed` | `123` | split + init seed — **also read by `run_evaluation.py`** |
| `phase1.epochs` | `30` | Phase 1 epochs |
| `phase1.bert_freeze_bottom_layers` | `8` | UniXcoder layers frozen |
| `phase2.epochs` | `40` | Phase 2 epochs |
| `phase2.top_k_lines` | `3` | Phase-1 lines pooled into Phase 2 |
| `phase2.num_commit_transformer_layers` | `2` | commit-Transformer depth |
| `phase2.use_dual_query` | `True` | dual-query cross-attention |
| `phase2.use_temporal_pe` | `True` | Phase 2 temporal positional encoding |

> `n_folds` and `seed` are shared with `run_evaluation.py`, which reconstructs
> the fold splits from them. Change them in one place only, or evaluation will
> silently score against different test sets than training used.

---

## Acknowledgments

- **SZZ baselines** — adapted from [juzizi44/LLM_SZZ](https://github.com/juzizi44/LLM_SZZ), refined for C/C++ datasets.
- **Remaining SZZ variants** — run using the replication package of *"Assessing and Enhancing Mining Techniques for Vulnerability-Contributing Commits"* ([Figshare](https://figshare.com/articles/software/Assessing-and-Enhancing-Mining-Techniques-for-Vulnerability-Contributing-Commits/24305659/4?file=54777614)).
- **NeuralSZZ** — adapted from the [NeuralSZZ paper](https://baolingfeng.github.io/papers/ASE2023.pdf).
- **Code embeddings** — [UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder) (`microsoft/unixcoder-base-nine`).
- **CPG generation** — [Joern](https://joern.io/).
- **AST differencing** — [GumTree](https://github.com/GumTreeDiff/gumtree).
