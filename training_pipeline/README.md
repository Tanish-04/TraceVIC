# TraceVIC — Training Pipeline

Two-phase training and evaluation for Vulnerability-Inducing Commit (VIC)
identification. Consumes the graphs produced by
[`../graph_construction`](../graph_construction/README.md).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Environment](#2-environment)
3. [Directory Structure](#3-directory-structure)
4. [Configuration](#4-configuration)
5. [Step 1 — Pre-compute Graph Structures](#5-step-1--pre-compute-graph-structures)
6. [Step 2 — Train](#6-step-2--train)
7. [The Three Arms](#7-the-three-arms)
8. [Step 3 — Evaluate](#8-step-3--evaluate)
9. [Outputs](#9-outputs)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

VIC identification runs in two sequential stages, each building on the previous.

**Stage 1 — Root-Cause Line Identification.** Given a patched file, the model
ranks the *deletion lines* most likely to represent the root cause of the
vulnerability. This narrows the search space and provides a fine-grained,
code-level signal.

**Stage 2 — VIC Ranking.** Using the top-ranked lines from Stage 1 as anchors,
the model pools each line's candidate commit chain and ranks those candidates to
surface the commit that introduced the vulnerability.

| Paper term | Codebase term | What it does |
|---|---|---|
| Root-Cause Line Identification | **Phase 1** | Trains on contrastive deletion-line pairs to learn which lines caused the vulnerability |
| VIC Ranking | **Phase 2** | Uses Phase 1 embeddings to rank candidate commits |

```
graph_construction/data/<project>/       (graph.json, graph_vszz.json, ...)
        │
        ▼
[Step 1]  build_graphs.py      → temporal_graph/<mode>/<case>/del_*.json
        │
        ▼
[Step 2]  main_kfold.py        → k-fold CV: Phase 1 → Phase 2, per fold
        │
        ▼
[Step 3]  run_evaluation.py    → P/R/F1 tables; optional external datasets
```

---

## 2. Environment

Training uses the same conda environment as graph construction.

```bash
conda activate tracevic
```

If it does not exist yet, create it from `environment.yml` at the repository
root — see [`../graph_construction/README.md`](../graph_construction/README.md).

Requires a CUDA GPU. The reported runs used a single RTX 4090 (24 GB); Phase 1
peaks around 13–15 GB at the default `max_graphs_per_batch: 41`.

---

## 3. Directory Structure

```text
training_pipeline/
├── training_config.yaml     # Every hyper-parameter, path and switch (see §4)
├── config_utils.py          # Loads training_config.yaml; resolves relative paths
│
├── build_graphs.py          # Step 1: pre-compute PyG graph structures
├── main_kfold.py            # Step 2: k-fold CV training (Phase 1 → Phase 2)
├── run_evaluation.py        # Step 3: aggregate results / evaluate on new data
│
├── data_processing/         # Data loading, graph assembly, batching
│   ├── constants.py         #   EdgeType vocabulary (CFG/DFG/LINEMAP/TEMPORAL)
│   ├── dataset.py           #   DeletionLineDataset, CommitRankingDataset, collate
│   └── phase1/
│       ├── minigraph.py     #   MiniGraph + localized CFG/DFG neighbourhood walk
│       ├── pairs.py         #   Contrastive deletion-line pair construction
│       └── processing.py    #   Graph builders (sections, chains, temporal edges)
│
├── models/                  # Architectures
│   ├── base.py              #   BaseEncoder / BasePhase1Model
│   ├── shared_encoder.py    #   GAT encoder over CPG + temporal edges, UniXcoder base
│   ├── phase1_model.py      #   Phase 1: deletion-line ranker
│   └── phase2_model.py      #   Phase 2: commit ranker (dual-query attention)
│
└── training/                # Training loops, losses, evaluation
    ├── phase1_trainer.py    #   Phase 1 loop, differential LRs, early stopping
    ├── phase2_trainer.py    #   Phase 2 loop
    ├── embedding_cache.py   #   Phase 1 scoring + Phase 2 candidate pooling
    ├── evaluation.py        #   Global top-k VIC metrics (evaluate_global)
    ├── loss.py              #   PairwiseRankingLoss, LabelSmoothingRankingLoss
    └── utils.py             #   Seeding, device setup, EarlyStopping, model builders
```

Not tracked in git (see `.gitignore`): `temporal_graph/`, `temporal_graph_gen/`,
`checkpoints_*/`.

**None of these are distributed** — they are all regenerated from
`graph_construction/data/`, which is the only thing on OSF:

| item | regenerate with | cost |
|---|---|---|
| `temporal_graph/{full_graph,no_temporal}` | `build_graphs.py --mode all` | ~30 s |
| `temporal_graph/bszz_candidates/` | `build_graphs.py --mode bszz_candidates` | ~10 s |
| `temporal_graph_gen/` | `build_graphs.py` with `--data_path`/`--output_dir` per project | seconds |
| `.pt` caches beside each `del_*.json` | built on first use, invalidated by mtime | automatic |

Checkpoints are not distributed either and must be retrained (§8).

---

## 4. Configuration

Everything lives in **`training_config.yaml`**, in five sections: `paths`,
`phase1`, `phase2`, `model`, `defaults`.

**Every value can also be overridden on the command line.** CLI flags default to
`None`, so omitting a flag always leaves the YAML value in effect:

```bash
python main_kfold.py --help
```

The ones you are most likely to change:

| Setting | CLI flag | Meaning |
|---|---|---|
| `paths.data_root` | `--data-root` | Dataset root, one subdirectory per test case. Point this anywhere to train on a different dataset. |
| `paths.prebuilt_dir` | `--prebuilt-dir` | Where `build_graphs.py` wrote its output |
| `paths.graph_mode` | `--graph-mode` | `full_graph` / `no_temporal` / `bszz_candidates` — selects both the model's view and the `prebuilt_dir` subfolder |
| `paths.save_dir` | `--save-dir` | Where checkpoints and fold results are written |
| `defaults.gpu_id` | `--gpu-id` | CUDA device index |
| `defaults.n_folds` | `--n-folds` | Number of CV folds |
| `defaults.seed` | `--seed` | Split + init seed |

> **`defaults.n_folds` and `defaults.seed` are read by `run_evaluation.py` too.**
> That is deliberate: evaluation reconstructs the fold splits from these values,
> so if they did not match what training used, it would silently report metrics
> against the wrong test sets. Change them in one place only.

---

## 5. Step 1 — Pre-compute Graph Structures

Converts the raw graphs from `graph_construction` into PyG-ready per-deletion-line
structures. Run once per graph mode.

```bash
# The two modes used by --mode all
python build_graphs.py --mode full_graph      # GAT + temporal   (main method)
python build_graphs.py --mode no_temporal     # GAT - temporal   (ablation)
python build_graphs.py --mode all             # both of the above

# The B-SZZ candidate arm — opt-in, NOT part of --mode all
python build_graphs.py --mode bszz_candidates
```

| flag | default |
|---|---|
| `--data_path` | `paths.data_root` |
| `--output_dir` | `paths.prebuilt_dir` |

**Output:** `temporal_graph/<mode>/<test_name>/del_<node_idx>.json`

Expected on the 755-case Linux dataset:

| mode | cases | graphs | source file |
|---|---|---|---|
| `full_graph` | 755 | 3,457 | `graph_vszz.json` |
| `no_temporal` | 755 | 3,457 | `graph_vszz.json` |
| `bszz_candidates` | 755 | 2,825 | `graph_bszz_chains.json` |

`bszz_candidates` uses the *same builder* as `full_graph`; the only difference is
the source file — one B-SZZ blame hop rather than the multi-hop V-SZZ chain. It
is excluded from `--mode all` so a routine rebuild does not silently triple in
cost. It requires `graph_bszz_chains.json`, produced by
`history_commits_retrieval.py --szz bszz` in graph construction.

> Because chains are depth-1, `bszz_candidates` graphs carry only temporal
> positions `{0, 1}` and one temporal edge pair per chain. That is what "no
> history" means structurally, not a bug.

**Caching.** The first training run writes a `.pt` cache next to each
`del_*.json`. Caches are invalidated automatically by mtime, so editing a graph
builder and rebuilding is safe — stale caches are never silently reused.

---

## 6. Step 2 — Train

`main_kfold.py` runs k-fold cross-validation. Within each fold it trains Phase 1,
evaluates it, caches the embeddings, then trains Phase 2 on top.

```bash
python main_kfold.py --graph-mode full_graph --save-dir checkpoints_GAT_Temporal
```

Splits are 3-way per fold: the held-out fold is the test set, and validation is
carved from the **non-test** portion only (`defaults.val_split`), so test data is
never used for model selection.

Useful flags beyond §4:

| flag | effect |
|---|---|
| `--phase1-epochs`, `--phase2-epochs` | epoch budgets |
| `--skip-phase1 --phase1-checkpoint-dir DIR` | reuse per-fold Phase 1 checkpoints and train only Phase 2 — use this to vary a Phase 2 setting with Phase 1 held fixed |
| `--no-model-use-temporal-pe`, `--no-phase2-use-temporal-pe` | disable the encoder / Phase 2 positional encodings |

Always redirect output to a file — an SSH drop otherwise kills the run and its
loss curve is unrecoverable:

```bash
nohup python -u main_kfold.py --graph-mode full_graph \
    --save-dir checkpoints_GAT_Temporal > train_gat_temporal.log 2>&1 &
```

Use `python -u`; without it stdout is block-buffered and the log stays empty for
a long time even while training runs normally.

---

## 7. The Three Arms

These are the three configurations reported in the paper. **Copy these commands
verbatim** — the ± temporal arm needs three flags set together, and omitting one
silently trains a different configuration that is not in the paper.

**Arm 1 — GAT + temporal (main method)**
```bash
python main_kfold.py \
    --graph-mode full_graph \
    --save-dir checkpoints_755_GAT_Temporal
```

**Arm 2 — GAT − temporal (ablation)**
```bash
python main_kfold.py \
    --graph-mode no_temporal \
    --no-model-use-temporal-pe \
    --no-phase2-use-temporal-pe \
    --save-dir checkpoints_755_GAT_NoTemporal
```

**Arm 3 — B-SZZ candidates (ablation)**
```bash
python build_graphs.py --mode bszz_candidates      # once, if not built yet
python main_kfold.py \
    --graph-mode bszz_candidates \
    --save-dir checkpoints_755_BSZZ_Candidates
```

What each arm isolates:

| arm | question it answers |
|---|---|
| GAT + temporal | the full method |
| GAT − temporal | does the *representation* of history help? Same candidates, temporal edges and positional encodings removed. |
| B-SZZ candidates | does having the *full chain* as candidates help, versus only the latest blame hop? Same model, different candidate set. |

Only the GAT encoder ships with TraceVIC; `--encoder-type` accepts `gat` only.

---

## 8. Step 3 — Evaluate

```bash
# Aggregate k-fold results into P/R/F1 tables (fast, no GPU)
python run_evaluation.py

# One arm, at a specific top_k_lines block
python run_evaluation.py --arm gat_temporal --k-lines 3

# Evaluate a trained model on an external dataset
python run_evaluation.py --arm gat_temporal \
    --data-root ../graph_construction/data/FFmpeg \
                ../graph_construction/data/OpenSSL \
    --prebuilt-dir temporal_graph_gen/FFmpeg temporal_graph_gen/OpenSSL
```

> **Checkpoints are not distributed.** `run_evaluation.py` reads each fold's
> `phase1_best.pt` / `phase2_best.pt`, so it only works after you have trained
> the arm yourself (§7). The `.pt` files total ~39 GB across all arms, which is
> why they are excluded.

If you trained with a non-default `--save-dir`, point evaluation at it:

```bash
python run_evaluation.py --ckpt-dir gat_temporal=checkpoints_myrun
```

`--verify-folds` re-runs inference and checks that the reconstruction reproduces
each fold's logged recall — worth doing before trusting a reconstruction on new
data. Small drift is expected: Phase 2 score spreads are narrow, so near-ties
flip under GPU non-determinism.

---

## 9. Outputs

```text
<save_dir>/
├── fold_0/
│   ├── phase1_best.pt        # Phase 1 weights (encoder + ranker)
│   ├── phase2_best.pt        # Phase 2 commit-ranker weights
│   └── fold_results.json     # metrics + the exact train/val/test split
├── fold_1/ ...
└── kfold_summary.json        # aggregated mean ± std across folds
```

`fold_results.json` records the split sizes, so a fold can always be
reconstructed and audited after the fact.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Log file stays empty while the GPU is busy | stdout is block-buffered | run with `python -u` |
| `[SKIP] ... not found` during Step 1 | graph construction did not produce that case's source file | re-run the relevant graph-construction step; `bszz_candidates` needs `graph_bszz_chains.json` |
| Evaluation numbers do not match training | `n_folds` / `seed` changed between the two | both read `defaults.*` in the YAML — change in one place only |
| `run_evaluation.py` cannot find checkpoints | checkpoints are not distributed, or `--save-dir` differed | train the arm (§7), or pass `--ckpt-dir ARM=DIR` |
| CUDA out of memory in Phase 1 | one oversized test case | lower `phase1.max_graphs_per_batch`; note batches are never split mid-case, so peak memory is set by the largest single case |
| Results differ after editing a graph builder | none — caches self-invalidate by mtime | rebuild with `build_graphs.py`; the `.pt` caches refresh automatically |
