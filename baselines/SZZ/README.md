# SZZ Baselines for TraceVIC

This directory contains the SZZ baseline implementations used for comparison against the TraceVIC approach. The codebase has been streamlined for C/C++ projects and includes scripts for both executing the baselines and evaluating their performance against our curated ground-truth vulnerability dataset.

> **Acknowledgments:** This codebase was originally adapted from [juzizi44/LLM_SZZ](https://github.com/juzizi44/LLM_SZZ) and has been refined for C/C++ datasets.
## Supported Baselines

Four SZZ variants are implemented here, and their 755-case outputs are shipped
under `results/` — see [Shipped results](#shipped-results). Additional variants
are documented under [Other SZZ variants](#other-szz-variants).

- **B-SZZ** (`b_szz.py`): The standard Base SZZ algorithm.
- **AG-SZZ** (`ag_szz.py`): Annotation-Graph SZZ, an extension of B-SZZ with size thresholds and structure awareness.
- **MA-SZZ** (`ma_szz.py`): Meta-Change-Aware SZZ.
- **V-SZZ** (`v_szz.py`): Vulnerability-SZZ. Traces deleted lines iteratively through history using Levenshtein distance matching. 
  *(Note: A more advanced graph-aware variant `MySZZAST` is used directly by the TraceVIC graph construction pipeline).*

## Prerequisites

### 1. Ground-truth dataset — already included

`data/verified_cve_with_versions_C.json` (755 Linux CVE cases, 787 ground-truth
inducing commits) ships with this repository. **No download is required.**

It is the same file the remaining-SZZ drivers use — see
[`../SZZ_remaining/`](../SZZ_remaining/README.md).

### 2. Repositories
The baselines share the cloned git repositories with the graph construction pipeline. Ensure that the repositories (e.g., `linux`, `FFmpeg`, `OpenSSL`) are cloned and available at:
```
../../graph_construction/repositories/
```

## Usage

### 1. Running the Baselines

Use `main.py` to run the SZZ baselines. By default, it processes the C dataset. 
The `--time` parameter acts as an output folder name for the given run (e.g., `run1`).

```bash
# Run Base-SZZ (B-SZZ)
python3 main.py --method b --language C --time run1

# Run V-SZZ
python3 main.py --method v --language C --time run1

# Run Annotation-Graph SZZ
python3 main.py --method ag --language C --time run1

# Run Meta-Change-Aware SZZ
python3 main.py --method ma --language C --time run1
```

The output JSON files tracking the predicted inducing commits will be generated inside `results/<method>-szz/C/<time>/`.

### 2. Evaluating the Baselines

Once the predictions have been generated, you can evaluate them against the ground truth using `evaluate.py`. 

The evaluation script uses a 12-character SHA prefix match and calculates **Precision**, **Recall**, and **F1-Score** globally across the entire dataset. It also provides a per-project breakdown.

```bash
# Evaluate Base-SZZ (B-SZZ)
python3 evaluate.py --method b --language C --time run1

# Evaluate V-SZZ
python3 evaluate.py --method v --language C --time run1

# Evaluate Annotation-Graph SZZ
python3 evaluate.py --method ag --language C --time run1

# Evaluate Meta-Change-Aware SZZ
python3 evaluate.py --method ma --language C --time run1
```

## Code Structure

- `main.py`: Entry point for running the baselines.
- `evaluate.py`: Calculates unified global metrics against the ground truth.
- `setting.py`: Configuration constants and shared relative path resolutions.
- `data_loader.py`: Handles loading and mapping from the `verified_cve_with_versions_C.json` dataset.
- `log_generation.py`: Contains Git operations for logging and diff analysis.
- `szz/`: Contains the specific SZZ baseline algorithm implementations.
- `szz/core/`: Contains the shared `AbstractSZZ` logic and the C/C++ `srcML`-based comment parser.


---

## Shipped results

The 755-case Linux outputs for all four variants are included, so the reported
numbers can be verified without re-running the baselines (which requires the
full kernel git history):

```
results/<method>-szz/C/all755/<method>-linux.json
```

Schema differs slightly by method:

| method | value per fixing commit |
|---|---|
| `b`, `ag`, `ma` | flat list of inducing commit SHAs |
| `v` | list of per-line records, each carrying `previous_commits`; `evaluate.py` takes the **last** entry of each chain |

Reproduce the evaluation directly:

```bash
python evaluate.py --method b  --time all755
python evaluate.py --method ag --time all755
python evaluate.py --method ma --time all755
python evaluate.py --method v  --time all755
```

---

## Other SZZ variants

**R-SZZ, L-SZZ, RA-SZZ and TSE-SZZ** are not implemented in this directory.
They were run from the replication package of *"Assessing and Enhancing Mining
Techniques for Vulnerability-Contributing Commits"*.

Downloading that package alone is **not** sufficient to regenerate our numbers —
it has no knowledge of our dataset. The drivers that bridge it to our 755-case
CVE set, together with their outputs, are shipped in
[`../SZZ_remaining/`](../SZZ_remaining/README.md), which documents the full
procedure.

---

## Evaluation protocol — identical to TraceVIC's

`evaluate.py` and TraceVIC's `training_pipeline/training/evaluation.py`
(`evaluate_global`) compute the **same** metric, so the numbers are directly
comparable:

| | SZZ `evaluate.py` | TraceVIC `evaluate_global` |
|---|---|---|
| cases | 755 | 755 (verified: 755/755 overlap on fixing commit) |
| ground-truth commits | **787** | **787** |
| SHA matching | `sha[:12]` prefix | `sha[:12]` prefix |
| precision | matches / predicted commits | hits / predicted commits |
| recall | matches / GT commits | hits / GT commits |
| averaging | micro (accumulate, then divide) | micro (accumulate, then divide) |

**The one structural difference, which should be stated when reporting.**
TraceVIC ranks and reports at a fixed cut-off (@1 / @2 / @3). The SZZ variants
do not rank — they return an *uncapped* set. Measured over the 755 cases:

| method | mean predictions / case | median | max | exactly 1 |
|---|---|---|---|---|
| B-SZZ | 1.45 | 1 | 13 | 43% |
| AG-SZZ | 1.26 | 1 | 13 | 48% |
| MA-SZZ | 1.37 | 1 | 13 | 45% |
| V-SZZ | 1.43 | 1 | 13 | 45% |

Their prediction sets therefore sit between TraceVIC's @1 and @2, which raises
their recall and lowers their precision relative to a strict @1 comparison.
