# Baselines

Comparison methods for TraceVIC, evaluated on the **same 755 Linux CVE cases**
and with the **same metric** as TraceVIC itself.

| baseline | where | what it is |
|---|---|---|
| **B-SZZ** | [`SZZ/`](SZZ/README.md) | standard Base SZZ |
| **AG-SZZ** | [`SZZ/`](SZZ/README.md) | Annotation-Graph SZZ |
| **MA-SZZ** | [`SZZ/`](SZZ/README.md) | Meta-change-Aware SZZ |
| **V-SZZ** | [`SZZ/`](SZZ/README.md) | Vulnerability SZZ (Levenshtein line tracing) |
| **NeuralSZZ** | [`neuralszz/`](neuralszz/README.md) | HAN + RankNet deletion-line ranking, then SZZ on the top-k lines |
| **R-SZZ**, **L-SZZ**, **RA-SZZ**, **TSE-SZZ** | [`SZZ_remaining/`](SZZ_remaining/README.md) | run via an external replication package; the drivers, results and full procedure are shipped |

---

## Why the numbers are comparable

All baselines and TraceVIC are scored against the identical population, with
the identical formula:

- **755 cases** — verified 755/755 overlap on fixing commit between
  `SZZ/data/verified_cve_with_versions_C.json` and
  `graph_construction/data/linux/*/info.json`, with none on either side only.
  The dataset's `case_id` field (`test2`, …) matches the case directory names.
- **787 ground-truth inducing commits** — the same total on both sides.
- **`sha[:12]` prefix matching** on both sides.
- **Micro-averaged** precision = matches / predicted, recall = matches / GT —
  accumulated across all cases, then divided.

**One structural difference to state when reporting:** TraceVIC ranks and
reports at a cut-off (@1/@2/@3); the SZZ variants return an *uncapped* set
(mean 1.26–1.45 commits per case, median 1, max 13). Their sets sit between
TraceVIC's @1 and @2, which raises their recall and lowers their precision
relative to a strict @1 comparison. See
[`SZZ/README.md`](SZZ/README.md#evaluation-protocol--identical-to-tracevics)
for the full table.

---

## Shared dependency

`graph_construction` uses this directory: `config.yaml`'s
`vszz.pyszz_path` points at `baselines/SZZ`, and
`history_commits_retrieval.py` (Step 4) imports `szz.my_szz_ast`,
`szz.b_szz` and `szz.core.abstract_szz` from it. **Graph construction Step 4
cannot run without `baselines/SZZ` present.**

---

## What is and is not shipped

**Included:** all SZZ implementations, every evaluation driver, the
755-case ground-truth dataset, result files for **all eight** SZZ variants, the
drivers that bridge the external replication package to our dataset, and
NeuralSZZ's pre-built mini-graphs.

**Not included:** trained NeuralSZZ checkpoints (~3 GB) and per-run logs.
NeuralSZZ must be retrained to reproduce its numbers; the SZZ variants need no
training and can be re-evaluated directly from the shipped results.
