# Remaining SZZ Baselines — R-SZZ, L-SZZ, RA-SZZ, TSE-SZZ

These four variants are **not implemented in this repository**. They were run
using the replication package of *"Assessing and Enhancing Mining Techniques for
Vulnerability-Contributing Commits"*, which ships its own implementations under
`method_replication_scripts/`.

> **Replication package (Figshare):**
> <https://figshare.com/articles/software/Assessing-and-Enhancing-Mining-Techniques-for-Vulnerability-Contributing-Commits/24305659/4?file=54777614>

What this directory contains is the **bridge**: the upstream package has no
knowledge of our dataset, so downloading it alone is not enough to regenerate
our numbers. The two drivers here connect it to our 755-case CVE set and score
the output with the **same metric** used for every other baseline and for
TraceVIC itself.

---

## Contents

| file | role |
|---|---|
| `run_remaining_szz.py` | drives R/L/RA/TSE-SZZ over our dataset, writing `results/<method>-szz/<project>/<run>/<method>-<project>.json` |
| `evaluate_remaining_szz.py` | scores those results — 12-char SHA prefix match, micro precision/recall/F1, identical to `../SZZ/evaluate.py` |
| `results/` | the outputs behind the reported numbers (see [Shipped results](#shipped-results)) |
| `datasets/generalizability_C.json` | CVE set for the non-Linux generalizability runs |

Both drivers resolve every path relative to their own location
(`HERE = dirname(__file__)`), so they run unmodified once dropped into a clone
of the upstream package.

---

## Reproducing

### The layout that must exist

The drivers resolve everything relative to their own location, and
`run_remaining_szz.py` `chdir`s into `method_replication_scripts/` (RA-SZZ's
vendored code hardcodes `../tools/...` relative to that directory). So the
package contents must end up as **siblings of the two scripts**:

```
<working dir>/
├── run_remaining_szz.py            ← from this repo
├── evaluate_remaining_szz.py       ← from this repo
├── method_replication_scripts/     ← from Figshare  (r_szz.py, l_szz.py, ra_szz.py, tse_szz.py, …)
├── tools/                          ← from Figshare  (RefactoringMiner-2.0 — RA-SZZ only)
├── datasets/
│   ├── tempovic/verified_cve_with_versions_C.json
│   └── tempovic_generalizability/generalizability_C.json
├── repos/                          ← you provide    (linux/, FFmpeg/, … full history)
└── results/                        ← created by the run
```

**The one thing to watch:** unpacking the Figshare archive so it becomes a
*nested* folder (e.g. `…/replication_package/method_replication_scripts/`) will
not work — the drivers would look for `method_replication_scripts/` directly
beside themselves and not find it. The contents must be merged into one
directory, not nested.

Either arrangement gets you there:

- **A** — unpack the Figshare package somewhere, then copy the two drivers into
  its root.
- **B** — unpack the Figshare package's *contents* into this directory
  (`baselines/SZZ_remaining/`), beside the drivers already here.

They are equivalent. Use B if you want everything under the repo.

### Steps

1. **Download and unpack the replication package** from the Figshare link above,
   arranging it per option A or B.

2. **Place our dataset.** This is the *same file* already shipped at
   `../SZZ/data/verified_cve_with_versions_C.json`:

   ```bash
   mkdir -p datasets/tempovic
   cp ../SZZ/data/verified_cve_with_versions_C.json datasets/tempovic/

   # generalizability runs (TSE-SZZ on the non-Linux projects)
   mkdir -p datasets/tempovic_generalizability
   cp datasets/generalizability_C.json datasets/tempovic_generalizability/
   ```

3. **Provide the git repositories** at `repos/<project>` with full history —
   the same clones `graph_construction` uses:

   ```bash
   ln -s ../../graph_construction/repositories repos      # or copy/clone
   ```

4. **RA-SZZ only:** it shells out to
   `../tools/RefactoringMiner-2.0/bin/RefactoringMiner`, so `tools/` must be
   unpacked and that binary executable. The other three methods do not need it.

5. **Run and evaluate:**

   ```bash
   python run_remaining_szz.py --method r   --time run1
   python run_remaining_szz.py --method l   --time run1
   python run_remaining_szz.py --method ra  --time run1
   python run_remaining_szz.py --method tse --time run1

   python evaluate_remaining_szz.py --all      # combined summary table
   ```

   Add `--project FFmpeg --time gen1` (etc.) with
   `--dataset datasets/tempovic_generalizability/generalizability_C.json`
   for the generalizability runs.

`run_remaining_szz.py` writes a `*-progress.json` beside each result listing the
fixing commits already completed, so an interrupted run resumes rather than
restarting. Those progress files are shipped too.

---

## Shipped results

```
results/<method>-szz/linux/run1/<method>-linux.json      R, L, RA, TSE  (755 cases each)
results/tse-szz/<project>/gen1/tse-<project>.json        TSE-SZZ generalizability:
                                                          FFmpeg, ImageMagick, OpenSSL, PHP-SRC
results/evaluation_summary.json                          combined metric table
```

Schema is `{fixing_commit_sha: [inducing_commit_shas]}` — the same flat form as
B-SZZ / AG-SZZ / MA-SZZ in `../SZZ/results/`.

Because the outputs are included, the reported numbers can be checked without
re-running the baselines (which needs the full kernel git history).

---

## Evaluation is identical to every other arm

`evaluate_remaining_szz.py` reimplements — deliberately, rather than importing,
so the upstream package stays self-contained — the metric from
`../SZZ/evaluate.py`. Confirmed against the shipped
`results/evaluation_summary.json`:

| | value |
|---|---|
| cases evaluated | **755** for all four methods |
| ground-truth commits | **787** for all four methods |
| SHA matching | `sha[:12]` prefix |
| precision / recall | matches / predicted, matches / GT |
| averaging | micro |

**787 is the same ground-truth total** as `../SZZ` (B/AG/MA/V-SZZ) and as
TraceVIC's own `evaluate_global`. All nine arms are scored against one
population with one formula.
