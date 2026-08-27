# Graph Construction — TraceVIC

> End-to-end pipeline that processes Linux kernel (and other C project) CVE commits into heterogeneous graphs.
---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Environment Setup](#environment-setup)
6. [Tool Setup](#tool-setup)
   - [Bear](#bear)
   - [GumTree](#gumtree)
   - [Joern](#joern)
7. [Configuration](#configuration)
8. [Data Preparation](#data-preparation)
9. [Pipeline Steps](#pipeline-steps)
   - [Step 1 — Extract Source Files](#step-1--extract-source-files)
   - [Step 2 — Generate Joern CPG](#step-2--generate-joern-cpg)
   - [Step 3 — Build Graphs (Fix Commit)](#step-3--build-graphs-fix-commit)
   - [Step 4 — Run V-SZZ](#step-4--run-v-szz)
   - [Step 4b — Create commits.json](#step-4b--create-commitsjson)
   - [Step 5 — Extract History Source Files](#step-5--extract-history-source-files)
   - [Step 6 — Generate Joern CPG (History)](#step-6--generate-joern-cpg-history)
   - [Step 7 — Build Graphs (History Commits)](#step-7--build-graphs-history-commits)
10. [Running a Specific Project](#running-a-specific-project)
11. [Output Structure](#output-structure)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The pipeline processes each CVE test case through the following stages:

```
[Step 0]  Build compile_commands.json with Bear   (once per repository)
        │
        ▼
CVE commit list (info.json)
        │
        ▼
[Step 1]  Extract before / after / fixing source files from git
        │
        ▼
[Step 2]  Run Joern to generate CPG dot files for each source directory
        │
        ▼
[Step 3]  Build heterogeneous graph.json for each fixing commit
        │
        ▼
[Step 4]  Run V-SZZ to trace deleted lines back through git history
        │
        ▼
[Step 4b] Collect all history commits into commits.json
        │
        ▼
[Step 5]  Extract source files for all history commits
        │
        ▼
[Step 6]  Run Joern on history commit source directories
        │
        ▼
[Step 7]  Build graph.json for all history commits
```

- **Step 0** is a one-off per repository; Clang needs the compile database to
  parse kernel sources.
- **Steps 1–3** operate on the **fixing commit** of each CVE.
- **Steps 4–7** extend coverage to all **history commits** discovered by V-SZZ.

Steps 2/6 and 3/7 are the *same two scripts* invoked twice — once for fixing
commits, once for history commits. Every step is **resumable**: re-running skips
work that is already on disk.

---

## Dataset

The fully processed dataset (ready for use without running the pipeline) is available for download:

> **[⬇ Download TraceVIC Dataset (OSF)](https://osf.io/qfkjm)**

This archive contains the complete `data/` directory — the `info.json`, extracted
sources, patches and built graphs for every test case. It is distributed as a
single **`tracevic_data.tar.gz`** (~648 MB compressed, 3.5 GB extracted;
59,512 files). Download and extract it into `graph_construction/`:

```bash
cd graph_construction
tar -xzf /path/to/tracevic_data.tar.gz     # creates ./data/
```

That is enough to skip the pipeline entirely — everything downstream is
regenerated locally (see below).

| project | test cases |
|---|---|
| `linux` | 755 |
| `FFmpeg` / `OpenSSL` / `ImageMagick` / `PHP-SRC` | 21 each |

`data/` is **not** tracked in git (see `.gitignore`) — it is ~3.5 GB and is
distributed through OSF instead.

### What the OSF archive contains

**`graph_construction/data/` only** (~3.5 GB) — the processed dataset: every
case's `info.json`, extracted sources, patches, and built graphs.

Everything downstream is **regenerated locally**, not downloaded:

| not distributed | regenerate with | cost |
|---|---|---|
| `training_pipeline/temporal_graph/` | `build_graphs.py --mode all` | ~30 s |
| `temporal_graph/bszz_candidates/` | `build_graphs.py --mode bszz_candidates` | ~10 s |
| `.pt` caches beside each `del_*.json` | built on first use, mtime-invalidated | automatic |
| `joern_output/` | Steps 2 / 6 | intermediate only |
| `repositories/` | `git clone` | see Tool Setup |
| `bear_compilation/compile_commands_*.json` | `bear_compilation/build_*.py` | multi-hour kernel build |

---

## Repository Structure

```
graph_construction/
│
├── config.yaml                        # Central config: project paths, tool paths, settings
├── config_loader.py                   # Loads config.yaml; exposes typed getter functions used by all scripts
├── pipeline_utils.py                  # Shared utilities (git helpers, file I/O) used across pipeline scripts
│
├── extract_source_files.py            # Step 1 / Step 5: Checks out before/after/fixing C source files from git
├── generate_joern.py                  # Step 2 / Step 6: Invokes Joern to generate CPG dot files for all source dirs
├── generate_graphs.py                 # Step 3 / Step 7: Builds heterogeneous graph.json from Joern CPG output
├── history_commits_retrieval.py       # Step 4: Runs V-SZZ to trace patch deletions back through git history
├── create_commits_json.py             # Step 4b: Reads V-SZZ output and writes structured commits.json per test
├── sem_szz_analysis.py                # Addition-only fixes: derives anchor lines when a patch deletes nothing
│
├── parse_patch/                       # Library: parses unified diff (.patch) files
│   ├── patch.py                       #   Main patch parser — reads hunks and changed lines
│   ├── patch_file.py                  #   Represents a single file changed in a patch
│   ├── patch_hunk.py                  #   Represents a hunk (contiguous block of changes) in a patch
│   └── patch_line.py                  #   Represents an individual added/deleted/context line in a hunk
│
├── parse_joerndot/                    # Library: parses Joern-generated CPG dot files into Python objects
│   ├── joern_graph.py                 #   Reads dot file and builds graph of Joern nodes and edges
│   ├── joern_node.py                  #   Represents a single CPG node (method, call, identifier, etc.)
│   └── joern_edge.py                  #   Represents a CPG edge (CFG, DFG, AST, call, etc.)
│
├── gen_finalgraph/                    # Library: assembles the final heterogeneous graph.json
│   ├── gen_graph.py                   #   Core builder — integrates patch, Joern CPG, and Clang data into one graph
│   ├── trim_graph.py                  #   Post-processes graph to prune unreachable or irrelevant nodes/edges
│   ├── node.py                        #   Node dataclass for the final output graph
│   └── edge.py                        #   Edge dataclass for the final output graph
│
├── project/                           # Library: Clang-based source file analysis
│   ├── project.py                     #   Manages a set of source files for a commit (before or after)
│   ├── source_file.py                 #   Parses a single C source file using Clang to extract functions/calls
│   ├── clang_function_tracker.py      #   Tracks function definitions and call sites via the Clang AST
│   ├── union_project.py               #   Merges before/after projects to align symbols across versions
│   ├── method.py                      #   Represents a function/method extracted from source
│   └── call.py                        #   Represents a function call site extracted from source
│
├── mapping/                           # Library: maps line numbers between before/after using GumTree AST diffs
│   ├── line_mapping.py                #   Main entry point — builds before↔after line number mapping
│   ├── gumtree_bridge.py              #   Calls GumTree binary and parses its AST edit script output
│   ├── python_parser_generator.py     #   Drives the GumTree binary and parses its tree output
│   ├── get_pos.py                     #   Utility to resolve line/column positions within source text
│   ├── genJoern.sc                    #   Joern Scala script: runs cpg2dot on all entries in joern_targets.txt
│   └── gumtree/                       #   GumTree tool binary (built from source; see Tool Setup)
│
├── bear_compilation/                  # Scripts to generate compile_commands.json via Bear
│   ├── build_linux_kernel_compile_dbs.py   # Builds Linux kernel across architectures and merges compile DBs
│   ├── build_general_projects_compile_db.py # Builds compile DBs for FFmpeg, OpenSSL, ImageMagick, PHP-SRC
│
├── app/                               # Library: low-level Clang parsing and shared data types
│   ├── clang_parser.py                #   Parses a C file with libclang using the project's compile DB
│   ├── clang_extract.py               #   Pulls typed constructs (functions, statements, macros) from the AST
│   ├── clang_gen_units.py             #   Turns extracted cursors into CUnit semantic units, de-dupes and filters
│   ├── compile_commands_supports.py   #   Looks up and normalises compile flags from compile_commands.json
│   └── unit.py                        #   CUnit / Position dataclasses shared across the pipeline
│
├── data/                              # Output directory — one subdirectory per project (e.g. linux/, FFmpeg/)
├── repositories/                      # Cloned source repos (linux, FFmpeg, etc.) with full git history
└── workspace/                         # Temporary working files created during pipeline execution
```

---

## Prerequisites

### Hardware

| Requirement | Minimum |
|---|---|
| OS | Linux x86-64 (Ubuntu 22.04) |
| RAM | 16 GB |
| Disk | 80 GB (see breakdown) |

A completed Linux run measures ~37 GB on disk, plus the working tree and object
files the Bear-wrapped kernel build produces:

| | size |
|---|---|
| `repositories/linux` (full git history) | ~15 GB |
| `joern_output/` (CPG dot files, Steps 2 / 6) | ~13 GB |
| `data/` (sources, patches, graphs) | ~3.5 GB |
| `bear_compilation/compile_commands_*.json` | ~250 MB |
| `mapping/gumtree/` (built from source) | ~280 MB |

`joern_output/` may be deleted once Steps 3 and 7 have run — it is an
intermediate, not an input to training.

### System packages

```bash
sudo apt-get install -y \
    git \
    build-essential \
    bear \
    openjdk-21-jdk \
    python3-dev \
    curl \
    jq
```

Verify Bear and Java:
```bash
bear --version
java -version    # must be 21+
```

> **Java 21 is mandatory for Joern.** Its C/C++ parser is compiled for class-file
> version 61; a Java 11 default fails with `UnsupportedClassVersionError` and
> produces **zero output** without any obvious error. Note that the conda
> environment below also ships `openjdk=21.0.6`, so `conda activate tracevic`
> puts a suitable `java` on your `PATH` even if the system default is older.
> GumTree, however, is launched with an explicit `JAVA_HOME` read from
> `config.yaml` — see [Configuration](#configuration).

---

## Environment Setup

The pipeline uses a **conda environment with Python 3.7** (`tracevic`), defined in `environment.yml` at the repo root.

### Key packages installed by environment.yml

| Category | Packages |
|---|---|
| **Python** | `python=3.7.16` |
| **Deep Learning** | `pytorch=1.8.0`, `torch-geometric=2.2.0`, `torchvision`, `torchaudio` |
| **Graph / ML** | `networkx=2.7`, `scikit-learn`, `scipy`, `xgboost` |
| **NLP / Transformers** | `transformers=4.30.2`, `tokenizers`, `gensim`, `nltk` |
| **C / Clang parsing** | `clang=12.0.1`, `libclang=18.1.1` |
| **Git analysis** | `pydriller=1.15.2`, `gitpython`, `unidiff` |
| **Code analysis** | `lizard`, `levenshtein`, `rapidfuzz` |
| **General** | `pandas`, `numpy`, `pyyaml`, `tqdm`, `matplotlib` |
| **Java (for Joern)** | `openjdk=21.0.6`, `maven` |

### 1. Install Miniconda (if not already installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 2. Create and activate the environment

```bash
cd /path/to/TraceVIC          # repo root (where environment.yml lives)
conda env create -f environment.yml
conda activate tracevic
```

### 3. Fix Python PATH priority (important)

If your system has a `python` or `python3` binary in `~/bin` that overrides conda, remove or check it:

```bash
ls -la ~/bin/python* 2>/dev/null
```

If those are symlinks to the system Python, remove them so conda's interpreter takes priority:

```bash
rm ~/bin/python ~/bin/python3 ~/bin/python3.12   # adjust as needed
```

Verify the correct Python is active after reactivating:

```bash
conda activate tracevic
which python      # should point to miniconda3/envs/tracevic/bin/python
python --version  # should show Python 3.7.x
```

### 4. Set a UTF-8 locale (required by Joern)

With `LANG` unset (`LC_CTYPE=POSIX`) the JVM defaults to ASCII and Joern aborts
with `java.nio.charset.MalformedInputException` **at script startup**, before
parsing anything — so the error looks unrelated to your source files. Add to
`~/.bashrc`:

```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
```

### 5. Set the libclang environment variable

Add this to your `~/.bashrc`:

```bash
export GRAPHGEN_LIBCLANG="$(python -c "import clang, os; print(os.path.join(os.path.dirname(clang.__file__), 'native', 'libclang.so'))")"
```

Then reload:
```bash
source ~/.bashrc
conda activate tracevic
```

---

## Tool Setup

### Bear

Bear intercepts compiler calls during the kernel build to produce `compile_commands.json`, which Clang needs to parse kernel source files.

Install via apt (see Prerequisites). Verify:
```bash
bear --version
```

#### Generate compile_commands.json for Linux

```bash
cd TraceVIC/graph_construction

# Clone the Linux kernel (full history required for V-SZZ git blame)
mkdir -p repositories
git clone https://github.com/torvalds/linux.git repositories/linux

# Run Bear for all kernel builds (multi-arch / multi-version shard build + merge)
python3 bear_compilation/build_linux_kernel_compile_dbs.py
```

Output is written to: `repositories/linux/compile_commands.json`

#### Generate compile_commands.json for other projects

FFmpeg, OpenSSL, ImageMagick, and PHP-SRC use a separate script:

```bash
cd TraceVIC/graph_construction
python3 bear_compilation/build_general_projects_compile_db.py
```

Each `compile_commands.json` is written by Bear in the corresponding repo under `repositories/<Project>/`.

---

### GumTree

GumTree performs AST-level diff between before/after source versions to map patch line numbers accurately.

#### Build from source

```bash
cd TraceVIC/graph_construction/mapping
git clone https://github.com/GumTreeDiff/gumtree.git gumtree
cd gumtree

# If a previous partial build exists, clean it first
rm -rf dist/build/install/gumtree

# Build the shadow distribution
./gradlew :dist:installShadowDist
```

The binary is placed at: `mapping/gumtree/dist/build/install/gumtree/bin/gumtree`

This path is already set in `config.yaml`. No further configuration needed.

---

### Joern

Joern generates Code Property Graphs (CPG) from C source files.

#### Install

```bash
mkdir -p ~/bin/joern
cd ~/bin/joern
curl -L "https://github.com/joernio/joern/releases/latest/download/joern-install.sh" \
    -o joern-install.sh
chmod +x joern-install.sh
./joern-install.sh --install-dir .
```

Add Joern to PATH in `~/.bashrc`:
```bash
export PATH="$HOME/bin/joern/joern-cli:$PATH"
```

Reload and verify:
```bash
source ~/.bashrc
joern --version
```

---

## Configuration

All paths and settings are managed in `graph_construction/config.yaml`. Typed getter functions in `config_loader.py` expose them to the pipeline.

### Machine-specific values

Two values must be set per machine (not committed to the repo):

| Setting | How to set |
|---|---|
| `libclang_path` | Set `GRAPHGEN_LIBCLANG` env var (see Environment Setup above) |
| `joern` binary | Must be in `PATH` |
| `gumtree.java_home` | Edit `config.yaml`. Defaults to `/usr/lib/jvm/java-21-openjdk-amd64`, which is where the `openjdk-21-jdk` apt package installs. There is **no env-var override and no fallback** — if your JDK 21 lives elsewhere (including inside the conda env), set this path explicitly or GumTree will fail. |

Everything else in `config.yaml` uses paths relative to `graph_construction/` and works once the tools are built.

### Environment variables

All three are optional overrides — the pipeline works from `config.yaml` alone.

| variable | overrides | typical use |
|---|---|---|
| `GRAPHGEN_LIBCLANG` | `clang.libclang_path` | point Clang at the `libclang.so` inside your conda env |
| `GRAPHGEN_PROJECT` | `default_project` | switch the active project without editing `config.yaml` |
| `GRAPHGEN_COMPILE_DB` | the active project's `compile_db` | use a compile database from a non-default location |

The *names* themselves are configurable: `libclang_path_env_var`,
`project_env_var` and `compile_db_env_var` in `config.yaml` decide which
variables are consulted. If you change a name there, change it in **both**
places — `generate_graphs.py` sets `GRAPHGEN_COMPILE_DB` by literal name when it
switches projects.

### Switching projects at runtime

```bash
# Process FFmpeg (no file edits required)
export GRAPHGEN_PROJECT=FFmpeg
python extract_source_files.py --mode fix

# Or pass explicitly via --projects
python extract_source_files.py --mode fix --projects FFmpeg OpenSSL
```

### Verify config resolves correctly

```bash
cd graph_construction
python -c "
from config_loader import get_libclang_path, get_gumtree_bin_path, get_project_compile_db
print('libclang  :', get_libclang_path())
print('gumtree   :', get_gumtree_bin_path())
print('compile_db:', get_project_compile_db('linux'))
"
```

All three should print real paths with no errors before proceeding.

---

## Data Preparation

Each test case requires an `info.json` file at: `graph_construction/data/linux/<test_name>/info.json`

### info.json schema

```json
{
    "cve_id": "CVE-YYYY-NNNNN",
    "fix": "<40-character fix commit SHA>",
    "induce": ["<40-character ground truth inducing commit SHA>"]
}
```

### Using the provided dataset

Download the pre-formatted dataset from OSF **[TraceVIC Full Dataset (OSF)](https://osf.io/qfkjm)** and extract it into the `graph_construction/` directory.

The structure for each test case will look like this:
```
data/linux/<test_name>/
├── info.json
├── commits.json                         (Step 4b)
├── <fix_commit_hash>/
│   ├── graph.json                       (Step 3)
│   ├── graph_vszz.json                  (Step 4)
│   ├── graph_addition.json              (optional — addition-only baseline variant)
│   └── graph_bszz_chains.json           (optional — B-SZZ candidate variant)
└── <history_commit_hash>/
    └── graph.json                       (Step 7)
```

---

## Pipeline Steps

All scripts are run from `graph_construction/` with the conda environment active.

---

### Step 1 — Extract Source Files

Checks out the before/after/fixing C source files for each fixing commit.

```bash
# Linux only
python extract_source_files.py --mode fix --projects linux

# All configured projects
python extract_source_files.py --mode fix

# Specific subset
python extract_source_files.py --mode fix --projects FFmpeg OpenSSL
```

**Output per test case:**
```
data/linux/<test_name>/<fix_sha>/before/    ← C files before the fix
data/linux/<test_name>/<fix_sha>/after/     ← C files after the fix
data/linux/<test_name>/<fix_sha>/fixing/    ← the .patch file
```

---

### Step 2 — Generate Joern CPG

Runs Joern to generate CPG dot files for all `before/`, `after/` source directories.

```bash
python generate_joern.py --projects linux

# restrict to specific cases
python generate_joern.py --projects linux --testcases test100 test101
```

Joern runs **once** for all targets. The script writes `mapping/joern_targets.txt` (one `srcDir;outDir` pair per line) and then invokes `genJoern.sc` which processes them all.

**Output per source directory** — mirrored under a top-level `joern_output/`
tree, deliberately kept **outside** `data/` so that `importCode()` can never
re-ingest Joern's own prior output as if it were source:
```
joern_output/linux/<test_name>/<fix_sha>/before/   ← dot files per method
joern_output/linux/<test_name>/<fix_sha>/after/
```

---

### Step 3 — Build Graphs (Fix Commit)

Parses the CPG dot files and constructs heterogeneous `graph.json` for each fixing commit.

```bash
python generate_graphs.py --mode fix --projects linux
```

**Output:**
```
data/linux/<test_name>/<fix_sha>/graph.json
```

**Additional options:**

| Flag | Effect |
|---|---|
| `--testcases test1 test2 …` | Rebuild only the named cases instead of the whole project |
| `--addition-also` | Baseline variant for addition-only patches: skips SEM-SZZ anchor extraction and writes `graph_addition.json`, leaving `graph.json` untouched. Used by the NeuralSZZ / B-SZZ arms, which must not receive SEM-SZZ's synthesised deletion anchors. `--mode fix` only. |

**Addition-only fixing commits.** A patch that only adds lines has no deleted
line for V-SZZ to blame. `sem_szz_analysis.py` handles these by deriving *anchor*
lines around each insertion (the nearest meaningful line above or below, or the
enclosing function signature) and marking them as deletions, so the chain search
has a starting point. Pass `--addition-also` to build the variant that omits this
step.

**Skip conditions** (reported as `[SKIP]`):
- `graph.json` already exists for that commit → delete it to force a rebuild.
  This guard applies to **both** `--mode fix` and `--mode history`, so re-running
  either step is safe and resumable.
- Both the `before/` and `after/` Joern output dirs under `joern_output/` are empty → run Step 2 first
- No patch file in `fixing/`
- Assembly-only patch (`.S` files, no `.c` or `.h`)
- Addition-only patch (no deleted lines to trace)

---

### Step 4 — Run V-SZZ

Traces deleted lines in `graph.json` back through git history to identify candidate inducing commits.

```bash
python history_commits_retrieval.py --projects linux
```

Requires the Linux kernel repo to be cloned at `repositories/linux` with full git history.

**Output:**
```
data/linux/<test_name>/<fix_sha>/graph_vszz.json
```

**Additional options:**

| Flag | Effect |
|---|---|
| `--szz vszz` *(default)* | Walks the full history and writes `graph_vszz.json` — TraceVIC's normal candidate set. |
| `--szz bszz` | Takes a single blame hop and writes `graph_bszz_chains.json`, for the TraceVIC-pipeline-with-B-SZZ-candidates arm. Prefers `graph_addition.json` where present, so SEM-SZZ anchors are excluded. |

---

### Step 4b — Create commits.json

Reads `graph_vszz.json` and collects all history commits into a structured `commits.json` file used by Steps 5–7.

```bash
python create_commits_json.py --projects linux
```

**Output:** `data/linux/<test_name>/commits.json`

**commits.json schema:**
```json
{
    "fix_commit": "abc123...",
    "ground_truth": ["def456..."],
    "vszz_introducer_commits": ["ghi789..."],
    "all_commits_in_history": ["jkl012...", "..."],
    "matches_ground_truth": true,
    "graph_type": "vszz_full_history",
    "stats": {
        "num_introducers": 3,
        "num_history_commits": 47,
        "nodes_with_history": 12
    }
}
```

---

### Step 5 — Extract History Source Files

Checks out source files for every commit listed in `commits.json`.

```bash
python extract_source_files.py --mode history --projects linux
```

> **Ordering constraint.** Steps 5–7 all read `commits.json`, which only exists
> after Step 4b. Running them earlier reports `[SKIP] missing commits.json`.
>
> Unlike Steps 2, 3 and 7, `extract_source_files.py` has **no `--testcases`
> flag** — it processes a whole project. It also skips any case whose `after/`
> directory already holds a `.c` or `.h` file, so re-running it will not repair
> a partially-extracted case; delete that case's `before/ after/ fixing/`
> directories first.

**Output per history commit:**
```
data/linux/<test_name>/<history_sha>/before/
data/linux/<test_name>/<history_sha>/after/
data/linux/<test_name>/<history_sha>/fixing/
```

---

### Step 6 — Generate Joern CPG (History)

Runs Joern on all history commit source directories.

```bash
python generate_joern.py --projects linux
```

This is the same command as Step 2. It picks up any new directories not yet processed.

---

### Step 7 — Build Graphs (History Commits)

Builds `graph.json` for every history commit.

```bash
python generate_graphs.py --mode history --projects linux
```

---

## Running a Specific Project

Every script accepts `--projects` to restrict processing:

```bash
# Single project
python extract_source_files.py --mode fix --projects FFmpeg

# Multiple projects
python generate_graphs.py --mode fix --projects FFmpeg OpenSSL PHP-SRC

# All configured projects (default when --projects is omitted)
python generate_graphs.py --mode fix
```

Configured projects: `linux`, `FFmpeg`, `ImageMagick`, `OpenSSL`, `PHP-SRC`.  
To add a project, add an entry to the `projects:` block in `config.yaml` — no script changes required.

---

## Output Structure

The directory layout after running the complete pipeline:

```
graph_construction/
├── data/
│   └── linux/
│       └── test42/
│           ├── info.json                       # CVE metadata: fix SHA + ground truth
│           ├── commits.json                    # All history commits discovered (Step 4b)
│           ├── <fix_sha>/
│           │   ├── before/                     # Source files before the fix (Step 1)
│           │   ├── after/                      # Source files after the fix (Step 1)
│           │   ├── fixing/                     # Patch file (Step 1)
│           │   ├── graph.json                  # Heterogeneous graph — fix commit (Step 3)
│           │   └── graph_vszz.json             # Full history graph with inducer chains (Step 4)
│           └── <history_sha>/
│               ├── before/
│               ├── after/
│               ├── fixing/
│               └── graph.json                  # Heterogeneous graph — history commit (Step 7)
├── joern_output/                               # Joern CPG dot files (Steps 2 / 6)
│   └── linux/
│       └── test42/
│           └── <fix_sha>/
│               ├── before/                     # one .dot per method
│               └── after/
├── repositories/
│   └── linux/
│       ├── compile_commands.json               # Bear output (Step 0)
│       └── ...                                 # Full kernel source + git history
└── mapping/
    └── joern_targets.txt                       # Generated list of Joern source→output pairs
```

> The structure is identical for each supported project (`linux`, `FFmpeg`, `ImageMagick`, `OpenSSL`, `PHP-SRC`).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `libclang not found` | `GRAPHGEN_LIBCLANG` not set | See [Environment Setup §5](#5-set-the-libclang-environment-variable) |
| `joern: command not found` | Joern not in PATH | Add `~/bin/joern/joern-cli` to PATH |
| `gumtree: No such file` | GumTree not built | Run `./gradlew :dist:installShadowDist` in `mapping/gumtree/` |
| `[SKIP] joern dirs empty` | Step 2 not run / failed | Re-run `generate_joern.py` and check Joern logs |
| Wrong Python version | System Python overrides conda | Remove `~/bin/python*` symlinks; see §3 above |
| Bear produces empty JSON | Kernel not configured | Run `make defconfig` before Bear-wrapped build |
| `UnsupportedClassVersionError` from Joern, or Joern produces no output at all | System `java` is 11, Joern needs 21 | `conda activate tracevic` (ships JDK 21) or put `openjdk-21` first on `PATH` |
| `java.nio.charset.MalformedInputException` at Joern startup | `LANG` unset, JVM defaults to ASCII | `export LANG=C.UTF-8 LC_ALL=C.UTF-8` — see [Environment Setup §4](#4-set-a-utf-8-locale-required-by-joern) |
| `[SKIP] graph already exists` | A `graph.json` is already present | Both `--mode fix` and `--mode history` skip commits that already have one. Delete the file to force a rebuild. |
| `[SKIP] missing commits.json` in Step 7 | Step 4b not run | Steps 5–7 all read `commits.json`; run Steps 4 → 4b first |