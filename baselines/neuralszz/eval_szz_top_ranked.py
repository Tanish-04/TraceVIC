"""
NeuralSZZ + SZZ Pipeline Evaluation.

After training, this script:
  1. Loads the trained HAN + RankNet model
  2. Ranks deletion lines by model score
  3. Runs B-SZZ / V-SZZ / AG-SZZ on the top-k ranked lines
  4. Computes Precision / Recall / F1 against ground truth

Usage:
    python eval_szz_top_ranked.py --seed 456 --top-k 3
    python eval_szz_top_ranked.py --seed 456 --bszz-only --eval-split test
    python eval_szz_top_ranked.py --seed 456 --top-k 3 --agszz
"""

import argparse
import json
import os
import sys
import gc
import traceback
from typing import Dict, List, Optional, Set, Tuple

import torch
from functools import cmp_to_key
from sklearn.model_selection import train_test_split

from config import (
    SZZ_DIR, REPOS_DIR, GRAPH_DATA_DIR, NEURALSZZ_DATA_DIR,
    PROJECT_TO_REPO,
)

# Add SZZ implementations to path
sys.path.insert(0, SZZ_DIR)
from szz.b_szz import BaseSZZ
from szz.ag_szz import AGSZZ
from szz.v_szz import VSZZ
from szz.core.abstract_szz import ImpactedFile

from model import HAN, rankNet
from genPyG import get_graph_data
from genPairs import get_dir_to_minigraphs
from eval import (
    get_true_cid_map,
    get_score,
    cmp,
    eval_top_metrics,
)


# Path constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Defaults point at the 755-case linux set. main() overrides them from
# --cases-file / --minigraphs so an external dataset (e.g. the cross-project
# generalizability set) can be evaluated with the same trained checkpoints.
ALL_CASES_FILE = str(NEURALSZZ_DATA_DIR / "test_cases.json")
MINI_GRAPHS_FILE = str(NEURALSZZ_DATA_DIR / "miniGraphs_bszz.json")


def _results_dir_for_seed(seed: int) -> str:
    return os.path.join(SCRIPT_DIR, f"results_szz_pipeline_seed{seed}")


def _resolve_checkpoints(
    seed: int,
    han: Optional[str],
    ranknet: Optional[str],
) -> Tuple[str, str]:
    """Default checkpoint paths aligned with train.py output."""
    if han and ranknet:
        return han, ranknet
    if han:
        d = os.path.dirname(han)
        return han, ranknet or os.path.join(d, "ranknet_best.pt")
    if ranknet:
        d = os.path.dirname(ranknet)
        return han or os.path.join(d, "han_best.pt"), ranknet

    base = os.path.join(SCRIPT_DIR, f"checkpoints_baseline_seed{seed}")
    return (
        os.path.join(base, "han_best.pt"),
        os.path.join(base, "ranknet_best.pt"),
    )


# Model loading & ranking

def load_model(device, han_path, ranknet_path, sample_mini_graph_data):
    """Instantiate HAN and RankNet from saved checkpoints."""
    sample_pyg = get_graph_data(sample_mini_graph_data)
    metadata = sample_pyg.metadata()

    han_model = HAN(
        device, in_channels=768, out_channels=768 * 2,
        metadata=metadata, heads=2, dropout=0.3,
        num_bert_layers_freeze=8,
    )
    han_model.load_state_dict(torch.load(han_path, map_location=device))
    han_model = han_model.to(device)

    ranknet_model = rankNet(768 * 2)
    ranknet_model.load_state_dict(torch.load(ranknet_path, map_location=device))
    ranknet_model = ranknet_model.to(device)

    han_model.eval()
    ranknet_model.eval()
    return han_model, ranknet_model


def rank_deletion_lines(test_cases, all_mini_graphs, han_model, ranknet_model, device):
    """Score every deletion-line mini-graph and sort descending per test case."""
    sub = {k: v for k, v in all_mini_graphs.items() if k in set(test_cases)}
    dir_to_minigraphs = get_dir_to_minigraphs(sub)

    han_model.eval()
    ranknet_model.eval()
    with torch.no_grad():
        for fdir in test_cases:
            if fdir not in dir_to_minigraphs:
                continue
            for mg in dir_to_minigraphs[fdir]:
                score = get_score(mg.pyg, han_model, ranknet_model, device).to("cpu")
                mg.score = score
            dir_to_minigraphs[fdir].sort(key=cmp_to_key(cmp))

    return dir_to_minigraphs


# Line info extraction

def _line_info_from_node(fdir: str, node: dict) -> dict:
    """Extract line metadata from a mini graph node for SZZ execution."""
    # fdir is composite key like "linux/CVE-2020-1234"
    project = fdir.split("/")[0]
    repo_name = PROJECT_TO_REPO.get(project, project)

    # Read fix commit from info.json
    info_path = GRAPH_DATA_DIR / fdir / "info.json"
    fix_sha = ""
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        fix_sha = info.get("fix", "")

    # Keep the underscore fName; the real git path is resolved at SZZ-run time
    # from the fix commit's own diff (get_impacted_files), matching how
    # baselines/SZZ does it -- no git ls-tree, no underscore guessing.
    fname = node.get("fName", "")

    return {
        "fdir": fdir,
        "project": project,
        "repo_name": repo_name,
        "fix_sha": fix_sha,
        "fname": fname,
        "file_path": fname.replace("_", "/"),   # fallback only
        "line_beg": node.get("lineBeg", 0),
        "line_end": node.get("lineEnd", 0),
        "rootcause": node.get("rootcause", False),
        "code": node.get("code", ""),
    }


def extract_all_topk_lines(
    test_cases: List[str],
    dir_to_minigraphs: dict,
    k: int,
) -> Dict[str, List[dict]]:
    """For each test case, return up to k line_info dicts (rank order preserved)."""
    results: Dict[str, List[dict]] = {}
    skipped = []

    for fdir in test_cases:
        if fdir not in dir_to_minigraphs or not dir_to_minigraphs[fdir]:
            results[fdir] = []
            skipped.append(fdir)
            continue

        lines = []
        for mg in dir_to_minigraphs[fdir][:k]:
            node = mg.g[0]
            lines.append(_line_info_from_node(fdir, node))
        results[fdir] = lines

    if skipped:
        print(f"  [line_ranker] {len(skipped)} cases with no top-{k} lines")
    return results


# SZZ runners (operate on extracted line_info dicts)

def _run_szz_single_line(line_info: dict, szz_class, repos_dir: str) -> List[str]:
    """Run a single SZZ variant on one line and return list of BIC SHAs."""
    repo_name = line_info["repo_name"]
    fix_sha = line_info["fix_sha"]
    file_path = line_info["file_path"]
    line_beg = line_info["line_beg"]
    line_end = line_info["line_end"]

    if not fix_sha or not file_path:
        return []

    mod_lines = list(range(line_beg, line_end + 1))
    impacted = ImpactedFile(file_path, mod_lines)

    # use_temp_dir=False is REQUIRED: the default copies the whole repo (15 GB
    # for linux) into a temp dir, and this constructs an SZZ object per line --
    # a 755x5xtop-k copy storm. SZZ is read-only here (blame only), so operating
    # on the shared clone in place is safe.
    szz = szz_class(repo_name, None, repos_dir, use_temp_dir=False)
    bug_commits = szz.find_bic(fix_sha, [impacted])
    return [c.hexsha for c in bug_commits]


def _run_szz_topk(
    lines_by_case: Dict[str, List[dict]],
    szz_class,
    szz_name: str,
    repos_dir: str,
) -> Dict[str, List[str]]:
    """Run an SZZ variant on each ranked line per case, union-dedup BICs."""
    results: Dict[str, List[str]] = {}
    total = len(lines_by_case)

    for idx, (fdir, line_infos) in enumerate(lines_by_case.items(), 1):
        acc: Set[str] = set()
        if not line_infos:
            results[fdir] = []
            continue

        err_last = None
        for line_info in line_infos:
            try:
                acc.update(_run_szz_single_line(line_info, szz_class, repos_dir))
            except Exception:
                err_last = traceback.format_exc().splitlines()[-1]

        results[fdir] = sorted(acc)
        if results[fdir]:
            print(f"  [{szz_name}] {idx}/{total} {fdir}: "
                  f"{len(results[fdir])} unique BIC(s) from {len(line_infos)} line(s)")
        elif err_last:
            print(f"  [{szz_name}] {idx}/{total} {fdir}: ERROR — {err_last}")
        else:
            print(f"  [{szz_name}] {idx}/{total} {fdir}: "
                  f"0 BIC from {len(line_infos)} line(s)")

    return results


# I/O helpers
def _save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print(f"  Saved → {path}")


def _load(path: str):
    with open(path) as fh:
        return json.load(fh)


# Step 1 — Ranking

def step1_rank(
    rank_cases: List[str],
    all_mini_graphs: dict,
    device: str,
    out_path: str,
    han_checkpoint: str,
    ranknet_checkpoint: str,
    top_k: int,
    eval_split: str,
    skip: bool = False,
) -> Dict[str, List[dict]]:
    """Score deletion lines and extract top-k line metadata per ranked case."""

    if skip and os.path.exists(out_path):
        print(f"\n[Step 1] SKIPPED — loading cached predictions from {out_path}")
        raw = _load(out_path)
        if isinstance(raw, dict) and raw.get("schema_version") == 2:
            return {k: list(v) for k, v in raw.get("lines", {}).items()}
        # Legacy format
        return {k: ([] if v is None else [v]) for k, v in raw.items()}

    print(f"\n{'='*70}")
    print("[Step 1] Loading model and ranking deletion lines")
    print(f"{'='*70}")
    print(f"  Cases to rank : {len(rank_cases)}  (eval_split={eval_split!r}, top_k={top_k})")
    print(f"  Device        : {device}")

    # Find a sample mini graph for metadata
    sample_mg = None
    for tc in rank_cases:
        if tc in all_mini_graphs and len(all_mini_graphs[tc]) > 0:
            sample_mg = all_mini_graphs[tc][0]
            break
    if sample_mg is None:
        raise RuntimeError("No mini-graphs found for any case in the rank set.")

    han_model, ranknet_model = load_model(
        device, han_checkpoint, ranknet_checkpoint, sample_mg
    )
    print("  Model loaded from checkpoints.")

    dir_to_minigraphs = rank_deletion_lines(
        rank_cases, all_mini_graphs, han_model, ranknet_model, device
    )

    del han_model, ranknet_model
    gc.collect()
    torch.cuda.empty_cache()

    lines_by_case = extract_all_topk_lines(rank_cases, dir_to_minigraphs, top_k)

    payload = {
        "schema_version": 2,
        "top_k": top_k,
        "eval_split": eval_split,
        "lines": lines_by_case,
    }
    _save(payload, out_path)
    n_valid = sum(1 for v in lines_by_case.values() if v)
    print(f"  Top-{top_k} lines extracted for {n_valid}/{len(rank_cases)} cases.")
    return lines_by_case


# Step 2 — SZZ execution

def step2_run_szz(
    lines_by_case: Dict[str, List[dict]],
    szz_name: str,
    out_path: str,
    skip: bool = False,
) -> Dict[str, List[str]]:
    """Run one SZZ variant on every ranked line per case."""

    if skip and os.path.exists(out_path):
        print(f"\n[Step 2 {szz_name}] SKIPPED — loading cached BICs from {out_path}")
        return _load(out_path)

    print(f"\n{'='*70}")
    print(f"[Step 2] Running {szz_name} on ranked deletion lines")
    print(f"{'='*70}")
    print(f"  Repos dir  : {REPOS_DIR}")

    szz_classes = {
        "B-SZZ": BaseSZZ,
        "V-SZZ": VSZZ,
        "AG-SZZ": AGSZZ,
    }
    if szz_name not in szz_classes:
        raise ValueError(f"Unknown SZZ backend: {szz_name!r}")

    # Use the shared-instance runner, NOT _run_szz_topk. The latter passes
    # line_info["file_path"] straight to SZZ -- that path is the graph's
    # underscore-flattened name naively converted back ("crypto_rsa_rsa_ameth.c"
    # -> "crypto/rsa/rsa/ameth.c"), which is wrong for any file whose BASENAME
    # contains an underscore. blame on a nonexistent path returns no commits and
    # raises nothing, so the baseline silently scored 0 BIC on those cases
    # (16/20 OpenSSL, 10/19 PHP-SRC; FFmpeg/ImageMagick were mostly unaffected
    # because their basenames have no underscores). run_szz_topk_shared resolves
    # the real path from the fix commit's own diff via get_impacted_files, which
    # is what baselines/SZZ does, and reuses one SZZ instance per repo.
    bic_commits = run_szz_topk_shared(
        lines_by_case, szz_classes[szz_name], szz_name
    )

    _save(bic_commits, out_path)
    n_found = sum(1 for v in bic_commits.values() if v)
    print(f"  BIC found for {n_found}/{len(bic_commits)} test cases.")
    return bic_commits


# Step 3 — Metrics

def step3_metrics(
    all_cases: List[str],
    split_cases: Dict[str, List[str]],
    bic_commits: Dict[str, List[str]],
    true_cid_map: Dict,
    szz_name: str,
    out_path: str,
    eval_split_scope: str = "all",
) -> dict:
    """Compute and save pipeline metrics."""

    print(f"\n{'='*70}")
    print(f"[Step 3] Metrics — {szz_name}  (scope={eval_split_scope!r})")
    print(f"{'='*70}")

    results: dict = {}

    if eval_split_scope == "all":
        results["all"] = eval_top_metrics(all_cases, bic_commits, true_cid_map)
        for split_name, cases in split_cases.items():
            results[split_name] = eval_top_metrics(cases, bic_commits, true_cid_map)
    else:
        cases = split_cases[eval_split_scope]
        results[eval_split_scope] = eval_top_metrics(cases, bic_commits, true_cid_map)

    _save(results, out_path)

    for split_name, m in results.items():
        print(
            f"  [{split_name:5s}]  P/R/F1="
            f"{m['precision']:.4f}/{m['recall']:.4f}/{m['f1']:.4f}  "
            f"hits={m['total_hits']}  GT={m['total_gt']}  "
            f"pred={m['total_identified']}  N={m['n_cases']}"
        )

    return results


# Comparison table


def print_comparison(
    bszz_metrics: Optional[dict],
    vszz_metrics: Optional[dict],
    agszz_metrics: Optional[dict] = None,
) -> None:
    """Print pipeline metrics for whichever SZZ backends produced results."""

    methods: List[Tuple[str, dict]] = []
    if bszz_metrics is not None:
        methods.append(("NeuralSZZ + B-SZZ", bszz_metrics))
    if vszz_metrics is not None:
        methods.append(("NeuralSZZ + V-SZZ", vszz_metrics))
    if agszz_metrics is not None:
        methods.append(("NeuralSZZ + AG-SZZ", agszz_metrics))

    if not methods:
        return

    print(f"\n{'='*80}")
    print("  FINAL COMPARISON — " + "  vs  ".join(m[0] for m in methods))
    print(f"{'='*80}")

    splits = list(methods[0][1].keys())
    print(f"  {'Split':<8}  {'Method':<22}  {'P':>7}  {'R':>7}  {'F1':>7}")
    print("  " + "-" * 60)

    for split in splits:
        for i, (label, mdict) in enumerate(methods):
            m = mdict[split]
            split_col = split if i == 0 else ""
            print(
                f"  {split_col:<8}  {label:<22}"
                f"  {m['precision']:>7.4f}"
                f"  {m['recall']:>7.4f}"
                f"  {m['f1']:>7.4f}"
            )
        print("  " + "-" * 60)

    print(f"{'='*80}\n")


# Main

def parse_args():
    parser = argparse.ArgumentParser(
        description="NeuralSZZ + SZZ Pipeline Evaluation"
    )
    parser.add_argument("--seed", type=int, default=456,
                        help="Training split seed (default: 456)")
    parser.add_argument("--bszz-only", action="store_true",
                        help="Run only B-SZZ")
    parser.add_argument("--vszz-only", action="store_true",
                        help="Run only V-SZZ")
    parser.add_argument("--agszz-only", action="store_true",
                        help="Run only AG-SZZ")
    parser.add_argument("--agszz", action="store_true",
                        help="Also run AG-SZZ alongside B-SZZ and V-SZZ")
    parser.add_argument("--han", default=None,
                        help="Path to HAN checkpoint (.pt)")
    parser.add_argument("--ranknet", default=None,
                        help="Path to RankNet checkpoint (.pt)")
    parser.add_argument("--device", default=None,
                        help="PyTorch device (default: auto)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Run SZZ on top-k ranked lines per case (default: 3)")
    parser.add_argument(
        "--eval-split",
        choices=("all", "train", "val", "test", "val_test"),
        default="all",
        help="Which cases to rank and evaluate (default: all)",
    )
    parser.add_argument("--skip-ranking", action="store_true",
                        help="Skip Step 1; load cached line predictions")
    parser.add_argument("--skip-bszz", action="store_true",
                        help="Skip B-SZZ; load cached BICs")
    parser.add_argument("--skip-vszz", action="store_true",
                        help="Skip V-SZZ; load cached BICs")
    parser.add_argument("--skip-agszz", action="store_true",
                        help="Skip AG-SZZ; load cached BICs")
    parser.add_argument("--kfold", action="store_true",
                        help="Run 5-fold CV over the 755-case set (KFold seed=123), "
                             "loading checkpoints_755_NeuralSZZ/fold_i and "
                             "micro-averaging over folds. Ignores --seed/--eval-split.")
    parser.add_argument("--kfold-ckpt", default="checkpoints_755_NeuralSZZ",
                        help="Root of per-fold NeuralSZZ checkpoints for --kfold.")
    parser.add_argument("--cases-file", default=None,
                        help="Composite-key case list (default: data/test_cases.json)")
    parser.add_argument("--minigraphs", default=None,
                        help="miniGraphs JSON matching --cases-file "
                             "(default: data/miniGraphs_bszz.json)")
    parser.add_argument("--results-dir", default=None,
                        help="Override the results directory.")

    args = parser.parse_args()
    global ALL_CASES_FILE, MINI_GRAPHS_FILE
    if args.cases_file:
        ALL_CASES_FILE = os.path.abspath(args.cases_file)
    if args.minigraphs:
        MINI_GRAPHS_FILE = os.path.abspath(args.minigraphs)
    if (args.cases_file is None) != (args.minigraphs is None):
        parser.error("--cases-file and --minigraphs must be given together: a "
                     "case list without its matching miniGraphs ranks nothing.")
    if sum(bool(x) for x in (args.bszz_only, args.vszz_only, args.agszz_only)) > 1:
        parser.error("Use at most one of --bszz-only, --vszz-only, --agszz-only.")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    return args


def _extract_bics(bic, szz_name):
    """Normalise a find_bic() return into a set of commit SHAs.

    B-SZZ / AG-SZZ / MA-SZZ return Set[Commit] (use .hexsha). V-SZZ returns a
    list of per-line records, each with a 'previous_commits' chain; the
    prediction is the INTRODUCER (deepest = previous_commits[-1]), matching how
    baselines/SZZ/evaluate.py scores V-SZZ.
    """
    out = set()
    if szz_name == "V-SZZ":
        for record in bic:
            prev = record.get("previous_commits") if isinstance(record, dict) else None
            if prev:
                out.add(prev[-1]["commit"])
    else:
        for c in bic:
            out.add(c.hexsha)
    return out


def run_szz_topk_shared(lines_by_case, szz_class, szz_name):
    """Run one SZZ variant on the top-k ranked lines, union-dedup BICs per case.

    Speed fixes over the single-split path (which was ~hours):
      * ONE SZZ instance per (variant, repo), reused across all cases/lines,
        instead of constructing per line (each ctor opens the repo).
      * Real file paths come from get_impacted_files(fix_sha) -- the fix commit's
        own diff (mod.old_path/new_path) -- not git ls-tree. Matches how
        baselines/SZZ resolves paths; no underscore guessing.
    use_temp_dir=False: operate on the shared clone in place (blame is read-only).
    """
    results = {}
    szz_by_repo = {}
    impacted_cache = {}                       # fix_sha -> {underscore_path: real_path}
    total = len(lines_by_case)

    for idx, (fdir, line_infos) in enumerate(lines_by_case.items(), 1):
        acc = set()
        if line_infos:
            repo_name = line_infos[0]["repo_name"]
            fix_sha = line_infos[0]["fix_sha"]
            szz = szz_by_repo.get(repo_name)
            if szz is None:
                szz = szz_class(repo_name, None, REPOS_DIR, use_temp_dir=False)
                szz_by_repo[repo_name] = szz

            if fix_sha not in impacted_cache:
                try:
                    imp = szz.get_impacted_files(fix_sha, only_deleted_lines=True)
                    impacted_cache[fix_sha] = {f.file_path.replace("/", "_"): f.file_path for f in imp}
                except Exception as exc:
                    # Was a bare `pass`. A swallowed failure here is indistinguishable
                    # from "SZZ legitimately found nothing", which silently understates
                    # the baseline -- surface it instead.
                    print(f"  [{szz_name}] {fdir}: get_impacted_files FAILED: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    impacted_cache[fix_sha] = {}
            path_map = impacted_cache[fix_sha]

            for li in line_infos:
                real_path = path_map.get(li["fname"], li["file_path"])   # diff path, else naive fallback
                if not fix_sha or not real_path:
                    continue
                mod_lines = list(range(li["line_beg"], li["line_end"] + 1))
                try:
                    bic = szz.find_bic(fix_sha, [ImpactedFile(real_path, mod_lines)])
                    acc.update(_extract_bics(bic, szz_name))
                except Exception as exc:
                    print(f"  [{szz_name}] {fdir}: find_bic FAILED on {real_path}"
                          f":{mod_lines}: {type(exc).__name__}: {exc}", flush=True)
        results[fdir] = sorted(acc)
        if idx % 25 == 0 or idx == total:
            print(f"  [{szz_name}] {idx}/{total} cases", flush=True)
    return results


def run_kfold(args, device):
    """5-fold CV over the 755 set (folds identical to main_kfold.py), running the
    chosen SZZ backends on the top-k RankNet-ranked lines and micro-averaging.

    'Line ranking reflects commit ranking': the prediction for a case is the
    union of the top-k ranked lines' SZZ commits (no learned commit ranker).
    """
    import numpy as np
    from sklearn.model_selection import KFold

    methods = ["B-SZZ", "V-SZZ"]
    if args.bszz_only:
        methods = ["B-SZZ"]
    elif args.vszz_only:
        methods = ["V-SZZ"]
    if args.agszz:
        methods.append("AG-SZZ")

    all_mg = _load(MINI_GRAPHS_FILE)
    sample = next((v[0] for v in all_mg.values() if v), None)

    data_root = str(GRAPH_DATA_DIR / "linux")
    bare = sorted(d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)))
    arr = np.array(bare)
    kf = KFold(n_splits=5, shuffle=True, random_state=123)

    composite_all = [f"linux/{c}" for c in bare]
    true_cid_map = get_true_cid_map(composite_all)
    results_dir = os.path.join(SCRIPT_DIR, "results_szz_pipeline_kfold")
    os.makedirs(results_dir, exist_ok=True)

    szz_classes = {"B-SZZ": BaseSZZ, "V-SZZ": VSZZ, "AG-SZZ": AGSZZ}

    agg = {m: [0, 0, 0] for m in methods}          # tp, identified, gt
    for fi, (_, test_idx) in enumerate(kf.split(arr)):
        test_cases = [f"linux/{c}" for c in arr[test_idx].tolist()]
        han_p = os.path.join(SCRIPT_DIR, args.kfold_ckpt, f"fold_{fi}", "han_best.pt")
        rk_p = os.path.join(SCRIPT_DIR, args.kfold_ckpt, f"fold_{fi}", "ranknet_best.pt")
        print(f"\n{'='*70}\n  FOLD {fi + 1}/5  test={len(test_cases)}\n{'='*70}", flush=True)

        han, rk = load_model(device, han_p, rk_p, sample)
        d2m = rank_deletion_lines(test_cases, all_mg, han, rk, device)
        lines_by_case = extract_all_topk_lines(test_cases, d2m, args.top_k)

        for m in methods:
            bic = run_szz_topk_shared(lines_by_case, szz_classes[m], m)
            met = eval_top_metrics(test_cases, bic, true_cid_map)
            agg[m][0] += met["total_hits"]
            agg[m][1] += met["total_identified"]
            agg[m][2] += met["total_gt"]
            print(f"  [fold {fi+1}] {m}: P={met['precision']:.3f} R={met['recall']:.3f} F1={met['f1']:.3f}")
        del han, rk
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*70}\n  MICRO over 5 folds (top-{args.top_k})\n{'='*70}")
    summary = {}
    for m in methods:
        tp, idn, gt = agg[m]
        p = tp / idn if idn else 0.0
        r = tp / gt if gt else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        summary[m] = {"precision": p, "recall": r, "f1": f, "tp": tp, "identified": idn, "gt": gt}
        print(f"  {m:8s}: P={p:.3f} R={r:.3f} F1={f:.3f}  (tp/ident/gt={tp}/{idn}/{gt})")
    _save(summary, os.path.join(results_dir, f"kfold_summary_top{args.top_k}.json"))


def main():
    args = parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.kfold:
        run_kfold(args, device)
        return

    results_dir = args.results_dir or _results_dir_for_seed(args.seed)
    han_ckpt, rank_ckpt = _resolve_checkpoints(args.seed, args.han, args.ranknet)

    # Determine which SZZ backends to run
    if args.bszz_only:
        do_bszz, do_vszz, do_agszz = True, False, False
    elif args.vszz_only:
        do_bszz, do_vszz, do_agszz = False, True, False
    elif args.agszz_only:
        do_bszz, do_vszz, do_agszz = False, False, True
    else:
        do_bszz, do_vszz, do_agszz = True, True, bool(args.agszz)

    print("=" * 80)
    print("  NeuralSZZ + SZZ Pipeline Evaluation")
    print("=" * 80)
    print(f"  Device       : {device}")
    print(f"  Seed         : {args.seed}")
    print(f"  Top-k        : {args.top_k}")
    print(f"  Eval split   : {args.eval_split}")
    print(f"  Checkpoints  : {han_ckpt}")
    print(f"  Results dir  : {results_dir}")

    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("\n[Data] Loading test cases and mini-graphs...")
    with open(ALL_CASES_FILE) as fh:
        all_cases = json.load(fh)
    print(f"  All cases: {len(all_cases)}")

    # 70/15/15 split matching train.py
    train_cases, temp = train_test_split(
        all_cases, test_size=0.3, random_state=args.seed
    )
    val_cases, test_cases = train_test_split(
        temp, test_size=0.5, random_state=args.seed
    )
    split_cases: Dict[str, List[str]] = {
        "train": train_cases,
        "val": val_cases,
        "test": test_cases,
    }
    if args.eval_split == "val_test":
        split_cases["val_test"] = list(val_cases) + list(test_cases)

    for name in ("train", "val", "test"):
        print(f"  {name:5s} split: {len(split_cases[name])} cases")

    print(f"\n[Data] Loading miniGraphs_bszz.json...")
    with open(MINI_GRAPHS_FILE) as fh:
        all_mini_graphs = json.load(fh)
    print(f"  Mini-graphs for {sum(1 for c in all_cases if c in all_mini_graphs)}/{len(all_cases)} cases")

    print(f"\n[Data] Loading ground-truth inducing commits...")
    true_cid_map = get_true_cid_map(all_cases)

    # Determine which cases to rank
    if args.eval_split == "all":
        rank_cases = all_cases
    else:
        rank_cases = split_cases[args.eval_split]

    metrics_scope = "all" if args.eval_split == "all" else args.eval_split
    pred_path = os.path.join(
        results_dir, f"line_predictions_top{args.top_k}_eval_{args.eval_split}.json"
    )

    # Step 1: Rank deletion lines
    lines_by_case = step1_rank(
        rank_cases, all_mini_graphs, device, pred_path,
        han_ckpt, rank_ckpt,
        top_k=args.top_k, eval_split=args.eval_split,
        skip=args.skip_ranking,
    )

    # Step 2-3: Run SZZ variants and compute metrics
    bszz_metrics = vszz_metrics = agszz_metrics = None

    if do_bszz:
        bszz_bic = step2_run_szz(
            lines_by_case, "B-SZZ",
            os.path.join(results_dir, "bszz_bic_commits.json"),
            skip=args.skip_bszz,
        )
        bszz_metrics = step3_metrics(
            all_cases, split_cases, bszz_bic, true_cid_map, "B-SZZ",
            os.path.join(results_dir, "bszz_metrics.json"),
            eval_split_scope=metrics_scope,
        )

    if do_vszz:
        vszz_bic = step2_run_szz(
            lines_by_case, "V-SZZ",
            os.path.join(results_dir, "vszz_bic_commits.json"),
            skip=args.skip_vszz,
        )
        vszz_metrics = step3_metrics(
            all_cases, split_cases, vszz_bic, true_cid_map, "V-SZZ",
            os.path.join(results_dir, "vszz_metrics.json"),
            eval_split_scope=metrics_scope,
        )

    if do_agszz:
        agszz_bic = step2_run_szz(
            lines_by_case, "AG-SZZ",
            os.path.join(results_dir, "agszz_bic_commits.json"),
            skip=args.skip_agszz,
        )
        agszz_metrics = step3_metrics(
            all_cases, split_cases, agszz_bic, true_cid_map, "AG-SZZ",
            os.path.join(results_dir, "agszz_metrics.json"),
            eval_split_scope=metrics_scope,
        )

    # Comparison table
    _save(
        {"seed": args.seed, "top_k": args.top_k, "eval_split": args.eval_split,
         "bszz": bszz_metrics, "vszz": vszz_metrics, "agszz": agszz_metrics},
        os.path.join(results_dir, "comparison_table.json"),
    )

    print_comparison(bszz_metrics, vszz_metrics, agszz_metrics)
    print(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()
