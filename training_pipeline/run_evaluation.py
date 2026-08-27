"""
Evaluate the trained TraceVIC models — on their own k-fold test splits, or on
any external dataset built through the same graph-construction pipeline.

Reports BOTH stages of the pipeline:
  * commit ranking (VIC)      @1/2/3  -- the final prediction
  * deletion-line ranking     @1/2/3  -- Phase 1's root-cause line identification

Scope: OUR trained models only. The SZZ baselines are evaluated by
baselines/SZZ/evaluate.py and NeuralSZZ by its own harness in
baselines/neuralszz/ -- deliberately not duplicated here.

--------------------------------------------------------------------------------
MODE 1 — K-FOLD RESULTS (default; fast, GPU-free)
--------------------------------------------------------------------------------
Commit ranking is read straight from each fold's
<ckpt>/fold_<i>/fold_results.json (written by main_kfold.py via
training.evaluation.evaluate_global). Micro-averaged:
    P = Σ hits / Σ identified,  R = Σ hits / Σ gt,  F1 = harmonic mean.

Deletion-line ranking is a Phase-1 property, so it must be computed by loading
each fold's Phase 1 model, ranking the fold's test deletion lines, and applying
the SAME metric Phase 1 already uses (training.evaluation.evaluate_topk_metrics)
at k=1/2/3. Run once with --compute-line-level; results are cached to
    <ckpt>/line_metrics.json   (per-fold [tp, fp, gt] for k=1,2,3)
and read from there afterwards.

--------------------------------------------------------------------------------
MODE 2 — EXTERNAL DATASET (--data-root; needs GPU)
--------------------------------------------------------------------------------
Runs the full Phase 1 → top-k lines → Phase 2 pipeline over every case in a
dataset the models never trained on (e.g. the cross-project generalization set),
using the SAME evaluate_global that produced the k-fold numbers, so the two
modes are directly comparable.

Because an external dataset is unseen by every fold, all N_FOLDS checkpoints are
valid evaluators for all of its cases. Each fold is run over the whole dataset
and results are reported as mean ± sd, which doubles as a stability estimate —
important at the small n these sets tend to have.

Metrics are defined exactly as in mode 1 — same evaluate_global, so a case's
prediction is its top-k ranked slots de-duplicated by SHA, precision is micro
Sigma-hits / Sigma-identified and recall Sigma-hits / Sigma-gt. Nothing about
the evaluation changes between the two modes; only the data does.

Usage:
    python run_evaluation.py                              # k-fold, commit + cached line
    python run_evaluation.py --arm gat_temporal
    python run_evaluation.py --compute-line-level         # (re)compute line-level
    python run_evaluation.py --k-lines 3                  # top_k_lines block (default 3)

    # external dataset (one --prebuilt-dir per --data-root, or a single shared one)
    python run_evaluation.py --arm gat_temporal \
        --data-root ../graph_construction/data/FFmpeg \
                    ../graph_construction/data/OpenSSL \
        --prebuilt-dir temporal_graph_gen/FFmpeg temporal_graph_gen/OpenSSL

Arm keys: gat_temporal, gat_notemporal, bszz_candidates
"""
import argparse
import json
import os
from pathlib import Path
from statistics import mean, pstdev

from config_utils import ConfigManager

HERE = Path(__file__).resolve().parent

# Fold count and seed MUST match what main_kfold.py trained with, or the
# reconstructed splits silently differ from the ones the checkpoints saw.
# Both are therefore read from the same YAML rather than hardcoded here.
_DEFAULTS = ConfigManager().raw["defaults"]
N_FOLDS = _DEFAULTS["n_folds"]
SEED = _DEFAULTS["seed"]


# ─────────────────────────────── arm registry ─────────────────────────────────
# arm key -> (display name, checkpoint dir, run settings).
#
# The settings are what a checkpoint needs to be rebuilt and loaded correctly.
# They are declared here because the k-fold checkpoint dirs do NOT persist the
# resolved config (fold_results.json holds metrics and the split only), so this
# dict is the only record of how each arm was run.
#
#   encoder     model.encoder_type
#   graph_mode  paths.graph_mode — also selects the prebuilt subdirectory
#   model_pe    model.use_temporal_pe   (Phase 1 encoder sinusoidal PE)
#   phase2_pe   phase2.use_temporal_pe  (Phase 2 distance embedding)
#
# model_pe and phase2_pe are set together per arm: an arm labelled "- temporal"
# disabled the positional signal in both stages. Use --verify-folds to confirm a
# reconstruction reproduces that arm's logged fold numbers before trusting it on
# new data — a wrong setting here loads real weights under the wrong positional
# signal, which degrades silently rather than raising.
# Default checkpoint directory per arm. Override on the command line with
#   --ckpt-dir gat_temporal=/path/to/dir [...]
# so a run started with `main_kfold.py --save-dir X` can be evaluated without
# editing this file.
ARMS = {
    "gat_temporal":     ("GAT + temporal (TraceVIC)", "checkpoints_755_GAT_Temporal",
                         dict(encoder="gat", graph_mode="full_graph",
                              model_pe=True, phase2_pe=True)),
    "gat_notemporal":   ("GAT - temporal", "checkpoints_755_GAT_NoTemporal",
                         dict(encoder="gat", graph_mode="no_temporal",
                              model_pe=False, phase2_pe=False)),
    "bszz_candidates":  ("Variant B (B-SZZ cands)", "checkpoints_755_BSZZ_Candidates",
                         dict(encoder="gat", graph_mode="bszz_candidates",
                              model_pe=True, phase2_pe=True)),
}


# ───────────────────────── shared inference scaffolding ───────────────────────
def arm_config(arm_key, data_root=None, prebuilt_dir=None):
    """
    Resolved config for one arm, plus the (data_root, prebuilt_dir) it runs on.

    Everything not listed in the arm registry is inherited from
    training_config.yaml, so an arm cannot silently drift from the trained
    hyper-parameters (hidden_dim, top_k_lines, ...). ``data_root`` and
    ``prebuilt_dir`` default to the config values and are overridden per call
    for external datasets.
    """

    _, ckpt_dir, spec = ARMS[arm_key]
    cfg = ConfigManager().raw
    cfg["model"]["encoder_type"] = spec["encoder"]
    cfg["model"]["use_temporal_pe"] = spec["model_pe"]
    cfg["phase2"]["use_temporal_pe"] = spec["phase2_pe"]
    cfg["paths"]["graph_mode"] = spec["graph_mode"]
    if data_root:
        cfg["paths"]["data_root"] = str(data_root)
    if prebuilt_dir:
        cfg["paths"]["prebuilt_dir"] = str(prebuilt_dir)
    return cfg, HERE / ckpt_dir, spec


def load_weights(model, path, device, allow_missing_buffers=True):
    """
    Load a saved checkpoint into ``model``, verifying nothing important is lost.

    Checkpoints are wrapped dicts (main.py saves {"model_state_dict": ...}), so
    the payload is unwrapped first. Loading is then validated rather than
    trusted: ``strict=False`` alone would silently accept a state dict from a
    different architecture and produce a randomly-initialised model that still
    returns plausible-looking scores. Missing NON-PERSISTENT BUFFERS are the one
    legitimate mismatch (e.g. the sinusoidal temporal PE, which is recomputed at
    construction and never saved); anything else raises.
    """
    import torch

    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "best_commit_ranker_state"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    result = model.load_state_dict(obj, strict=False)
    buffers = {name for name, _ in model.named_buffers()}
    lost = [k for k in result.missing_keys
            if not (allow_missing_buffers and k in buffers)]
    if lost or result.unexpected_keys:
        raise RuntimeError(
            "checkpoint does not match the model built for this arm: %s\n"
            "  missing parameters : %s\n"
            "  unexpected keys    : %s\n"
            "Check the arm's encoder / graph_mode / PE settings in ARMS."
            % (path, lost[:8], result.unexpected_keys[:8]))
    return model


def dataset_cases(data_root, prebuilt_dir, graph_mode):
    """
    Cases that have BOTH a data directory and pre-built graphs, plus the rest.

    A case with no ``<prebuilt_dir>/<graph_mode>/<case>/`` produced no deletion
    lines (addition-only patch, failed extraction, missing graph_vszz.json) and
    would silently score zero if evaluated. Returning the excluded names lets
    the caller report them instead of burying them in the denominator.
    """
    data_root, prebuilt_dir = Path(data_root), Path(prebuilt_dir)
    if not data_root.is_dir():
        raise FileNotFoundError("data root not found: %s" % data_root)
    mode_dir = prebuilt_dir / graph_mode
    if not mode_dir.is_dir():
        raise FileNotFoundError(
            "no pre-built graphs at %s — run build_graphs.py --mode %s "
            "--data_path %s --output_dir %s" % (mode_dir, graph_mode,
                                                data_root, prebuilt_dir))
    everything = sorted(d for d in os.listdir(data_root) if (data_root / d).is_dir())
    usable = [d for d in everything if (mode_dir / d).is_dir()]
    return usable, [d for d in everything if d not in set(usable)]


def fold_splits(cases):
    """The k-fold test splits, reconstructed exactly as main_kfold.py builds them."""
    import numpy as np
    from sklearn.model_selection import KFold

    arr = np.array(cases)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return [arr[te].tolist() for _, te in kf.split(arr)]


def rank_deletion_lines(cfg, ckpt_dir, fold, dataset, cases, device):
    """Phase 1: score every deletion line of ``cases`` with one fold's model."""
    import torch
    from training.utils import build_phase1_model, coerce_idx

    model = build_phase1_model(cfg, device)
    load_weights(model, ckpt_dir / ("fold_%d" % fold) / "phase1_best.pt", device)
    model.eval()

    graphs = dataset.get_mini_graphs_dict()
    ranked = {}
    with torch.no_grad():
        for name in cases:
            mgs = graphs.get(name) or []
            for mg in mgs:
                try:
                    g = mg.pyg.to(device)
                    idx = coerce_idx(mg.del_idx)
                    mg.score = model.predict(g, idx).item() if idx < g.num_nodes else 0.0
                except Exception:
                    mg.score = 0.0
            ranked[name] = sorted(mgs, key=lambda x: x.score, reverse=True)

    del model
    torch.cuda.empty_cache()
    return ranked


# ───────────────────── commit ranking — k-fold (from JSON) ────────────────────
def commit_micro(ckpt_dir, k_lines, at_k):
    """Micro (P, R, F1) over folds from fold_results.json, or None if incomplete."""
    d = HERE / ckpt_dir
    hits = ident = gt = 0
    for i in range(N_FOLDS):
        fr = d / f"fold_{i}" / "fold_results.json"
        if not fr.exists():
            return None
        p2 = json.load(open(fr))["phase2"]
        m = p2["test_metrics_by_k"][str(k_lines)][str(at_k)]
        hits += m["total_hits"]; ident += m["total_identified"]; gt += m["total_gt"]
    P = hits / ident if ident else 0.0
    R = hits / gt if gt else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


# ───────────────────── deletion-line ranking (GPU; cached) ────────────────────
def compute_line_level(arm_key, data_root=None, prebuilt_dir=None):
    """Load each fold's Phase 1, rank test lines, score @1/2/3, cache raw counts."""
    import torch
    from data_processing.dataset import DeletionLineDataset
    from models.shared_encoder import UnixcoderEmbedder
    from training.evaluation import evaluate_topk_metrics, load_true_commit_map

    cfg, ckpt_dir, spec = arm_config(arm_key, data_root, prebuilt_dir)
    root, prebuilt = cfg["paths"]["data_root"], cfg["paths"]["prebuilt_dir"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedder = UnixcoderEmbedder(
        model_name=cfg["model"].get("unixcoder_model_name",
                                    "microsoft/unixcoder-base-nine"),
        max_len=cfg["model"].get("max_seq_len", 64),
        tokenizer_only=True)

    cases, _ = dataset_cases(root, prebuilt, spec["graph_mode"])
    per_fold = []
    for fold, test_cases in enumerate(fold_splits(cases)):
        ds = DeletionLineDataset(root, test_cases, embedder,
                                 prebuilt_dir=prebuilt, graph_mode=spec["graph_mode"])
        ranked = rank_deletion_lines(cfg, ckpt_dir, fold, ds, test_cases, device)
        true_cid = load_true_commit_map(test_cases, root)

        counts = {}
        for k in (1, 2, 3):
            r = evaluate_topk_metrics(ranked, true_cid, root, k=k)
            counts[str(k)] = [r[f"tp@{k}"], r[f"fp@{k}"], r["total_inducing_commits"]]
        per_fold.append(counts)
        print("    fold %d: %s" % (fold + 1,
              "  ".join("@%d %s" % (k, counts[str(k)]) for k in (1, 2, 3))))

    json.dump(per_fold, open(ckpt_dir / "line_metrics.json", "w"), indent=2)


def line_micro(ckpt_dir, at_k):
    """Micro (P, R, F1) over folds from cached line_metrics.json, or None."""
    f = HERE / ckpt_dir / "line_metrics.json"
    if not f.exists():
        return None
    folds = json.load(open(f))
    tp = sum(x[str(at_k)][0] for x in folds)
    fp = sum(x[str(at_k)][1] for x in folds)
    gt = sum(x[str(at_k)][2] for x in folds)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / gt if gt else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


# ──────────────────────── external-dataset evaluation ─────────────────────────
def evaluate_dataset(arm_key, data_root, prebuilt_dir, k_lines, cases_by_fold=None):
    """
    Run every fold checkpoint over a whole external dataset.

    ``cases_by_fold`` restricts each fold to its own subset instead of the whole
    dataset. That is only correct when the data IS the training data — it is how
    --verify-folds reproduces the logged k-fold numbers. Leave it None for an
    external dataset, where every fold is unseen on every case.

    Returns (per-fold results_by_k, per-fold decompositions, excluded cases).
    Headline metrics come from training.evaluation.evaluate_global — the same
    function that produced the k-fold numbers — so the two modes are comparable
    and the SHA-level de-duplication is not reimplemented here. The
    decomposition needs per-case pool sizes and argmax positions, which
    evaluate_global does not expose, so it is derived from a second pass over
    the (already built) Phase 2 items.
    """
    import torch
    from data_processing.dataset import DeletionLineDataset
    from models.shared_encoder import UnixcoderEmbedder
    from training.embedding_cache import score_deletion_lines, build_phase2_items
    from training.evaluation import evaluate_global, load_true_commit_map
    from training.utils import build_phase2_model, setup_device

    cfg, ckpt_dir, spec = arm_config(arm_key, data_root, prebuilt_dir)
    # Path, not str: evaluate_global joins it with `/`, as main_kfold.py does.
    root = Path(cfg["paths"]["data_root"])
    prebuilt = cfg["paths"]["prebuilt_dir"]
    device = setup_device(cfg["defaults"]["gpu_id"])

    cases, excluded = dataset_cases(root, prebuilt, spec["graph_mode"])
    if not cases:
        raise RuntimeError("no evaluable cases in %s" % root)

    # tokenizer_only=True is required, not a choice: build_pyg stores token ids
    # for the model's own UniXcoder to embed, and its other branch calls an
    # encode_texts() that does not exist on this embedder. Scripts that pass a
    # full embedder only survive because a cached .pt bypasses build_pyg — the
    # first dataset without caches raises.
    embedder = UnixcoderEmbedder(
        model_name=cfg["model"].get("unixcoder_model_name",
                                    "microsoft/unixcoder-base-nine"),
        max_len=cfg["model"].get("max_seq_len", 64),
        tokenizer_only=True)
    ds = DeletionLineDataset(root, cases, embedder,
                             prebuilt_dir=prebuilt, graph_mode=spec["graph_mode"])
    true_cid = load_true_commit_map(cases, root)

    metrics = []
    for fold in range(N_FOLDS):
        fold_cases = cases if cases_by_fold is None else cases_by_fold[fold]
        scored = score_deletion_lines(
            _phase1_for(cfg, ckpt_dir, fold, device), ds, fold_cases, device,
            max_nodes=cfg["defaults"]["max_nodes_per_batch"], top_k=k_lines)

        items = build_phase2_items(
            scored, fold_cases, graph_mode=spec["graph_mode"],
            top_k_lines=k_lines)[k_lines]
        item_map = {it["test_name"]: it for it in items if it.get("test_name")}

        model = build_phase2_model(cfg, device)
        load_weights(model, ckpt_dir / ("fold_%d" % fold) / "phase2_best.pt", device)
        model.eval()

        fold_metrics = evaluate_global(fold_cases, item_map, true_cid, model,
                                       device, root)
        cve = case_level_identified(fold_cases, item_map, true_cid, model,
                                    device, root, (1, 2, 3))
        for at_k, count in cve.items():
            fold_metrics[at_k]["cve_identified"] = count
            fold_metrics[at_k]["cve_total"] = sum(
                1 for c in fold_cases if true_cid.get(c))
        metrics.append(fold_metrics)

        del model, scored, items, item_map
        torch.cuda.empty_cache()
        print("    fold %d done" % fold)

    del ds, embedder
    torch.cuda.empty_cache()
    return metrics, excluded


def case_level_identified(cases, item_map, true_cid, model, device, root, k_values):
    """
    Per k, how many CASES have at least one ground-truth VIC in the top-k.

    evaluate_global counts COMMITS (a case with 3 GT VICs contributes 3 to
    total_gt), which is what precision and recall are built on. A paper table
    also wants the CVE-level view: was this CVE solved at all. Predictions come
    from the same _prefix_to_full_sha_map + get_ranked_commit_shas used by
    evaluate_global, so the two can never disagree about what was predicted.
    """
    import torch
    from training.evaluation import (_prefix_to_full_sha_map,
                                     get_ranked_commit_shas)

    model.eval()
    counts = {k: 0 for k in k_values}
    with torch.no_grad():
        for name in cases:
            gt_set = true_cid.get(name, set())
            item = item_map.get(name)
            if not gt_set or not item or not item.get("valid"):
                continue
            cidx = item["commit_indices"].to(device)
            scores = model(item["node_embeddings"].to(device), cidx,
                           item["is_temporal_node"].to(device),
                           commit_counts=[int(cidx.max().item()) + 1])
            if scores.dim() > 1:
                scores = scores.squeeze(-1)
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            expand = _prefix_to_full_sha_map(name, root)
            for k in k_values:
                if set(get_ranked_commit_shas(item, scores, k, expand)) & gt_set:
                    counts[k] += 1
    return counts


def _phase1_for(cfg, ckpt_dir, fold, device):
    """One fold's loaded Phase 1 model (score_deletion_lines calls .eval())."""
    from training.utils import build_phase1_model

    model = build_phase1_model(cfg, device)
    return load_weights(model, ckpt_dir / ("fold_%d" % fold) / "phase1_best.pt", device)


# ─────────────────────────────── presentation ─────────────────────────────────
def table(title, arms, fn):
    """fn(arm_key, at_k) -> (P, R, F1) or None. Passing the arm key (not the
    checkpoint dir) lets fn look up per-arm settings, so a new arm needs no
    change here -- only a correct ARMS entry."""
    print(f"\n{'='*74}\n  {title}  (micro over {N_FOLDS} folds)\n{'='*74}")
    print(f"  {'arm':28s} " + "  ".join(f"{'@'+str(k)+' P/R/F1':>18s}" for k in (1, 2, 3)))
    for key in arms:
        name = ARMS[key][0]
        cells = []
        for at_k in (1, 2, 3):
            m = fn(key, at_k)
            cells.append("      --          " if m is None
                         else f"{m[0]:.3f}/{m[1]:.3f}/{m[2]:.3f}".rjust(18))
        print(f"  {name:28s} " + "  ".join(cells))


def spread(values):
    """mean ± population sd, as a display string."""
    return "%5.1f%% ± %.1f" % (100 * mean(values),
                               100 * (pstdev(values) if len(values) > 1 else 0.0))


def f_beta(p, r, beta=2.0):
    """
    F-beta, matching analysis/add_f2_metric.py so the two never diverge.

    beta=2 weights recall twice as heavily as precision: F2 = 5PR / (4P + R).
    That is the right emphasis here — missing the vulnerability-inducing commit
    costs far more than asking a developer to look at one extra candidate.
    """
    b2 = beta * beta
    return ((1 + b2) * p * r / (b2 * p + r)) if (p + r) else 0.0


def report_dataset(label, arm_key, metrics, excluded, k_lines):
    """
    P/R/F1 @1/2/3 for one dataset (or a pooled set), per fold checkpoint.

    Identical in definition to the k-fold table: both come from
    evaluate_global, where a case's prediction is the top-k ranked slots
    de-duplicated by SHA, precision is micro-averaged Σhits/Σidentified and
    recall Σhits/Σgt.
    """
    print(f"\n{'='*74}")
    print("  %s on %s" % (ARMS[arm_key][0], label))
    print("  top_k_lines=%d, mean ± sd over %d fold checkpoints "
          "(every fold is unseen on this data)" % (k_lines, len(metrics)))
    print("=" * 74)
    if excluded:
        print("  excluded (no pre-built deletion graphs): %s" % ", ".join(excluded))

    print("  cases evaluated: %d   ground-truth VIC commits: %d"
          % (metrics[0][1]["n_evaluated"], metrics[0][1]["total_gt"]))
    print("  %-6s %-18s %-18s %-18s %-18s" % ("", "precision", "recall", "F1", "F2"))
    for at_k in (1, 2, 3):
        print("  @%-5d %-18s %-18s %-18s %-18s" % (
            at_k,
            spread([m[at_k]["precision"] for m in metrics]),
            spread([m[at_k]["recall"] for m in metrics]),
            spread([m[at_k]["f1"] for m in metrics]),
            spread([f_beta(m[at_k]["precision"], m[at_k]["recall"])
                    for m in metrics])))


def latex_table(rows, at_k, arm_name):
    """
    Emit the generalizability table as LaTeX.

    ``rows`` is [(label, metrics_by_fold), ...] with the pooled row last.
    Counts are integers, but each case is evaluated once per fold, so a count is
    the mean over the folds. Where that mean is not a whole number the folds
    disagreed; it is rounded for display and the exact per-fold values are
    printed underneath so nothing is hidden. P/R/F1 are micro-averaged over the
    folds' summed counts, matching how commit_micro aggregates the k-fold table.
    """
    def agg(metrics):
        hits = sum(m[at_k]["total_hits"] for m in metrics)
        ident = sum(m[at_k]["total_identified"] for m in metrics)
        gt = sum(m[at_k]["total_gt"] for m in metrics)
        cve = sum(m[at_k]["cve_identified"] for m in metrics)
        P = hits / ident if ident else 0.0
        R = hits / gt if gt else 0.0
        n = len(metrics)
        return dict(cases=metrics[0][at_k]["cve_total"], cve=cve / n,
                    gt=metrics[0][at_k]["total_gt"], hits=hits / n,
                    P=P, R=R, F2=f_beta(P, R))

    print("\n" + "=" * 74)
    print("  LaTeX table (@%d), %s" % (at_k, arm_name))
    print("=" * 74)
    print(r"\begin{table*}[htbp]")
    print(r"    \centering")
    print(r"    \small")
    print(r"    \caption{Generalizability Results Across Unseen Projects. "
          r"``Identified CVE'' reports CVE-level success, while Precision@%d, "
          r"Recall@%d, and F2@%d are computed at the commit level over all "
          r"ground-truth VICs.}" % (at_k, at_k, at_k))
    print(r"    \label{tab:generalizability}")
    print(r"    \begin{tabular}{lccccccc}")
    print(r"        \toprule")
    print(r"        \textbf{Project} & \textbf{Total CVE} & \textbf{Identified CVE} & "
          r"\textbf{GT VICs} & \textbf{Correctly Identified} & "
          r"\textbf{Precision@%d} & \textbf{Recall@%d} & \textbf{F2@%d}  \\" % (at_k, at_k, at_k))
    print(r"        \midrule")
    for i, (label, metrics) in enumerate(rows):
        a = agg(metrics)
        bold = (i == len(rows) - 1)
        fmt = (r"        \textbf{%s} & \textbf{%d} & \textbf{%d} & \textbf{%d} & "
               r"\textbf{%d} & \textbf{%.3f} & \textbf{%.3f} & \textbf{%.3f}  \\"
               if bold else
               r"        %-12s &%3d &%3d & %d & %d & %.3f & %.3f & %.3f  \\")
        if bold:
            print(r"        \midrule")
        print(fmt % (label, a["cases"], round(a["cve"]), a["gt"],
                     round(a["hits"]), a["P"], a["R"], a["F2"]))
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table*}")

    print("\n  per-fold counts (Identified CVE | Correctly Identified):")
    for label, metrics in rows:
        print("    %-14s %s | %s" % (
            label,
            " ".join(str(m[at_k]["cve_identified"]) for m in metrics),
            " ".join(str(m[at_k]["total_hits"]) for m in metrics)))


def pool_across(per_dataset):
    """
    Combine several datasets' per-fold results into one pooled set.

    Fold i of every dataset used the same checkpoint, so folds combine
    element-wise. P/R/F1 are recomputed from summed hit / identified / gt counts
    — micro-averaging, exactly as commit_micro does over folds — rather than
    averaged across datasets, which would weight a 19-case project the same as a
    20-case one.
    """
    metrics_by_fold = []
    n_folds = len(next(iter(per_dataset.values())))
    for fold in range(n_folds):
        merged = {}
        for at_k in (1, 2, 3):
            cells = [per_dataset[d][fold][at_k] for d in per_dataset]
            hits = sum(c["total_hits"] for c in cells)
            ident = sum(c["total_identified"] for c in cells)
            gt = sum(c["total_gt"] for c in cells)
            P = hits / ident if ident else 0.0
            R = hits / gt if gt else 0.0
            merged[at_k] = dict(precision=P, recall=R,
                                f1=(2 * P * R / (P + R) if P + R else 0.0),
                                total_hits=hits, total_identified=ident,
                                total_gt=gt,
                                n_evaluated=sum(c["n_evaluated"] for c in cells),
                                cve_identified=sum(c["cve_identified"] for c in cells),
                                cve_total=sum(c["cve_total"] for c in cells))
        metrics_by_fold.append(merged)
    return metrics_by_fold


def verify_folds(arm_key, k_lines, fix=False):
    """
    Re-run an arm on its OWN k-fold test splits and diff against the log.

    This is the correctness guard for the arm registry. The settings in ARMS are
    declared, not persisted by the training run, so a wrong encoder / graph_mode
    / PE entry would load real weights under the wrong architecture assumptions
    and quietly produce degraded scores on new data. Reproducing each fold's
    logged recall proves the reconstruction is faithful before it is trusted
    elsewhere. Small drift is expected: Phase 2 score spreads are ~0.1, so
    near-ties flip under GPU non-determinism.

    fix=True overwrites each fold's fold_results.json with the recomputed
    @1/@2/@3 metrics instead of just diffing -- for correcting a checkpoint's
    logged numbers after a bug fix in how they were computed, WITHOUT
    retraining (the saved weights are untouched; only the reported metrics
    are recomputed from them). Use sparingly and only when you know exactly
    why the logged numbers are wrong.
    """
    cfg, _, spec = arm_config(arm_key)
    root, prebuilt = cfg["paths"]["data_root"], cfg["paths"]["prebuilt_dir"]
    cases, _ = dataset_cases(root, prebuilt, spec["graph_mode"])
    splits = fold_splits(cases)

    metrics, _ = evaluate_dataset(arm_key, root, prebuilt, k_lines,
                                  cases_by_fold=dict(enumerate(splits)))

    print("\n  fold   logged hits   recomputed   split")
    ok = True
    for fold, m in enumerate(metrics):
        fr_path = HERE / ARMS[arm_key][1] / ("fold_%d" % fold) / "fold_results.json"
        rec = json.load(open(fr_path))
        if rec["split"]["test"] != len(splits[fold]):
            raise RuntimeError(
                "fold %d reconstruction is misaligned: %d cases vs %d logged. "
                "Every paired comparison built on these splits would be invalid."
                % (fold, len(splits[fold]), rec["split"]["test"]))

        p2_block = rec["phase2"]["test_metrics_by_k"][str(k_lines)]
        logged = p2_block["1"]["total_hits"]
        got = m[1]["total_hits"]
        flag = "" if logged == got else "   <- differs"
        ok = ok and logged == got
        print("   %d      %6d       %6d       %d cases%s"
              % (fold, logged, got, len(splits[fold]), flag))

        if fix:
            for at_k in (1, 2, 3):
                p2_block[str(at_k)] = {
                    "precision": m[at_k]["precision"], "recall": m[at_k]["recall"],
                    "f1": m[at_k]["f1"], "total_hits": m[at_k]["total_hits"],
                    "total_identified": m[at_k]["total_identified"],
                    "total_gt": m[at_k]["total_gt"], "n_evaluated": m[at_k]["n_evaluated"],
                }
            json.dump(rec, open(fr_path, "w"), indent=2)

    if fix:
        print("  fixed: fold_results.json rewritten for every fold above "
              "(weights untouched, only reported metrics recomputed)")
    else:
        print("  %s" % ("exact match on every fold" if ok else
                        "differences above are hit counts at @1; a few cases of drift is "
                        "near-tie flipping, a large gap means ARMS is wrong"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="+", choices=list(ARMS), default=list(ARMS))
    ap.add_argument("--ckpt-dir", nargs="+", metavar="ARM=DIR", default=[],
                    help="Override an arm's checkpoint directory, e.g. "
                         "--ckpt-dir gat_temporal=checkpoints_myrun. Repeatable.")
    ap.add_argument("--k-lines", type=int,
                    default=ConfigManager().raw["phase2"]["top_k_lines"],
                    help="top_k_lines block for commit ranking (default 3).")
    ap.add_argument("--compute-line-level", action="store_true",
                    help="(Re)compute deletion-line ranking on GPU and cache it.")
    ap.add_argument("--data-root", nargs="+", metavar="DIR",
                    help="Evaluate on these external dataset roots instead of "
                         "reading k-fold results. Each holds <case>/info.json.")
    ap.add_argument("--prebuilt-dir", nargs="+", metavar="DIR",
                    help="Pre-built graph dir per --data-root (build_graphs.py "
                         "--output_dir), or one shared dir for all of them.")
    ap.add_argument("--out", metavar="FILE",
                    help="Write external-dataset results to this JSON file.")
    ap.add_argument("--latex-at", nargs="+", type=int, default=[3], metavar="K",
                    help="Emit the generalizability LaTeX table at these @k "
                         "(default 3).")
    ap.add_argument("--verify-folds", action="store_true",
                    help="Re-run each arm on its own k-fold splits and diff "
                         "against fold_results.json, to prove the ARMS settings "
                         "rebuild the checkpoints faithfully.")
    ap.add_argument("--fix-metrics", action="store_true",
                    help="With --verify-folds, OVERWRITE each fold's "
                         "fold_results.json with the recomputed @1/@2/@3 "
                         "metrics instead of just diffing. Weights are never "
                         "touched, only reported numbers.")
    args = ap.parse_args()

    for spec in args.ckpt_dir:
        if "=" not in spec:
            ap.error("--ckpt-dir expects ARM=DIR, got %r" % spec)
        arm, path = spec.split("=", 1)
        if arm not in ARMS:
            ap.error("--ckpt-dir: unknown arm %r (choose from %s)"
                     % (arm, ", ".join(ARMS)))
        name, _old, settings = ARMS[arm]
        ARMS[arm] = (name, path, settings)
        print("[ckpt-dir] %s -> %s" % (arm, path))

    if args.verify_folds:
        for arm_key in args.arm:
            print("\n[verify] %s" % ARMS[arm_key][0])
            verify_folds(arm_key, args.k_lines, fix=args.fix_metrics)
        return

    if args.data_root:
        if not args.prebuilt_dir:
            ap.error("--data-root requires --prebuilt-dir")
        if len(args.prebuilt_dir) == 1:
            prebuilt = args.prebuilt_dir * len(args.data_root)
        elif len(args.prebuilt_dir) == len(args.data_root):
            prebuilt = args.prebuilt_dir
        else:
            ap.error("--prebuilt-dir takes one directory, or one per --data-root "
                     "(got %d for %d roots)" % (len(args.prebuilt_dir), len(args.data_root)))

        collected = {}
        for arm_key in args.arm:
            per_dataset = {}
            for root, pre in zip(args.data_root, prebuilt):
                label = Path(root).name
                print("\n[%s / %s]" % (arm_key, label))
                metrics, excluded = evaluate_dataset(arm_key, root, pre, args.k_lines)
                report_dataset(label, arm_key, metrics, excluded, args.k_lines)
                per_dataset[label] = metrics
                collected.setdefault(arm_key, {})[label] = dict(
                    data_root=str(root), prebuilt_dir=str(pre),
                    excluded=excluded, k_lines=args.k_lines,
                    per_fold_metrics=[{str(k): v for k, v in m.items()} for m in metrics])

            rows = list(per_dataset.items())
            if len(per_dataset) > 1:
                metrics = pool_across(per_dataset)
                report_dataset("ALL DATASETS POOLED (%s)" % ", ".join(per_dataset),
                               arm_key, metrics, [], args.k_lines)
                collected[arm_key]["_pooled"] = dict(
                    datasets=list(per_dataset), k_lines=args.k_lines,
                    per_fold_metrics=[{str(k): v for k, v in m.items()} for m in metrics])
                rows.append(("Overall", metrics))
            for at_k in args.latex_at:
                latex_table(rows, at_k, ARMS[arm_key][0])

        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(collected, open(args.out, "w"), indent=1)
            print("\n  results -> %s" % args.out)
        return

    if args.compute_line_level:
        for key in args.arm:
            print(f"\n[line-level] computing {ARMS[key][0]} ...")
            compute_line_level(key)

    table(f"Commit ranking (VIC), top_k_lines={args.k_lines}", args.arm,
          lambda key, at_k: commit_micro(ARMS[key][1], args.k_lines, at_k))
    table("Deletion-line ranking (Phase 1)", args.arm,
          lambda key, at_k: line_micro(ARMS[key][1], at_k))
    if any(line_micro(ARMS[k][1], 1) is None for k in args.arm):
        print("\n  (line-level '--' = not cached; run with --compute-line-level)")


if __name__ == "__main__":
    main()
