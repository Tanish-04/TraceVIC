"""
main_kfold.py
─────────────
5-fold cross-validation wrapper for the unified two-phase pipeline.

Usage:
    nohup python main_kfold.py \
    --save-dir checkpoints_kfold_TemporalGAT \
    --encoder-type gat \
    --graph-mode full_graph \
    --prebuilt-dir temporal_graph \
    > kfold_TemporalGAT.logs 2>&1 &

Each fold trains Phase 1 + Phase 2 independently, using 1 fold for test,
1 fold for val, and the remaining 3 for train (~60/20/20).

Outputs:
    <save_dir>/fold_0/ ... fold_4/   — per-fold checkpoints + results
    <save_dir>/kfold_summary.json    — aggregated mean ± std across folds
"""

import argparse
import gc
import json
import os
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config_utils import ConfigManager
CONFIG = ConfigManager().raw
from data_processing.dataset import DeletionLineDataset, CommitRankingDataset
from models.shared_encoder import UnixcoderEmbedder

from training.embedding_cache import score_deletion_lines, build_phase2_items
from training.phase1_trainer import train_phase1_fold
from training.phase2_trainer import train_phase2_fold
from training.utils import build_phase1_model, build_phase2_model, set_seed, setup_device
from training.evaluation import evaluate_global, load_true_commit_map


# ── Reuse arg parsing and config overlay from main.py ────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-fold CV for TraceVIC")
    # Phase 1
    p.add_argument("--phase1-epochs", type=int)
    p.add_argument("--phase1-lr", type=float)
    p.add_argument("--phase1-bert-lr", type=float)
    p.add_argument("--phase1-rest-lr", type=float)
    p.add_argument("--phase1-bert-freeze-bottom-layers", type=int)
    p.add_argument("--max-graphs-per-batch", type=int)
    p.add_argument("--phase1-patience", type=int)

    # Phase 2
    p.add_argument("--phase2-epochs", type=int, default=None)
    p.add_argument("--phase2-lr", type=float, default=None)
    p.add_argument("--phase2-weight-decay", type=float, default=None)
    p.add_argument("--phase2-batch-size", type=int, default=None)
    p.add_argument("--phase2-patience", type=int, default=None)
    p.add_argument("--phase2-dropout", type=float, default=None)
    p.add_argument("--phase2-gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--phase2-temperature", type=float, default=None)
    p.add_argument("--phase2-label-smoothing", type=float, default=None)
    p.add_argument("--phase2-top-k-lines", type=int, default=None)
    p.add_argument("--phase2-focal-gamma", type=float, default=None)
    p.add_argument("--phase2-focal-alpha", type=float, default=None)
    p.add_argument("--phase2-margin", type=float, default=None)
    p.add_argument("--phase2-margin-weight", type=float, default=None)
    p.add_argument("--phase2-hidden-dim", type=int, default=None)
    p.add_argument("--phase2-num-heads", type=int, default=None)
    p.add_argument("--phase2-num-commit-transformer-layers", type=int, default=None)
    p.add_argument("--phase2-max-temporal-dist", type=int, default=None)
    p.set_defaults(phase2_use_dual_query=None, phase2_use_temporal_pe=None)
    p.add_argument("--phase2-use-dual-query", dest="phase2_use_dual_query",
                   action="store_const", const=True)
    p.add_argument("--no-phase2-use-dual-query", dest="phase2_use_dual_query",
                   action="store_const", const=False)
    p.add_argument("--phase2-use-temporal-pe", dest="phase2_use_temporal_pe",
                   action="store_const", const=True)
    p.add_argument("--no-phase2-use-temporal-pe", dest="phase2_use_temporal_pe",
                   action="store_const", const=False)

    # Shared
    p.add_argument("--hidden-dim", type=int)
    p.add_argument("--num-gt-layers", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--save-dir", type=str)

    # Temporal-ablation switches.  Default None so omitting them leaves
    # CONFIG (and therefore existing behaviour) completely untouched.
    p.set_defaults(model_use_temporal_pe=None)
    p.add_argument("--model-use-temporal-pe", dest="model_use_temporal_pe",
                   action="store_const", const=True,
                   help="Enable model.use_temporal_pe (encoder node-level PE)")
    p.add_argument("--no-model-use-temporal-pe", dest="model_use_temporal_pe",
                   action="store_const", const=False,
                   help="Disable model.use_temporal_pe (encoder node-level PE)")

    # Model / graph variant
    p.add_argument("--encoder-type", type=str, choices=["gat"],
                   help="Encoder architecture (only 'gat' ships in TraceVIC)")
    p.add_argument("--graph-mode", type=str)
    p.add_argument("--prebuilt-dir", type=str)

    # Reuse an existing per-fold Phase 1 instead of retraining it. The
    # directory must contain fold_0/ .. fold_{n-1}/ each with phase1_best.pt.
    p.add_argument("--skip-phase1", action="store_true",
                   help="Load each fold's Phase 1 checkpoint instead of training it")
    p.add_argument("--phase1-checkpoint-dir", type=str, default=None,
                   help="Directory of per-fold Phase 1 checkpoints (fold_N/phase1_best.pt)")
    # Data / device
    p.add_argument("--data-root", type=str,
                   help="Dataset root (one subdirectory per test case). "
                        "Overrides paths.data_root — use this to train on a "
                        "different dataset without editing the YAML.")
    p.add_argument("--gpu-id", type=int,
                   help="CUDA device index. Overrides defaults.gpu_id.")

    # K-fold specific. Default None so defaults.n_folds in the YAML wins
    # unless the flag is given.
    p.add_argument("--n-folds", type=int, default=None,
                   help="Number of CV folds (default: defaults.n_folds in the YAML)")

    return p.parse_args()


def _apply_cli(args: argparse.Namespace) -> None:
    CLI_MAP = {
        "phase1_epochs":                     ("phase1", "epochs"),
        "phase1_lr":                         ("phase1", "lr"),
        "phase1_bert_lr":                    ("phase1", "bert_lr"),
        "phase1_rest_lr":                    ("phase1", "rest_lr"),
        "phase1_bert_freeze_bottom_layers":  ("phase1", "bert_freeze_bottom_layers"),
        "phase1_patience":                   ("phase1", "patience"),
        "max_graphs_per_batch":              ("phase1", "max_graphs_per_batch"),
        "phase2_epochs":                     ("phase2", "epochs"),
        "phase2_lr":                         ("phase2", "lr"),
        "phase2_weight_decay":               ("phase2", "weight_decay"),
        "phase2_batch_size":                 ("phase2", "batch_size"),
        "phase2_patience":                   ("phase2", "patience"),
        "phase2_dropout":                    ("phase2", "dropout"),
        "phase2_gradient_accumulation_steps": ("phase2", "gradient_accumulation_steps"),
        "phase2_temperature":              ("phase2", "temperature"),
        "phase2_label_smoothing":          ("phase2", "label_smoothing"),
        "phase2_top_k_lines":              ("phase2", "top_k_lines"),
        "phase2_focal_gamma":              ("phase2", "focal_gamma"),
        "phase2_focal_alpha":              ("phase2", "focal_alpha"),
        "phase2_margin":                   ("phase2", "margin"),
        "phase2_margin_weight":            ("phase2", "margin_weight"),
        "phase2_hidden_dim":               ("phase2", "hidden_dim"),
        "phase2_num_heads":                ("phase2", "num_heads"),
        "phase2_num_commit_transformer_layers": ("phase2", "num_commit_transformer_layers"),
        "phase2_max_temporal_dist":        ("phase2", "max_temporal_dist"),
        "phase2_use_dual_query":           ("phase2", "use_dual_query"),
        "phase2_use_temporal_pe":          ("phase2", "use_temporal_pe"),
        "hidden_dim":                        ("model",  "hidden_dim"),
        "num_gt_layers":                     ("model",  "num_gt_layers"),
        "dropout":                           ("model",  "dropout"),
        "seed":                              ("defaults", "seed"),
        "save_dir":                          ("paths",   "save_dir"),
        "encoder_type":                      ("model", "encoder_type"),
        "data_root":                         ("paths",   "data_root"),
        "gpu_id":                            ("defaults","gpu_id"),
        "n_folds":                           ("defaults","n_folds"),
        "graph_mode":                        ("paths", "graph_mode"),
        "prebuilt_dir":                      ("paths", "prebuilt_dir"),
        "model_use_temporal_pe":             ("model", "use_temporal_pe"),
    }
    for cli_key, (section, yaml_key) in CLI_MAP.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            CONFIG.setdefault(section, {})[yaml_key] = val


# ── Phase 1 helpers (from main.py) ───────────────────────────────────

def diagnose_phase1_accuracy(scored, cases, label, top_k=3):
    correct = sum(
        1 for name in cases
        if name in scored and any(
            mg.rootcause
            for _, mg, _ in scored[name][:top_k]
        )
    )
    total = sum(1 for name in cases if name in scored)
    pct = correct / max(total, 1) * 100
    print(f"  Phase 1 top-{top_k} correct ({label}): "
          f"{correct}/{total} = {pct:.1f}%")
    return correct, total


# ── Core fold runner ─────────────────────────────────────────────────

def run_single_fold(
    fold_idx: int,
    train_cases: List[str],
    val_cases: List[str],
    test_cases: List[str],
    p1_dataset: DeletionLineDataset,
    device: torch.device,
    fold_save_dir: str,
    p1_ckpt_src: Optional[str] = None,
) -> Optional[Dict]:
    """Run full Phase 1 + Phase 2 for one fold. Returns metrics dict."""

    # Override save_dir for this fold
    CONFIG["paths"]["save_dir"] = fold_save_dir
    os.makedirs(fold_save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  FOLD {fold_idx + 1}")
    print(f"  Train: {len(train_cases)} | Val: {len(val_cases)} | Test: {len(test_cases)}")
    print(f"  Save dir: {fold_save_dir}")
    print(f"{'='*70}")

    # ── Phase 1 ──────────────────────────────────────────────────
    if p1_ckpt_src:
        src = Path(p1_ckpt_src) / f"fold_{fold_idx}" / "phase1_best.pt"
        if not src.exists():
            raise FileNotFoundError(f"--skip-phase1 set but not found: {src}")
        print(f"  Loading Phase 1 for fold {fold_idx + 1}: {src}")
        p1_result = {"model_state": torch.load(src, map_location="cpu")["model_state_dict"],
                     "metrics": {}, "best_epoch": None, "best_f1": None,
                     "loaded_from": str(src)}
    else:
        p1_result = train_phase1_fold(
            fold_idx, train_cases, val_cases, p1_dataset, CONFIG, device=device
        )
        if p1_result is None:
            print(f"  ✗ Phase 1 failed for fold {fold_idx + 1}")
            return None

    p1_ckpt_path = Path(fold_save_dir) / "phase1_best.pt"
    torch.save({"model_state_dict": p1_result["model_state"]}, p1_ckpt_path)
    print(f"  ✓ Phase 1 checkpoint saved to {p1_ckpt_path}")

    # ── Score deletion lines ─────────────────────────────────────
    all_cases = train_cases + val_cases + test_cases

    p1_model = build_phase1_model(CONFIG, device)
    p1_model.load_state_dict(p1_result["model_state"], strict=False)
    p1_model.encoder.eval()
    for param in p1_model.encoder.parameters():
        param.requires_grad = False

    print(f"\n  Scoring deletion lines for {len(all_cases)} cases...")
    scored = score_deletion_lines(
        p1_model, p1_dataset, all_cases, device,
        max_nodes=CONFIG["defaults"]["max_nodes_per_batch"],
        top_k=CONFIG["phase2"]["top_k_lines"],
    )
    print(f"  Top graphs selected: {len(scored)}/{len(all_cases)}")

    # Phase 1 diagnostics
    tr_correct, tr_total = diagnose_phase1_accuracy(scored, train_cases, "train")
    vl_correct, vl_total = diagnose_phase1_accuracy(scored, val_cases, "val")
    te_correct, te_total = diagnose_phase1_accuracy(scored, test_cases, "test")

    del p1_model
    gc.collect()

    # ── Build Phase 2 data ───────────────────────────────────────
    print("\n  Building Phase 2 embedding items...")
    p2_items_by_k = build_phase2_items(
        scored, all_cases,
        graph_mode=CONFIG["paths"]["graph_mode"],
        top_k_lines=CONFIG["phase2"]["top_k_lines"],
    )

    del scored
    gc.collect()
    torch.cuda.empty_cache()

    p2_datasets = {k: CommitRankingDataset(items) for k, items in p2_items_by_k.items()}
    case_to_idx = {name: i for i, name in enumerate(all_cases)}

    # ── Phase 2 ──────────────────────────────────────────────────
    train_idx = [case_to_idx[c] for c in train_cases if c in case_to_idx]
    val_idx   = [case_to_idx[c] for c in val_cases   if c in case_to_idx]
    test_idx  = [case_to_idx[c] for c in test_cases  if c in case_to_idx]

    train_k  = CONFIG["phase2"]["top_k_lines"]
    train_ds = p2_datasets[train_k]

    item_map = {
        item["test_name"]: item
        for item in train_ds.items
        if item.get("test_name")
    }
    true_cid_map = load_true_commit_map(all_cases, CONFIG["paths"]["data_root"])
    data_root = Path(CONFIG["paths"]["data_root"])

    p2_result = train_phase2_fold(
        fold_idx, train_idx, val_idx, train_ds, CONFIG,
        train_cases=train_cases,
        val_cases=val_cases,
        item_map=item_map,
        true_cid_map=true_cid_map,
        data_root=data_root,
    )

    # ── Test evaluation ──────────────────────────────────────────
    print(f"\n  Evaluating on held-out test set ({len(test_idx)} cases)...")
    test_model = build_phase2_model(CONFIG, device)
    if p2_result.get("best_commit_ranker_state"):
        test_model.load_state_dict(p2_result["best_commit_ranker_state"])
    test_model.eval()   # belt-and-braces; evaluate_global also does this

    top_k_max = CONFIG["phase2"]["top_k_lines"]
    test_metrics_by_k = {}

    for k in range(1, top_k_max + 1):
        k_ds = p2_datasets[k]
        k_item_map = {
            item["test_name"]: item
            for item in k_ds.items
            if item.get("test_name")
        }
        print(f"\n  Evaluating test set (top_k_lines={k}, {len(test_cases)} cases)...")
        test_results = evaluate_global(
            cases=test_cases,
            item_map=k_item_map,
            true_cid_map=true_cid_map,
            p2_model=test_model,
            device=device,
            data_root=data_root,
        )
        test_metrics_by_k[k] = test_results

        for vk in sorted(test_results.keys()):
            vm = test_results[vk]
            print(
                f"    k={k} @{vk}: P={vm.get('precision', 0):.4f}  "
                f"R={vm.get('recall', 0):.4f}  "
                f"F1={vm.get('f1', 0):.4f}"
            )

    # Save Phase 2 checkpoint
    torch.save(
        {"model_state_dict": p2_result["best_commit_ranker_state"]},
        Path(fold_save_dir) / "phase2_best.pt",
    )

    del test_model, p2_datasets
    gc.collect()
    torch.cuda.empty_cache()

    # ── Collect fold metrics ─────────────────────────────────────
    fold_metrics = {
        "fold": fold_idx,
        "phase1": {
            "best_epoch": p1_result["best_epoch"],
            "best_f1": p1_result["best_f1"],
            "metrics": p1_result.get("metrics", {}),
            "test_top3_correct": te_correct,
            "test_top3_total": te_total,
            "test_top3_pct": te_correct / max(te_total, 1) * 100,
            "train_top3_correct": tr_correct,
            "train_top3_total": tr_total,
            "train_top3_pct": tr_correct / max(tr_total, 1) * 100,
        },
        "phase2": {
            "best_epoch": p2_result.get("best_epoch", 0),
            "val_metrics": p2_result.get("final_metrics", {}),
            "test_metrics_by_k": test_metrics_by_k,
        },
        "split": {
            "train": len(train_cases),
            "val": len(val_cases),
            "test": len(test_cases),
        },
    }

    # Save per-fold summary
    with open(Path(fold_save_dir) / "fold_results.json", "w") as f:
        json.dump(fold_metrics, f, indent=2)

    return fold_metrics


# ── Aggregation ──────────────────────────────────────────────────────

def _stat(values: List) -> Dict:
    """mean ± std over values, skipping None.

    --skip-phase1 loads a checkpoint instead of training, so it has no
    best_epoch/best_f1 to report; those arrive here as None.  Returns
    None rather than raising when nothing is available.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None}
    return {"mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0}


def aggregate_results(all_fold_metrics: List[Dict], n_folds: int) -> Dict:
    """Compute mean ± std across folds for key metrics."""

    summary = {"n_folds": n_folds, "per_fold": all_fold_metrics}

    # Phase 1
    p1_f1s       = [m["phase1"]["best_f1"] for m in all_fold_metrics]
    p1_epochs    = [m["phase1"]["best_epoch"] for m in all_fold_metrics]
    p1_test_top3 = [m["phase1"]["test_top3_pct"] for m in all_fold_metrics]
    p1_train_top3 = [m["phase1"]["train_top3_pct"] for m in all_fold_metrics]

    summary["phase1_aggregate"] = {
        "val_f1@1":  _stat(p1_f1s),
        "best_epoch": _stat(p1_epochs),
        "test_top3_pct": _stat(p1_test_top3),
        "train_top3_pct": _stat(p1_train_top3),
        "train_test_gap": {"mean": statistics.mean(p1_train_top3) - statistics.mean(p1_test_top3)},
    }

    # Phase 2 — aggregate test F1@1 for each top_k_lines setting
    top_k_max = max(
        int(k)
        for m in all_fold_metrics
        for k in m["phase2"]["test_metrics_by_k"].keys()
    )

    p2_agg = {}
    for k in range(1, top_k_max + 1):
        f1s_at1 = []
        ps_at1 = []
        rs_at1 = []
        for m in all_fold_metrics:
            k_results = m["phase2"]["test_metrics_by_k"].get(k, m["phase2"]["test_metrics_by_k"].get(str(k), {}))
            # Get @1 metrics
            at1 = k_results.get(1, k_results.get("1", {}))
            if at1:
                f1s_at1.append(at1.get("f1", 0))
                ps_at1.append(at1.get("precision", 0))
                rs_at1.append(at1.get("recall", 0))

        if f1s_at1:
            p2_agg[f"k={k}_@1"] = {
                "precision": {"mean": statistics.mean(ps_at1), "std": statistics.stdev(ps_at1) if len(ps_at1) > 1 else 0},
                "recall":    {"mean": statistics.mean(rs_at1), "std": statistics.stdev(rs_at1) if len(rs_at1) > 1 else 0},
                "f1":        {"mean": statistics.mean(f1s_at1), "std": statistics.stdev(f1s_at1) if len(f1s_at1) > 1 else 0},
                "per_fold": f1s_at1,
            }

    summary["phase2_aggregate"] = p2_agg

    return summary


def print_summary(summary: Dict) -> None:
    """Pretty-print the cross-validation summary."""
    n = summary["n_folds"]
    p1 = summary["phase1_aggregate"]
    p2 = summary["phase2_aggregate"]

    print(f"\n{'='*70}")
    print(f"  {n}-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")

    def _fmt(stat, spec=".4f", suffix=""):
        """None-safe '<mean> ± <std>' — mean is None when --skip-phase1 was used."""
        if stat["mean"] is None:
            return "n/a (loaded checkpoint)"
        return f"{stat['mean']:{spec}}{suffix} ± {stat['std']:{spec}}{suffix}"

    print("\n  Phase 1 (Deletion Line Ranking):")
    print(f"    Val F1@1:       {_fmt(p1['val_f1@1'])}")
    print(f"    Best epoch:     {_fmt(p1['best_epoch'], '.1f')}")
    print(f"    Test top-3:     {_fmt(p1['test_top3_pct'], '.1f', '%')}")
    print(f"    Train top-3:    {_fmt(p1['train_top3_pct'], '.1f', '%')}")
    print(f"    Train-test gap: {p1['train_test_gap']['mean']:.1f} pts")

    print("\n  Phase 2 (Commit Ranking) — Test:")
    for key in sorted(p2.keys()):
        m = p2[key]
        print(f"    {key}: P={m['precision']['mean']:.4f}±{m['precision']['std']:.4f}  "
              f"R={m['recall']['mean']:.4f}±{m['recall']['std']:.4f}  "
              f"F1={m['f1']['mean']:.4f}±{m['f1']['std']:.4f}")
        print(f"      per-fold F1: {[f'{v:.4f}' for v in m['per_fold']]}")

    print("\n  Per-fold Phase 1 details:")
    for fm in summary["per_fold"]:
        f = fm["fold"]
        bf1 = fm["phase1"]["best_f1"]
        print(f"    Fold {f+1}: best_epoch={fm['phase1']['best_epoch']}, "
              f"val_F1@1={f'{bf1:.4f}' if bf1 is not None else 'n/a'}, "
              f"test_top3={fm['phase1']['test_top3_pct']:.1f}%")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    _apply_cli(args)

    seed = CONFIG["defaults"]["seed"]
    n_folds = CONFIG["defaults"]["n_folds"]
    set_seed(seed)
    print("Seed check:", torch.rand(3).tolist())

    base_save_dir = CONFIG["paths"]["save_dir"]
    os.makedirs(base_save_dir, exist_ok=True)

    device = setup_device(CONFIG["defaults"]["gpu_id"])

    print("=" * 70)
    print(f"{n_folds}-FOLD CROSS-VALIDATION")
    print(f"  device       : {device}")
    print(f"  encoder_type : {CONFIG['model'].get('encoder_type', 'gat')}")
    print(f"  graph_mode   : {CONFIG['paths'].get('graph_mode', 'temporal')}")
    print(f"  base_save_dir: {base_save_dir}")
    print("=" * 70)

    # ── Load all cases ───────────────────────────────────────────
    data_root = Path(CONFIG["paths"]["data_root"])
    all_cases: List[str] = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(data_root / d)
    )
    print(f"Total cases: {len(all_cases)}")

    # ── Build k-fold splits ──────────────────────────────────────
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_cases_arr = np.array(all_cases)

    # ── Load dataset once (shared across all folds) ──────────────
    print("\nInitialising Unixcoder tokenizer...")
    embedder = UnixcoderEmbedder(
        model_name=CONFIG["model"].get("unixcoder_model_name",
                                       "microsoft/unixcoder-base-nine"),
        max_len=CONFIG["model"].get("max_seq_len", 64),
        tokenizer_only=True)

    print("\nLoading dataset...")
    p1_dataset = DeletionLineDataset(
        data_path=CONFIG["paths"]["data_root"],
        test_cases=all_cases,
        embedder=embedder,
        prebuilt_dir=CONFIG["paths"]["prebuilt_dir"],
        graph_mode=CONFIG["paths"]["graph_mode"],
    )

    # ── Run each fold ────────────────────────────────────────────
    all_fold_metrics: List[Dict] = []

    for fold_idx, (non_test_indices, test_indices) in enumerate(kf.split(all_cases_arr)):
        # Reset seed for reproducibility within each fold
        set_seed(seed + fold_idx)

        test_cases = all_cases_arr[test_indices].tolist()

        # Split non-test into train/val (defaults.val_split of the non-test part)
        non_test_cases = all_cases_arr[non_test_indices].tolist()
        train_cases, val_cases = train_test_split(
            non_test_cases,
            test_size=CONFIG["defaults"].get("val_split", 0.20),
            random_state=seed + fold_idx, shuffle=True
        )

        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold_idx}")

        fold_metrics = run_single_fold(
            fold_idx=fold_idx,
            train_cases=train_cases,
            val_cases=val_cases,
            test_cases=test_cases,
            p1_dataset=p1_dataset,
            device=device,
            fold_save_dir=fold_save_dir,
            p1_ckpt_src=(args.phase1_checkpoint_dir if args.skip_phase1 else None),
        )

        if fold_metrics is not None:
            all_fold_metrics.append(fold_metrics)
        else:
            print(f"\n  ✗ Fold {fold_idx + 1} failed, skipping")

    # ── Aggregate and save ───────────────────────────────────────
    if len(all_fold_metrics) < 2:
        print("\n  ✗ Fewer than 2 folds succeeded, cannot compute std")
        return

    summary = aggregate_results(all_fold_metrics, n_folds)
    print_summary(summary)

    summary_path = os.path.join(base_save_dir, "kfold_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to {summary_path}")
    print(f"✓ All outputs saved to: {base_save_dir}")


if __name__ == "__main__":
    main()