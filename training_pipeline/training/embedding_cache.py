"""
Scores Phase 1 deletion lines and caches encoder outputs for Phase 2 training.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

import torch
from torch_geometric.data import Batch

from training.utils import coerce_idx
from data_processing.constants import EdgeType

# Step 1: score deletion lines, cache top-1 encoder output
def score_deletion_lines(
    model,
    dataset,
    test_cases: List[str],
    device: torch.device,
    max_nodes: int = 4096,
    top_k: int = 1,
) -> Dict[str, Tuple]:

    """
    Score every deletion-line graph and return the top-k MiniGraph per test
    case together with its full encoder output tensor.
    """
    model.eval()
    graphs_dict = dataset.get_mini_graphs_dict()
    results: Dict[str, Tuple] = {}

    def _score_batch(batch_list: List[Tuple]) -> None:
        nonlocal all_scored        
        if not batch_list:
            return
        try:
            batch_data = Batch.from_data_list(
                [gd for _, gd in batch_list]
            ).to(device)
            h_all = model.encoder.encode_pyg(batch_data).cpu()

            for i, (mg, _) in enumerate(batch_list):
                start = batch_data.ptr[i].item()
                end   = batch_data.ptr[i + 1].item()
                h_i   = h_all[start:end]
                idx   = coerce_idx(mg.del_idx)
                if idx < h_i.size(0):
                    s = model.ranker.score(h_i[idx].to(device)).item()
                    all_scored.append((s, mg, h_i.clone()))
        except Exception:
            # Batching failed — fall back to one-by-one encoding
            for mg, gd in batch_list:
                _score_single(mg, gd)

    def _score_single(mg, gd) -> None:
        nonlocal all_scored
        try:
            h   = model.encoder.encode_pyg(gd.to(device)).cpu()
            idx = coerce_idx(mg.del_idx)
            if idx < h.size(0):
                s = model.ranker.score(h[idx].to(device)).item()
                all_scored.append((s, mg, h))
        except Exception as exc:
            logger.debug("Failed to score graph: %s", exc)

    with torch.no_grad():
        for name in test_cases:
            if name not in graphs_dict:
                continue

            all_scored: List[Tuple] = []
            batch_list: List[Tuple] = []
            batch_nodes = 0

            for mg in graphs_dict[name]:
                n = mg.pyg.num_nodes

                if n > max_nodes:
                    # Graph too large to batch — flush current batch, then encode alone
                    _score_batch(batch_list)
                    batch_list, batch_nodes = [], 0
                    _score_single(mg, mg.pyg)
                    continue

                if batch_nodes + n > max_nodes and batch_list:
                    # Would overflow batch capacity — flush first
                    _score_batch(batch_list)
                    batch_list, batch_nodes = [], 0

                batch_list.append((mg, mg.pyg))
                batch_nodes += n

            _score_batch(batch_list)

            if all_scored:
                all_scored.sort(key=lambda x: x[0], reverse=True)
                results[name] = all_scored[:top_k]
    return results



#  Step 2: convert cached encoder outputs (1 CVE -> 1 Item ranked from pooled chain) into Phase 2 training items


def build_phase2_items(
    scored: Dict[str, Tuple],
    all_cases: List[str],
    graph_mode: str = "full_graph",
    top_k_lines: int = 3,
) -> Dict[int, List[dict]]:
    """
    
    For each test case:
      - node_embeddings is taken directly from the cached encoder output produced by score_deletion_lines.
      - commit_indices comes from mg.pyg.temporal_pos
      - ground_truth_position maps each inducing commit SHA to its temporal-position index via the tp_to_commit dict stored on the MiniGraph.
   """

    def _make_invalid(name: str) -> dict:
        return {
            "test_name": name,
            "valid": False,
            "node_embeddings": None,
            "commit_indices": None,
            "is_temporal_node": None,
            "ground_truth_positions": [],
            "num_unique_vics": 0,
            "is_correct_deletion_line": False,
            "p1_score": 0.0,
            "commit_idx_to_sha": {},
        }

    # results_by_k[k] = list of items for all test cases using top-k lines
    results_by_k: Dict[int, List[dict]] = {k: [] for k in range(1, top_k_lines + 1)}
    n_valid_by_k: Dict[int, int] = {k: 0 for k in range(1, top_k_lines + 1)}

    for test_name in all_cases:
        entry = scored.get(test_name)
        if entry is None:
            for k in range(1, top_k_lines + 1):
                results_by_k[k].append(_make_invalid(test_name))
            continue

        # Get Unique VIC from info.json via first MiniGraph
        unique_inducing_shas = set()
        first_mg = entry[0][1]  # (p1_score, mg, cached_h)
        for sha in first_mg.inducing_commits:
            unique_inducing_shas.add(sha[:12])
        
        num_unique_vics = len(unique_inducing_shas)

        # Pre-process each deletion line independently
        # so we can accumulate progressively for k=1,2,3
        per_line_data = []  # list of processed data per deletion line

        for p1_score, mg, cached_h in entry[:top_k_lines]:
            # GT Position for this deletion line
            commit_to_tp = {
                sha[:12]: tp
                for tp, sha in mg.tp_to_commit.items()
                if tp > 0
            }
            
            gt_positions_raw = sorted({
                commit_to_tp[sha[:12]]
                for sha in mg.inducing_commits
                if sha[:12] in commit_to_tp
            })
            gt_positions = [tp-1 for tp in gt_positions_raw]

          
            # Keep only commit nodes (temporal_pos > 0); MiniGraph nodes have temporal_pos == 0
            temporal_pos_all = mg.pyg.temporal_pos.cpu()
            commit_node_mask = temporal_pos_all > 0

            node_embeddings  = cached_h[commit_node_mask]
            commit_indices   = temporal_pos_all[commit_node_mask] - 1

            if node_embeddings.size(0) == 0 or commit_indices.numel() == 0:
                per_line_data.append(None)
                continue

            n_commits = int(commit_indices.max().item()) + 1
            gt_positions = [g for g in gt_positions if g < n_commits]
            # commit_count_dist = {1: 0, 2: 0, 3: 0, "4+": 0}

            # Temporal Mask
            edge_index = mg.pyg.edge_index
            edge_type = mg.pyg.edge_type
            temporal_fwd_type = EdgeType.TEMPORAL_FWD
            temporal_mask_full = torch.zeros(mg.pyg.num_nodes, dtype=torch.bool)
            temporal_dst = edge_index[1][edge_type == temporal_fwd_type]
            temporal_mask_full[temporal_dst] = True
            is_temporal_node = temporal_mask_full[commit_node_mask]

            base_map: Dict[int, str] = {}
            for tp, sha in mg.tp_to_commit.items():
                if tp <= 0:
                    continue
                s = sha if isinstance(sha, str) else str(sha)
                base_map[int(tp - 1)] = s[:12]

            line_commit_idx_to_sha = dict(base_map)

            per_line_data.append({
                "node_embeddings":        node_embeddings,
                "commit_indices":         commit_indices,
                "is_temporal_node":       is_temporal_node,
                "gt_positions":           gt_positions,
                "n_commits":              n_commits,
                "p1_score":               p1_score,
                "commit_idx_to_sha":      line_commit_idx_to_sha,
            })

        # Build one item per k. Each (line, depth) slot in the pooled top-k
        # chains is its own candidate.
        for k in range(1, top_k_lines+1):
            lines = [ld for ld in per_line_data[:k] if ld is not None]
            if not lines:
                results_by_k[k].append(_make_invalid(test_name))
                continue

            best_p1_score = max(ld["p1_score"] for ld in lines)

            def _key(li, local_i, sha):
                return "\x00slot:%d:%d" % (li, local_i)

            # Pass 1 -- one record per unique candidate commit
            cand: Dict[str, dict] = {}
            for li, ld in enumerate(lines):
                cmap = ld.get("commit_idx_to_sha") or {}
                n_loc = ld["n_commits"]
                for local_i in range(n_loc):
                    sha = cmap.get(local_i)
                    key = _key(li, local_i, sha)
                    tp = local_i + 1
                    rec = cand.get(key)
                    if rec is None:
                        rec = {"sha": sha, "lines": set(), "introducer": 0,
                               "depth": 0.0, "min_tp": tp, "order": len(cand)}
                        cand[key] = rec
                    rec["lines"].add(li)
                    rec["min_tp"] = min(rec["min_tp"], tp)
                    if local_i == n_loc - 1:
                        rec["introducer"] = 1
                    rec["depth"] = max(rec["depth"],
                                       tp / n_loc if n_loc > 1 else 1.0)

            # Order candidates most-recent-first so index 0 is closest to the
            # fixing commit and C-1 is oldest -- which is what the temporal PE
            # in CommitRankingModule documents. The old per-line concatenation
            # broke that (index 3 could be line 3's depth-0 commit).
            ordered = sorted(cand.items(),
                             key=lambda kv: (kv[1]["min_tp"], kv[1]["order"]))
            key_to_g = {key: gi for gi, (key, _) in enumerate(ordered)}
            n_global = len(ordered)

            # Pass 2 -- remap every line's nodes into the global candidate
            # space. Nodes belonging to the same SHA land on the same index, so
            # attention pooling merges their neighbourhoods into one candidate.
            emb_parts, idx_parts, tmp_parts = [], [], []
            for li, ld in enumerate(lines):
                cmap = ld.get("commit_idx_to_sha") or {}
                n_loc = ld["n_commits"]
                lut = torch.full((n_loc,), -1, dtype=torch.long)
                for local_i in range(n_loc):
                    lut[local_i] = key_to_g[_key(li, local_i, cmap.get(local_i))]
                ci = ld["commit_indices"].long()
                ok = (ci >= 0) & (ci < n_loc)
                if not bool(ok.any()):
                    continue
                emb_parts.append(ld["node_embeddings"][ok])
                idx_parts.append(lut[ci[ok]])
                tmp_parts.append(ld["is_temporal_node"][ok])

            if not emb_parts:
                results_by_k[k].append(_make_invalid(test_name))
                continue

            pooled_embeddings = torch.cat(emb_parts, dim=0)
            pooled_indices = torch.cat(idx_parts, dim=0)
            pooled_temporal = torch.cat(tmp_parts, dim=0)

            idx_to_sha = {key_to_g[key]: rec["sha"]
                          for key, rec in cand.items() if rec["sha"]}
            gt_global = {key_to_g[key] for key, rec in cand.items()
                         if rec["sha"] and rec["sha"] in unique_inducing_shas}

            # Compact away candidates that ended up with no nodes, so the
            # commit indices stay contiguous (the model derives its commit
            # count from the indices).
            present = torch.unique(pooled_indices)
            if int(present.numel()) != n_global:
                remap = torch.full((n_global,), -1, dtype=torch.long)
                remap[present] = torch.arange(present.numel(), dtype=torch.long)
                pooled_indices = remap[pooled_indices]
                old_to_new = {int(o): i for i, o in enumerate(present.tolist())}
                idx_to_sha = {old_to_new[o]: s for o, s in idx_to_sha.items()
                              if o in old_to_new}
                gt_global = {old_to_new[g] for g in gt_global if g in old_to_new}

            valid_gt = sorted(gt_global)
            is_correct = len(valid_gt) > 0

            results_by_k[k].append({
                "test_name":                test_name,
                "valid":                    True,
                "node_embeddings":          pooled_embeddings,
                "commit_indices":           pooled_indices,
                "is_temporal_node":         pooled_temporal,
                "ground_truth_positions":   valid_gt,
                "num_unique_vics":          num_unique_vics,
                "is_correct_deletion_line": is_correct,
                "p1_score":                 best_p1_score,
                "commit_idx_to_sha":        idx_to_sha,
            })
            n_valid_by_k[k] += 1
        



    return results_by_k