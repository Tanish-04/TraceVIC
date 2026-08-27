"""
models/phase2_model.py
───────────────────────
Phase 2 model: commit ranking head.

  CommitRankingModule  — trainable head: node embeddings → commit embeddings → scores

Architecture
────────────
Pre-computed node embeddings [N, hidden_dim]
    (produced by the frozen Phase 1 encoder; temporal PE already applied)
    ↓  multi-head attention pooling     (nodes per commit → one vector)
[C, hidden_dim]
    ↓  TransformerEncoder               (commit-level sequence modelling)
    ↓  RankingHead  (Linear → GELU → Dropout → Linear)
scores [C]

Ablation flag
─────────────
use_dual_query=True  (default) — correspondence-aware dual query:
    temporal nodes   → temporal_query  [H, D_h]
    non-temporal     → general_query   [H, D_h]
    Tests whether distinguishing V-SZZ-traced nodes at the attention level matters.

use_dual_query=False — single shared query ablation:
    all nodes        → shared_query    [H, D_h]   (is_temporal_node is ignored)
    Removes the temporal/general split; serves as the ablation baseline for the
    "single shared query vs. correspondence-aware (dual) query" experiment.
"""
import torch
import torch.nn as nn


class CommitRankingModule(nn.Module):
    """
    Trainable commit ranking head operating on pre-computed node embeddings.
    """

    def __init__(self, 
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_commit_transformer_layers: int = 1,
        dropout: float = 0.3,
        max_temporal_dist: int = 300,
        use_dual_query: bool = False,
        use_temporal_pe: bool = True,
    ):
        """
        Parameters
        ----------
        use_dual_query : bool
            True  — correspondence-aware dual query (temporal_query / general_query).
            False — single shared query ablation (shared_query; is_temporal_node ignored).
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_dual_query = use_dual_query
        self.use_temporal_pe = use_temporal_pe
        

        # input_dim=768 (Phase 1 output) → hidden_dim=256 (Phase 2 internal)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if use_dual_query:
            # Correspondence-aware: separate learnable queries per node type
            self.temporal_query = nn.Parameter(torch.randn(num_heads, self.head_dim))
            self.general_query  = nn.Parameter(torch.randn(num_heads, self.head_dim))
        else:
            # Single shared query ablation: one query for all nodes
            self.shared_query = nn.Parameter(torch.randn(num_heads, self.head_dim))


        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj_pool = nn.Linear(hidden_dim, hidden_dim)
        self.norm_pool = nn.LayerNorm(hidden_dim)

        self.distance_embedding = nn.Embedding(max_temporal_dist, hidden_dim)

        # Explicit per-candidate features (multiplicity / introducer / depth).
        # Multiplicity in particular is signal that SHA dedup removes from the
        # labels, so it is fed in here instead of leaking through duplicates.


        # Commit-level Transformer
        self.commit_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation="gelu",
            ),
            num_layers=num_commit_transformer_layers,
        )
        

        # Final Ranking head
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )



    @staticmethod
    def _build_block_diagonal_mask(commit_counts, device):
        """Build a float attention mask that blocks cross-case attention.

        Returns a [C_total, C_total] mask where allowed positions are 0.0
        and blocked positions are -inf, matching PyTorch TransformerEncoder's
        additive mask convention.
        """
        C_total = sum(commit_counts)
        mask = torch.full((C_total, C_total), float("-inf"), device=device)
        offset = 0
        for c in commit_counts:
            mask[offset:offset + c, offset:offset + c] = 0.0
            offset += c
        return mask

    def forward(self,
                node_embeddings: torch.Tensor,
                commit_indices: torch.Tensor,
                is_temporal_node: torch.Tensor,
                commit_counts: list = None,
        ) -> torch.Tensor:
        """
        Args:
            node_embeddings  : [N, hidden_dim]  pre-computed by frozen encoder
            commit_indices   : [N]              which commit each node belongs to
            is_temporal_node : [N]              bool mask — True for nodes that
                               are targets of TEMPORAL_FWD edges (V-SZZ traced)
            commit_counts    : list of int, one per case in the batch — how many
                               commits each case contributes. Used to build a
                               block-diagonal attention mask so the transformer
                               only attends within each case.
                               (multiplicity fraction, is_introducer, depth
                               fraction). Rows must align with commit indices.
        Returns:
            scores : [C]  per-commit ranking scores
        """
        device = node_embeddings.device
        N = node_embeddings.size(0)
        num_commits = commit_indices.max().item() + 1

        # node_embeddings: [N, input_dim] → [N, hidden_dim]
        node_embeddings = self.input_proj(node_embeddings)

        # Project all nodes to keys and values in one pass
        K_all = self.k_proj(node_embeddings).view(N, self.num_heads, self.head_dim) # [N, H, D_h]
        V_all = self.v_proj(node_embeddings).view(N, self.num_heads, self.head_dim) # [N, H, D_h]


        # Build per-node query vectors [N, H, D_h]
        if self.use_dual_query:
            # Correspondence-aware: temporal nodes use temporal_query, others general_query
            mask = is_temporal_node.view(N, 1, 1).expand(N, self.num_heads, self.head_dim)
            queries = torch.where(
                mask,
                self.temporal_query.unsqueeze(0).expand(N, -1, -1),
                self.general_query.unsqueeze(0).expand(N, -1, -1),
            )  # [N, H, D_h]
        else:
            # Single shared query ablation: same query for every node
            queries = self.shared_query.unsqueeze(0).expand(N, -1, -1)  # [N, H, D_h]

        # Attention logits using per-node query
        logits = (queries * K_all).sum(dim=-1) * self.scale # [N, H]

        idx = commit_indices.unsqueeze(1).expand_as(logits)  # [N, H]

        logits_stable = logits - logits.max().detach()
        exp_logits = torch.exp(logits_stable)   # [N, H]

        # per-commit sum of exp — shape [C, H]
        exp_sum = torch.zeros(num_commits, self.num_heads, device=device)
        exp_sum.scatter_add_(0, idx, exp_logits)

        # normalize: divide each node's exp by its commit's sum
        # exp_sum[commit_indices] broadcasts sum back to each node
        attn_weights = exp_logits / (exp_sum[commit_indices] + 1e-9)  # [N, H]

        # Step 3 — scatter weighted sum of values
        # attn_weights: [N, H], V_all: [N, H, D_h]
        # weighted values: [N, H, D_h]
        weighted_V = attn_weights.unsqueeze(-1) * V_all   # [N, H, D_h]

        # sum weighted values within each commit → [C, H, D_h]
        idx_v = commit_indices.view(N, 1, 1).expand_as(weighted_V)

        pooled = torch.zeros(num_commits, self.num_heads, self.head_dim, device=device)
        pooled.scatter_add_(0, idx_v, weighted_V)   # [C, H, D_h]

        # Step 4 — project and normalize
        # reshape [C, H, D_h] → [C, H*D_h] = [C, hidden_dim]
        # then linear projection + LayerNorm
        commit_embeddings = self.norm_pool(self.out_proj_pool(pooled.reshape(num_commits, -1))) # [C, hidden_dim]

        # Temporal position encoding — positions reset per case
        if self.use_temporal_pe:
            if commit_counts is not None:
                per_case_pos = []
                for c in commit_counts:
                    per_case_pos.append(torch.arange(c, device=device))
                positions = torch.cat(per_case_pos)
            else:
                positions = torch.arange(num_commits, device=device)
            positions = positions.clamp(max=self.distance_embedding.num_embeddings - 1)
            temporal_pe = self.distance_embedding(positions)
            commit_embeddings = commit_embeddings + temporal_pe


        # Block-diagonal mask so transformer only attends within each case
        if commit_counts is not None and len(commit_counts) > 1:
            attn_mask = self._build_block_diagonal_mask(commit_counts, device)
        else:
            attn_mask = None

        # Transformer + ranking head
        x = commit_embeddings.unsqueeze(1)                          # [C, 1, D] (seq, batch, dim)
        x = self.commit_transformer(x, mask=attn_mask).squeeze(1)   # [C, D]
        return self.ranking_head(x).squeeze(-1)                     # [C]