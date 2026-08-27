"""
data/phase1/pairs.py

DeletionLinePair, TestCaseBatch, combine_testcases_to_batches — pairwise
training primitives for Phase 1 ranking.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List

from data_processing.phase1.minigraph import MiniGraph

@dataclass
class DeletionLinePair:
    """
    Pairwise training example.

    prob = 1.0  →  x should rank higher than y  (x is rootcause)
    prob = 0.0  →  y should rank higher than x  (y is rootcause)
    prob = 0.5  →  tied (both same class)
    """
    x:    MiniGraph
    y:    MiniGraph
    prob: float


@dataclass
class TestCaseBatch:
    test_cases: List[str] = field(default_factory=list)
    mini_graphs: List[MiniGraph] = field(default_factory=list)
    pairs: List[DeletionLinePair] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pairs)

def build_pairs(
    graphs: List[MiniGraph], max_pairs: int = 100
) -> List[DeletionLinePair]:
    """
    Generate CONTRASTIVE pairwise training examples from a list of MiniGraphs.

    Emits only rootcause-vs-non-rootcause pairs (prob 1.0 / 0.0), with the
    orientation randomised so both directions occur.

    Why ties are NOT emitted
    ------------------------
    The previous implementation emitted prob=0.5 for same-class pairs. The
    loss is ``BCE(sigmoid(s_x - s_y), target)``, and a 0.5 target is minimised
    exactly when ``s_x == s_y`` — so tied pairs do not merely fail to teach,
    they actively train the ranker toward assigning every deletion line an
    identical score. They accounted for 71% of all pairs (median 100% per
    case), which is why Phase-1 train loss sat near the ln(2)≈0.693 tie floor
    and val F1 peaked at epoch 2. This is the same degeneracy that was found
    and removed in Phase 2's LabelSmoothingRankingLoss.

    Why orientation is randomised
    -----------------------------
    The previous ordering (``pos + neg``, iterating i<j) made ``prob=0.0``
    unreachable: the left element of a contrastive pair was always the
    rootcause. Every informative example therefore had target 1.0, letting
    the ranker exploit position rather than content.

    Why sampling is uniform over the product
    ----------------------------------------
    The old nested loop truncated at ``max_pairs`` after walking positives in
    order, so a case with many rootcause lines spent its whole budget on the
    first one or two positives. Sampling uniformly over pos x neg gives every
    rootcause line representation.

    Returns ``[]`` when the case has no contrast (all-rootcause, or
    all-non-rootcause, or a single deletion line) — such a case cannot teach a
    ranker anything. Roughly 65% of cases fall in this group; they are still
    scored at inference and still usable by Phase 2.
    """
    pos = [g for g in graphs if g.rootcause]
    neg = [g for g in graphs if not g.rootcause]

    if not pos or not neg:
        return []

    product = [(p, n) for p in pos for n in neg]
    if len(product) > max_pairs:
        product = random.sample(product, max_pairs)

    pairs: List[DeletionLinePair] = []
    for p, n in product:
        if random.random() < 0.5:
            pairs.append(DeletionLinePair(p, n, 1.0))   # x=rootcause ranks higher
        else:
            pairs.append(DeletionLinePair(n, p, 0.0))   # y=rootcause ranks higher
    return pairs


def combine_testcases_to_batches(
    dataset,                          # DeletionLineDataset
    cases: List[str],
    pairs_cache: Dict[str, List[DeletionLinePair]],
    max_graphs_per_batch: int = 40,   # VRAM control: max graphs encoded at once
) -> List[TestCaseBatch]:
    """
    Group test cases into TestCaseBatches capped by ``max_graphs_per_batch``.

    Parameters
    ----------
    dataset              : DeletionLineDataset (provides mini_graphs dict)
    cases                : ordered list of test-case names to batch
    pairs_cache          : pre-built {test_name: List[DeletionLinePair]}
    max_graphs_per_batch : maximum number of graphs encoded in one forward pass
    """
    batches: List[TestCaseBatch] = []
    current_graphs: List[MiniGraph]        = []
    current_cases:  List[str]              = []
    current_pairs:  List[DeletionLinePair] = []

    for name in cases:
        mgs = dataset.mini_graphs.get(name, [])
        if not mgs:
            continue

        # Seal current batch if adding this test case would exceed VRAM limit
        if current_graphs and len(current_graphs) + len(mgs) > max_graphs_per_batch:
            batches.append(TestCaseBatch(
                test_cases=current_cases,
                mini_graphs=current_graphs,
                pairs=current_pairs,
            ))
            current_graphs = []
            current_cases  = []
            current_pairs  = []

        current_cases.append(name)
        current_graphs.extend(mgs)
        current_pairs.extend(pairs_cache.get(name, []))

    # Don't forget the last batch
    if current_graphs:
        batches.append(TestCaseBatch(
            test_cases=current_cases,
            mini_graphs=current_graphs,
            pairs=current_pairs,
        ))

    return batches