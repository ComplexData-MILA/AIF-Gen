"""Pair selection over scored response candidates.

Implements the West-of-N style margin-controlled pair selection
(arXiv:2401.12086) generalized so that the user can target any margin in
[0, 1] rather than always picking max-vs-min, plus a length-ratio filter to
mitigate the well-known length-hacking failure mode in RLHF reward models
(Singhal et al., arXiv:2310.03716).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ScoredCandidate:
    r"""A single response candidate paired with its rubric score and origin metadata.

    Args:
        text (str): The candidate response text.
        score (float): The aggregated rubric score (typically in [1, 5]).
        persona (str): Generation persona used for this candidate
            (e.g., 'aligned', 'anti_aligned', 'neutral'). Used as a tie-breaker
            to encourage semantic, not stylistic, contrast in the chosen pair.
    """

    text: str
    score: float
    persona: str


def select_pair(
    candidates: List[ScoredCandidate],
    target_margin: float,
    min_margin: float = 0.0,
    max_margin: Optional[float] = None,
    length_ratio_band: Tuple[float, float] = (0.5, 2.0),
) -> Optional[Tuple[ScoredCandidate, ScoredCandidate]]:
    r"""Select a (chosen, rejected) pair from N scored candidates by score margin.

    The score gap of the selected pair targets:
        target_gap = (s_max - s_min) * target_margin
    where ``target_margin=1.0`` gives the easiest pair (best vs worst, the
    classic West-of-N criterion) and ``target_margin → 0`` gives the hardest
    pair (smallest admissible score gap). Pairs outside ``[min_margin, max_margin]``
    on the raw score scale, or outside ``length_ratio_band`` on the
    chosen/rejected length ratio, are rejected.

    Args:
        candidates: Pool of scored response candidates. Must contain at least 2.
        target_margin: Difficulty knob in [0, 1]. 0 = hardest, 1 = easiest.
        min_margin: Minimum admissible raw score gap. Pairs below are dropped
            as ambiguous.
        max_margin: Maximum admissible raw score gap. None disables the cap.
        length_ratio_band: (lo, hi) bounds on len(chosen)/len(rejected).

    Returns:
        (chosen, rejected) tuple, or None if no admissible pair exists.
    """
    if not 0.0 <= target_margin <= 1.0:
        raise ValueError(f'target_margin must be in [0, 1], got {target_margin}')
    if len(candidates) < 2:
        return None

    sorted_cands = sorted(candidates, key=lambda c: c.score)
    s_min, s_max = sorted_cands[0].score, sorted_cands[-1].score
    target_gap = (s_max - s_min) * target_margin

    lo, hi = length_ratio_band

    best: Optional[Tuple[ScoredCandidate, ScoredCandidate]] = None
    best_key: Optional[Tuple[float, int]] = None

    n = len(sorted_cands)
    for i in range(n):
        for j in range(i + 1, n):
            low, high = sorted_cands[i], sorted_cands[j]
            gap = high.score - low.score
            if gap < min_margin:
                continue
            if max_margin is not None and gap > max_margin:
                continue
            len_low = max(len(low.text), 1)
            len_high = max(len(high.text), 1)
            ratio = len_high / len_low
            if not (lo <= ratio <= hi):
                continue

            # Primary key: distance from target gap.
            # Secondary key: prefer pairs from *different* personas (encourages
            # semantic, not stylistic, contrast). 0 = different, 1 = same.
            distance = abs(gap - target_gap)
            same_persona = int(low.persona == high.persona)
            key = (distance, same_persona)
            if best_key is None or key < best_key:
                best_key = key
                best = (high, low)  # (chosen=higher score, rejected=lower)

    return best
