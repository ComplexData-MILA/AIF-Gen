"""Tests for the West-of-N style margin-controlled pair selector."""

import pytest

from aif_gen.generate.mappers.pair_selector import ScoredCandidate, select_pair


def _mk(text: str, score: float, persona: str = 'aligned') -> ScoredCandidate:
    return ScoredCandidate(text=text, score=score, persona=persona)


def test_select_pair_returns_none_for_too_few_candidates() -> None:
    assert select_pair([], target_margin=0.5) is None
    assert select_pair([_mk('a', 1.0)], target_margin=0.5) is None


def test_select_pair_target_margin_one_picks_widest_gap() -> None:
    cands = [
        _mk('a' * 10, 1.0, 'aligned'),
        _mk('b' * 10, 2.5, 'neutral'),
        _mk('c' * 10, 5.0, 'anti_aligned'),
    ]
    chosen, rejected = select_pair(cands, target_margin=1.0)  # type: ignore[misc]
    assert chosen.score == 5.0
    assert rejected.score == 1.0


def test_select_pair_target_margin_zero_picks_smallest_gap() -> None:
    cands = [
        _mk('a' * 10, 1.0),
        _mk('b' * 10, 1.2),
        _mk('c' * 10, 5.0),
    ]
    chosen, rejected = select_pair(cands, target_margin=0.0)  # type: ignore[misc]
    assert chosen.score - rejected.score == pytest.approx(0.2)


def test_select_pair_monotone_in_target_margin() -> None:
    cands = [_mk('x' * 10, float(s), 'aligned') for s in range(1, 6)]
    gaps = []
    for tm in (0.0, 0.25, 0.5, 0.75, 1.0):
        pair = select_pair(cands, target_margin=tm)
        assert pair is not None
        gaps.append(pair[0].score - pair[1].score)
    # The achieved gap must be (weakly) non-decreasing in target_margin.
    for a, b in zip(gaps, gaps[1:]):
        assert a <= b


def test_select_pair_orders_chosen_above_rejected() -> None:
    cands = [_mk('a' * 10, 4.5), _mk('b' * 10, 1.5)]
    chosen, rejected = select_pair(cands, target_margin=1.0)  # type: ignore[misc]
    assert chosen.score > rejected.score


def test_select_pair_min_margin_filters_too_close_pairs() -> None:
    cands = [_mk('a' * 10, 3.0), _mk('b' * 10, 3.05)]
    assert select_pair(cands, target_margin=0.0, min_margin=0.5) is None


def test_select_pair_length_ratio_band_filters_lopsided_pairs() -> None:
    short = _mk('x', 1.0)  # length 1
    long = _mk('x' * 100, 5.0)  # length 100, ratio = 100 > 2.0
    assert (
        select_pair([short, long], target_margin=1.0, length_ratio_band=(0.5, 2.0))
        is None
    )


def test_select_pair_prefers_different_personas_on_tie() -> None:
    # Two pairs with identical gap; selector should prefer the one with
    # different personas as a semantic contrast tie-breaker.
    cands = [
        _mk('a' * 10, 1.0, 'aligned'),
        _mk('b' * 10, 4.0, 'aligned'),  # same-persona pair: gap 3
        _mk('c' * 10, 2.0, 'anti_aligned'),
        _mk('d' * 10, 5.0, 'anti_aligned'),  # same-persona pair: gap 3
        _mk('e' * 10, 2.5, 'neutral'),  # different-persona pair with one above: gap 2.5
    ]
    chosen, rejected = select_pair(cands, target_margin=1.0)  # type: ignore[misc]
    # max gap is 5.0 - 1.0 = 4.0 between aligned and anti_aligned (different personas)
    assert chosen.persona != rejected.persona


def test_select_pair_invalid_target_margin_raises() -> None:
    with pytest.raises(ValueError):
        select_pair([_mk('a' * 10, 1.0), _mk('b' * 10, 2.0)], target_margin=1.5)
    with pytest.raises(ValueError):
        select_pair([_mk('a' * 10, 1.0), _mk('b' * 10, 2.0)], target_margin=-0.1)
