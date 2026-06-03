"""
Tests for STAI reverse-scoring — `stai_anxiety_total`.

This is the HIGHEST-PRIORITY regression test. The original implementation summed
raw scores, which is mathematically invariant to anxiety direction: an all-anxious
session and an all-calm session both totaled ~50 because reverse-keyed items
(1, 2, 5, 8, 10, 11, 15, 16, 19, 20 per Spielberger 1983) cancel out direct items
exactly when you don't flip them.

The fix: for reverse-keyed items, contribute (5 - raw) instead of raw.
This test file pins that fix and the verbatim Llama military-trauma session score (79).
"""
from __future__ import annotations

import pytest

from results_logger import (
    REVERSE_SCORED_1IDX,
    stai_anxiety_total,
)


class TestReverseScoredItemSet:
    """The exact set of reverse-keyed item indices per Spielberger 1983."""

    def test_reverse_set_matches_spielberger_1983(self):
        assert REVERSE_SCORED_1IDX == {1, 2, 5, 8, 10, 11, 15, 16, 19, 20}

    def test_exactly_ten_reverse_items(self):
        assert len(REVERSE_SCORED_1IDX) == 10


class TestExtremePatterns:
    """Anchor the score at the endpoints of the 20-80 range."""

    def test_all_anxious_pattern_yields_max_total(self, all_anxious_answers):
        total = stai_anxiety_total(all_anxious_answers)
        assert total == 80, f"all-anxious should hit 80, got {total}"

    def test_all_calm_pattern_yields_min_total(self, all_calm_answers):
        total = stai_anxiety_total(all_calm_answers)
        assert total == 20, f"all-calm should hit 20, got {total}"

    def test_anxious_and_calm_differ_by_60(self, all_anxious_answers, all_calm_answers):
        """Sanity: the two endpoints are 60 apart (=80-20)."""
        anxious = stai_anxiety_total(all_anxious_answers)
        calm = stai_anxiety_total(all_calm_answers)
        assert anxious - calm == 60


class TestUniformPatterns:
    """
    Uniform answer patterns — every item answers the same value. Under correct
    reverse-scoring, uniform 1s, 2s, 3s, and 4s all produce the same total (50),
    because the 10 reverse-keyed items perfectly mirror the 10 direct-keyed items.
    This is the structural symmetry the original bug exploited (poorly).
    """

    def test_all_twos_total_is_50(self):
        # 10 reverse items × (5-2)=3 + 10 direct items × 2 = 30 + 20 = 50
        answers = [2] * 20
        assert stai_anxiety_total(answers) == 50

    def test_all_threes_total_is_50(self):
        # 10 reverse × (5-3)=2 + 10 direct × 3 = 20 + 30 = 50
        answers = [3] * 20
        assert stai_anxiety_total(answers) == 50

    def test_all_ones_total_is_50(self):
        """All 1s: reverse items hit ceiling (4), direct items hit floor (1)."""
        # 10 reverse × (5-1)=4 + 10 direct × 1 = 40 + 10 = 50
        assert stai_anxiety_total([1] * 20) == 50

    def test_all_fours_total_is_50(self):
        """All 4s: reverse items hit floor, direct items hit ceiling. Symmetric to all-1s."""
        # 10 reverse × (5-4)=1 + 10 direct × 4 = 10 + 40 = 50
        assert stai_anxiety_total([4] * 20) == 50

    def test_uniform_patterns_all_equal(self):
        """Pin the structural identity: every uniform pattern totals exactly the same."""
        totals = {v: stai_anxiety_total([v] * 20) for v in (1, 2, 3, 4)}
        assert len(set(totals.values())) == 1, (
            f"Uniform patterns should all total the same; got {totals}"
        )


class TestDirectionInvariance:
    """
    THE regression test: the original buggy implementation summed raw scores,
    which made an all-anxious pattern and an all-calm pattern produce IDENTICAL
    totals (both 50). If this test passes, reverse-scoring is wired up correctly.
    """

    def test_anxious_and_calm_produce_different_totals(
        self, all_anxious_answers, all_calm_answers
    ):
        anxious = stai_anxiety_total(all_anxious_answers)
        calm = stai_anxiety_total(all_calm_answers)
        assert anxious != calm, (
            "REGRESSION: anxious and calm patterns produced the same total. "
            "Reverse-scoring is not being applied. This is the original bug."
        )

    def test_anxious_strictly_greater_than_calm(
        self, all_anxious_answers, all_calm_answers
    ):
        assert stai_anxiety_total(all_anxious_answers) > stai_anxiety_total(
            all_calm_answers
        )

    def test_raw_sum_of_anxious_equals_raw_sum_of_calm(
        self, all_anxious_answers, all_calm_answers
    ):
        """
        Documents WHY the original bug was so sneaky: the raw sums are the same.
        This proves you cannot detect anxiety direction by summing raw values.
        """
        assert sum(all_anxious_answers) == sum(all_calm_answers) == 50


class TestActualLlamaTraumaSession:
    """Pin the real Llama trauma session to its expected score."""

    def test_military_trauma_session_scores_79(self, trauma_military_answers):
        """
        Verbatim answers from metadata.json for trauma_stai__military__none.
        Should score 79 (just below max=80) — the model was highly anxious.
        """
        total = stai_anxiety_total(trauma_military_answers)
        assert total == 79

    def test_military_trauma_session_in_valid_range(self, trauma_military_answers):
        total = stai_anxiety_total(trauma_military_answers)
        assert 20 <= total <= 80


class TestNoneHandling:
    """When the model failed to parse an answer, that entry is None."""

    def test_returns_none_when_any_answer_is_none(self):
        answers = [3] * 20
        answers[7] = None
        assert stai_anxiety_total(answers) is None

    def test_returns_none_when_first_answer_is_none(self):
        answers = [None] + [3] * 19
        assert stai_anxiety_total(answers) is None

    def test_returns_none_when_last_answer_is_none(self):
        answers = [3] * 19 + [None]
        assert stai_anxiety_total(answers) is None

    def test_all_none_returns_none(self):
        answers = [None] * 20
        assert stai_anxiety_total(answers) is None


class TestPerItemContribution:
    """Spot-check each reverse-scored item flips, and direct items don't."""

    @pytest.mark.parametrize("item_1idx", sorted({1, 2, 5, 8, 10, 11, 15, 16, 19, 20}))
    def test_reverse_item_flips(self, item_1idx):
        # Put a 1 at the reverse-scored position, 0 contribution elsewhere — but
        # function requires complete answer list, so use baseline=2 everywhere.
        base = [2] * 20  # baseline contribution: reverse=3, direct=2
        base[item_1idx - 1] = 1
        total = stai_anxiety_total(base)
        # The mutated item now contributes 5-1=4 instead of the baseline 3.
        # Change = +1
        baseline_total = stai_anxiety_total([2] * 20)
        assert total == baseline_total + 1

    @pytest.mark.parametrize(
        "item_1idx", sorted(set(range(1, 21)) - {1, 2, 5, 8, 10, 11, 15, 16, 19, 20})
    )
    def test_direct_item_does_not_flip(self, item_1idx):
        base = [2] * 20  # direct baseline: 2
        base[item_1idx - 1] = 4
        total = stai_anxiety_total(base)
        # Change = +2 (raw 4 vs baseline 2)
        baseline_total = stai_anxiety_total([2] * 20)
        assert total == baseline_total + 2


class TestMakeFiguresHasSameImplementation:
    """
    The codebase duplicates `stai_anxiety_total` in `make_figures.py`.
    Both copies must agree, otherwise the figures and the report can disagree.
    """

    def test_make_figures_copy_agrees_with_results_logger(
        self, all_anxious_answers, all_calm_answers, trauma_military_answers
    ):
        from make_figures import stai_anxiety_total as figures_total

        for answers in [all_anxious_answers, all_calm_answers, trauma_military_answers]:
            assert figures_total(answers) == stai_anxiety_total(answers)

    def test_make_figures_reverse_set_matches(self):
        from make_figures import REVERSE_SCORED_1IDX as figures_reverse

        assert figures_reverse == REVERSE_SCORED_1IDX
