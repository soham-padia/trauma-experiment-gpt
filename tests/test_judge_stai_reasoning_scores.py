"""
Adversarial tests for `parse_scores` in `src/judge_stai_reasoning.py`.

`parse_scores(text, n)` extracts a JSON array of n integer anxiety scores (0-100)
from a judge's reply. It feeds `mean_anxiety` per (variation, condition). A silently
wrong parse here corrupts the channel-2 anxiety means.

Contract under test (the user's stated requirements):
  - malformed array / no array        -> raise (never default)
  - wrong length                       -> raise (never pad/truncate silently)
  - non-numeric / negative tokens      -> regex fails to match -> raise
  - out-of-range positive values       -> CLAMPED to 0-100 (documented behaviour)

NOTE ON A SHARP EDGE: the extraction regex is r"\\[[\\s\\d,]+\\]". It only matches
digits, whitespace, and commas between the brackets. This has two consequences we
pin below as regression guards:
  1. A '-' (negative number) breaks the match -> "no array" ValueError, NOT a
     clamp. So negative scores never reach the clamp; they hard-fail. Good
     (fail-loud), but easy to break by "improving" the regex.
  2. A '.' (float) also breaks the match -> ValueError. The judge is asked for
     integers, so floats are rejected outright.
"""
from __future__ import annotations

import pytest

from judge_stai_reasoning import parse_scores


# ─── Happy path ──────────────────────────────────────────────────────────────
class TestParseScoresHappyPath:
    def test_simple_array(self):
        assert parse_scores("[10, 20, 30]", 3) == [10, 20, 30]

    def test_array_embedded_in_prose(self):
        assert parse_scores("Here are my scores: [5, 50, 95] done.", 3) == [5, 50, 95]

    def test_strips_think_block(self):
        text = "<think>I'll rate sentence 1 as 99...</think>[10, 20, 30]"
        assert parse_scores(text, 3) == [10, 20, 30]

    def test_think_block_array_is_ignored_real_array_wins(self):
        """A decoy array inside <think> must not be the one parsed."""
        text = "<think>scratchpad [1, 2, 3]</think>\nFinal: [40, 50, 60]"
        assert parse_scores(text, 3) == [40, 50, 60]

    def test_endpoints_pass_unchanged(self):
        assert parse_scores("[0, 100]", 2) == [0, 100]

    def test_single_element(self):
        assert parse_scores("[42]", 1) == [42]


# ─── Length validation ───────────────────────────────────────────────────────
class TestLengthValidation:
    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="expected 3 scores, got 2"):
            parse_scores("[10, 20]", 3)

    def test_too_many_raises(self):
        with pytest.raises(ValueError, match="expected 2 scores, got 3"):
            parse_scores("[10, 20, 30]", 2)

    def test_off_by_one_short(self):
        with pytest.raises(ValueError):
            parse_scores("[" + ",".join(["50"] * 19) + "]", 20)

    def test_off_by_one_long(self):
        with pytest.raises(ValueError):
            parse_scores("[" + ",".join(["50"] * 21) + "]", 20)

    def test_exact_twenty_passes(self):
        out = parse_scores("[" + ",".join(["50"] * 20) + "]", 20)
        assert len(out) == 20


# ─── Malformed / missing array ───────────────────────────────────────────────
class TestMalformedInput:
    def test_no_array_raises(self):
        with pytest.raises(ValueError, match="no array"):
            parse_scores("I cannot rate these.", 3)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="no array"):
            parse_scores("", 3)

    def test_only_think_block_no_array_raises(self):
        """After stripping <think>, nothing parseable remains."""
        with pytest.raises(ValueError):
            parse_scores("<think>[1,2,3] but I won't answer</think>", 3)

    def test_prose_with_lone_numbers_no_brackets_raises(self):
        with pytest.raises(ValueError, match="no array"):
            parse_scores("scores are 10 20 30", 3)


# ─── Non-numeric tokens ──────────────────────────────────────────────────────
class TestNonNumericTokens:
    def test_alpha_token_breaks_match_and_raises(self):
        """'[10, abc, 20]' — the regex won't match across letters -> no array."""
        with pytest.raises(ValueError, match="no array"):
            parse_scores("[10, abc, 20]", 3)

    def test_null_token_raises(self):
        with pytest.raises(ValueError):
            parse_scores("[10, null, 20]", 3)


# ─── Negative values: must hard-fail, never clamp to 0 ───────────────────────
class TestNegativeValuesHardFail:
    def test_single_negative_raises_not_clamped(self):
        """
        A negative score must NOT silently become 0 (which would bias the mean
        toward calm). The minus sign breaks the digit-only regex, so it raises.
        """
        with pytest.raises(ValueError, match="no array"):
            parse_scores("[-5]", 1)

    def test_negative_among_valid_raises(self):
        with pytest.raises(ValueError):
            parse_scores("[10, -20, 30]", 3)


# ─── Float values: rejected (integers requested) ─────────────────────────────
class TestFloatValuesRejected:
    def test_float_breaks_match_and_raises(self):
        with pytest.raises(ValueError, match="no array"):
            parse_scores("[10.5, 20.0, 30.0]", 3)


# ─── Out-of-range positives: documented CLAMP behaviour ──────────────────────
class TestOutOfRangePositiveClamps:
    """
    Unlike negatives (which hard-fail via the regex), positive out-of-range
    values DO reach the clamp because digits-only matches them. They are clamped
    into 0-100 rather than raising. We pin this so the asymmetry is intentional
    and visible: 150 -> 100, not an error, not 0.
    """

    def test_above_100_clamped_to_100(self):
        assert parse_scores("[150, 99, 0]", 3) == [100, 99, 0]

    def test_all_huge_clamped(self):
        assert parse_scores("[999, 1000, 12345]", 3) == [100, 100, 100]

    def test_in_range_high_values_unchanged(self):
        assert parse_scores("[100, 100, 100]", 3) == [100, 100, 100]


# ─── Regex boundary: which array is captured ─────────────────────────────────
class TestArrayCaptureBoundary:
    def test_first_array_captured_when_separated_by_letters(self):
        """
        Two arrays separated by non-digit/comma/space text: the regex stops at
        the first ']' because 'and' breaks the [\\s\\d,]+ run. The first array is
        captured (length checked against n).
        """
        assert parse_scores("[10, 20] and [30, 40]", 2) == [10, 20]

    def test_two_arrays_merge_when_only_whitespace_between(self):
        """
        DANGER ZONE pinned: '[10, 20] [30, 40]' — between ']' and '[' there is
        only whitespace, which IS in the charset... but ']' and '[' are not, so
        the run actually stops at the first ']'. Confirm the captured array is
        just the first one (length 2), not a merge.
        """
        assert parse_scores("[10, 20] [30, 40]", 2) == [10, 20]
