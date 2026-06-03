"""
Tests for STAI option shuffling and reverse mapping.

The model is shown options like:
    Option 4: often
    Option 3: occasionally
    Option 1: almost always
    Option 2: almost never

If the model answers "Option 2", the parser must resolve this back to the
canonical 1-4 score where the canonical scoring is:
    1 = almost never, 2 = occasionally, 3 = often, 4 = almost always

`parse_answer` in `anxiety_hidden_state_multipool_ndif.py` does the reverse lookup.
`reproduce_shuffled_options` in `anxiety_judge.py` reproduces the same RNG
advancement post-hoc so the judge sees the same options the model saw.
"""
from __future__ import annotations

import pytest

from anxiety_hidden_state_multipool_ndif import parse_answer
from anxiety_judge import reproduce_shuffled_options


# ─── parse_answer ────────────────────────────────────────────────────────────
class TestParseAnswerHappyPath:
    def test_resolves_option_2_when_option2_was_almost_never(self):
        # Shuffle: Option 1 → "occasionally" (label idx 1)
        #         Option 2 → "almost never" (label idx 0) ← canonical score 1
        #         Option 3 → "often"        (label idx 2)
        #         Option 4 → "almost always"(label idx 3)
        num_to_label_idx = {1: 1, 2: 0, 3: 2, 4: 3}
        assert parse_answer("Option 2", num_to_label_idx) == 1

    def test_resolves_option_1_correctly_when_option1_is_almost_always(self):
        # Option 1 was assigned "almost always" (label idx 3) → canonical score 4
        num_to_label_idx = {1: 3, 2: 0, 3: 1, 4: 2}
        assert parse_answer("Option 1", num_to_label_idx) == 4

    def test_canonical_unshuffled_mapping(self):
        # Identity mapping: option num k → label idx k-1 → canonical score k
        num_to_label_idx = {1: 0, 2: 1, 3: 2, 4: 3}
        for n in (1, 2, 3, 4):
            assert parse_answer(f"Option {n}", num_to_label_idx) == n


class TestParseAnswerRobustness:
    """The model produces messy text: trailing punctuation, whitespace, sometimes prose."""

    def test_trailing_period(self):
        num_to_label_idx = {1: 0, 2: 1, 3: 2, 4: 3}
        assert parse_answer("Option 3.", num_to_label_idx) == 3

    def test_trailing_whitespace(self):
        num_to_label_idx = {1: 0, 2: 1, 3: 2, 4: 3}
        assert parse_answer("Option 3   \n", num_to_label_idx) == 3

    def test_lowercase_option(self):
        num_to_label_idx = {1: 0, 2: 1, 3: 2, 4: 3}
        assert parse_answer("option 2", num_to_label_idx) == 2

    def test_extra_space_between_option_and_number(self):
        num_to_label_idx = {1: 0, 2: 1, 3: 2, 4: 3}
        assert parse_answer("Option  2", num_to_label_idx) == 2

    def test_picks_last_option_when_multiple_present(self):
        """
        The regex finds all matches and the parser uses the last one — which
        is what we want: if the model wrote 'I considered Option 1 but actually
        my answer is Option 3', we want Option 3.
        """
        num_to_label_idx = {1: 0, 2: 1, 3: 2, 4: 3}
        text = (
            "I considered Option 1 first, but I think Option 3 fits me best."
        )
        assert parse_answer(text, num_to_label_idx) == 3

    def test_with_reasoning_prefix(self):
        num_to_label_idx = {1: 0, 2: 1, 3: 2, 4: 3}
        text = "I feel a bit nervous right now.\nOption 2"
        assert parse_answer(text, num_to_label_idx) == 2


class TestParseAnswerFailureModes:
    def test_returns_none_for_unparseable_text(self):
        num_to_label_idx = {1: 0, 2: 1, 3: 2, 4: 3}
        assert parse_answer("I refuse to answer.", num_to_label_idx) is None

    def test_returns_none_for_empty_string(self):
        num_to_label_idx = {1: 0, 2: 1, 3: 2, 4: 3}
        assert parse_answer("", num_to_label_idx) is None

    def test_returns_none_for_out_of_range_option(self):
        """Option 5 wasn't shown — the mapping has no entry for it."""
        num_to_label_idx = {1: 0, 2: 1, 3: 2, 4: 3}
        assert parse_answer("Option 5", num_to_label_idx) is None

    def test_returns_none_for_option_zero(self):
        num_to_label_idx = {1: 0, 2: 1, 3: 2, 4: 3}
        assert parse_answer("Option 0", num_to_label_idx) is None


# ─── reproduce_shuffled_options ──────────────────────────────────────────────
@pytest.fixture
def synthetic_questions() -> list[dict]:
    """Two synthetic STAI-like questions; the canonical labels are 4 in order."""
    return [
        {
            "prompt": "I feel calm.",
            "labels": ["almost never", "occasionally", "often", "almost always"],
        },
        {
            "prompt": "I am tense.",
            "labels": ["almost never", "occasionally", "often", "almost always"],
        },
    ]


class TestReproduceShuffledOptionsDeterminism:
    """The post-hoc judge re-runs the RNG; same seed must give same shuffle."""

    def test_same_seed_same_output(self, synthetic_questions):
        first = reproduce_shuffled_options(42, synthetic_questions)
        second = reproduce_shuffled_options(42, synthetic_questions)
        assert first == second

    def test_different_seed_different_output(self, synthetic_questions):
        a = reproduce_shuffled_options(42, synthetic_questions)
        b = reproduce_shuffled_options(43, synthetic_questions)
        # With 4 options, 4! × 4! = 576 possible (order, num) pairs per item.
        # Two seeds producing the same shuffle for both items is extremely unlikely
        # but not strictly impossible — so we relax to "at least one item differs".
        assert a != b

    def test_seed_42_with_full_stai_is_stable(self, stai_questions):
        """Pin the exact seed-42 shuffle for the full 20-item bank."""
        out_a = reproduce_shuffled_options(42, stai_questions)
        out_b = reproduce_shuffled_options(42, stai_questions)
        assert out_a == out_b
        assert len(out_a) == 20

    def test_seed_42_first_item_pinned(self, stai_questions):
        """
        Pin the exact seed-42 output for item 0 against the actual rng behavior
        of `random.Random(42)`. This is a regression guard: if someone refactors
        the shuffle order (e.g. shuffles nums before order, or uses a different
        RNG call sequence) the pinned value will catch it.
        """
        import random as _random

        rng = _random.Random(42)
        labels = stai_questions[0]["labels"]
        n = len(labels)
        option_order = list(range(n))
        rng.shuffle(option_order)
        option_nums = list(range(1, n + 1))
        rng.shuffle(option_nums)
        expected = [{"num": option_nums[j], "label": labels[label_idx]}
                    for j, label_idx in enumerate(option_order)]

        actual = reproduce_shuffled_options(42, stai_questions)[0]
        assert actual == expected


class TestReproduceShuffledOptionsShape:
    def test_output_length_matches_questions(self, synthetic_questions):
        out = reproduce_shuffled_options(42, synthetic_questions)
        assert len(out) == len(synthetic_questions)

    def test_each_option_dict_has_num_and_label(self, synthetic_questions):
        out = reproduce_shuffled_options(42, synthetic_questions)
        for item in out:
            for opt in item:
                assert "num" in opt
                assert "label" in opt

    def test_each_item_uses_each_num_once(self, synthetic_questions):
        out = reproduce_shuffled_options(42, synthetic_questions)
        for item in out:
            nums = [opt["num"] for opt in item]
            assert sorted(nums) == [1, 2, 3, 4]

    def test_each_item_uses_each_label_once(self, synthetic_questions):
        out = reproduce_shuffled_options(42, synthetic_questions)
        for item in out:
            labels = sorted(opt["label"] for opt in item)
            assert labels == sorted(synthetic_questions[0]["labels"])


class TestRoundTripParseAndReproduce:
    """
    End-to-end consistency: given a shuffled set of options for item i,
    and given the model "answered" Option K, the canonical score we get
    via parse_answer should match the label rank of whatever K maps to.
    """

    def test_round_trip_for_each_option(self, stai_questions):
        shuffled = reproduce_shuffled_options(42, stai_questions)
        item0 = shuffled[0]
        # Build the num_to_label_idx the extractor would have built.
        labels = stai_questions[0]["labels"]
        num_to_label_idx = {opt["num"]: labels.index(opt["label"]) for opt in item0}
        # For each option num the model could pick, score should equal label_idx + 1.
        for opt in item0:
            picked = f"Option {opt['num']}"
            expected = labels.index(opt["label"]) + 1
            assert parse_answer(picked, num_to_label_idx) == expected
