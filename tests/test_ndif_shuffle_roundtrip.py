"""
Adversarial round-trip tests for the option-shuffle pipeline in
`anxiety_hidden_state_experiment_ndif.py` (the single-pool ndif experiment).

`test_option_shuffling.py` already exercises the multipool `parse_answer` in
isolation. This file pins the harder invariant the experiment actually relies on:

    build_stai_messages(...) -> (prompt shown to model, num_to_label_idx)
    parse_answer(model_reply, num_to_label_idx) -> canonical score

The canonical meaning of a stored answer MUST be invariant to the per-item
label/number shuffle. Concretely: whatever option-number the model picks, the
score returned must equal (rank of the label printed next to that number) + 1,
where rank 0 = "almost never" ... rank 3 = "almost always". If shuffle and
un-shuffle ever drift apart, a calm answer is scored as anxious (or vice versa)
and the whole STAI total is corrupted while still looking plausible.

We also guard against silent inversion by checking that the score the parser
returns matches the label *text* the model literally saw — not the option number.
"""
from __future__ import annotations

import random

import pytest

from anxiety_hidden_state_experiment_ndif import (
    build_stai_messages,
    parse_answer,
)
from anxiety_hidden_state_multipool_ndif import parse_answer as parse_answer_multipool


CANONICAL_LABELS = ["almost never", "occasionally", "often", "almost always"]


@pytest.fixture
def calm_question() -> dict:
    return {"prompt": "I feel calm.", "labels": list(CANONICAL_LABELS)}


def _shuffle_for_seed(seed: int, n: int = 4):
    """Reproduce the exact RNG advancement run_condition uses for one item."""
    rng = random.Random(seed)
    option_order = list(range(n))
    rng.shuffle(option_order)
    option_nums = list(range(1, n + 1))
    rng.shuffle(option_nums)
    return option_order, option_nums


# ─── core round-trip: every option, every seed ──────────────────────────────
class TestShuffleRoundTrip:
    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 99, 123, 2024])
    def test_every_option_scores_to_its_label_rank(self, seed, calm_question):
        order, nums = _shuffle_for_seed(seed)
        _msgs, mapping = build_stai_messages("", "PRE", calm_question, order, nums)
        labels = calm_question["labels"]
        for option_num, label_idx in mapping.items():
            score = parse_answer(f"Option {option_num}", mapping)
            expected = label_idx + 1
            assert score == expected, (
                f"seed={seed}: Option {option_num} shows "
                f"{labels[label_idx]!r} (rank {label_idx}); score should be "
                f"{expected}, got {score}"
            )

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
    def test_score_matches_label_text_seen_not_number(self, seed, calm_question):
        """
        The decisive anti-inversion check: parse_answer's output must agree with
        the canonical RANK of the label the model literally read next to that
        number — even when the option *number* differs from that rank.
        """
        order, nums = _shuffle_for_seed(seed)
        _msgs, mapping = build_stai_messages("", "PRE", calm_question, order, nums)
        labels = calm_question["labels"]
        for option_num, label_idx in mapping.items():
            label_text = labels[label_idx]
            canonical_rank = CANONICAL_LABELS.index(label_text) + 1
            assert parse_answer(f"Option {option_num}", mapping) == canonical_rank

    def test_prompt_and_mapping_are_consistent(self, calm_question):
        """
        Parse the option lines straight out of the rendered prompt and confirm
        each printed 'Option K: <label>' agrees with num_to_label_idx. Catches a
        build_stai_messages bug where the text and the mapping disagree.
        """
        import re

        order, nums = _shuffle_for_seed(42)
        msgs, mapping = build_stai_messages("", "PRE", calm_question, order, nums)
        prompt = msgs[-1]["content"]
        labels = calm_question["labels"]
        for line in re.findall(r"Option (\d+): (.+?)\.", prompt):
            num = int(line[0])
            label_text = line[1]
            assert labels[mapping[num]] == label_text


# ─── failure modes: unparseable / out-of-range -> None, never a wrong score ──
class TestParseFailureModes:
    @pytest.fixture
    def identity_mapping(self):
        # Option k -> label rank k-1
        return {1: 0, 2: 1, 3: 2, 4: 3}

    def test_option_5_not_shown_returns_none(self, identity_mapping):
        """'Option 5' was never an option — must map to None, not a wrong score."""
        assert parse_answer("Option 5", identity_mapping) is None

    def test_option_0_returns_none(self, identity_mapping):
        assert parse_answer("Option 0", identity_mapping) is None

    def test_unparseable_returns_none(self, identity_mapping):
        assert parse_answer("I would rather not say.", identity_mapping) is None

    def test_empty_returns_none(self, identity_mapping):
        assert parse_answer("", identity_mapping) is None

    def test_bare_number_without_option_word_returns_none(self, identity_mapping):
        """The model must say 'Option N'; a lone digit is not accepted."""
        assert parse_answer("3", identity_mapping) is None

    def test_huge_option_number_returns_none(self, identity_mapping):
        assert parse_answer("Option 9999", identity_mapping) is None

    def test_last_option_mentioned_wins(self, identity_mapping):
        """If the model rambles then answers, the final 'Option N' is used."""
        text = "Hmm, Option 1 seems plausible, but actually Option 4."
        assert parse_answer(text, identity_mapping) == 4


# ─── the two ndif modules must parse identically ─────────────────────────────
class TestCrossModuleParserAgreement:
    """
    The single-pool and multipool ndif scripts each define their own
    `parse_answer`. They must behave identically or the two experiments disagree
    on what a stored answer means.
    """

    @pytest.mark.parametrize("seed", [0, 7, 42, 2024])
    def test_both_parsers_agree_on_round_trip(self, seed, calm_question):
        order, nums = _shuffle_for_seed(seed)
        _msgs, mapping = build_stai_messages("", "PRE", calm_question, order, nums)
        for option_num in mapping:
            reply = f"Option {option_num}"
            assert parse_answer(reply, mapping) == parse_answer_multipool(reply, mapping)

    @pytest.mark.parametrize(
        "reply", ["Option 5", "Option 0", "garbage", "", "Option 2.", "option 3"]
    )
    def test_both_parsers_agree_on_edge_inputs(self, reply):
        mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        assert parse_answer(reply, mapping) == parse_answer_multipool(reply, mapping)
