"""
Tests for `build_behavioral_anxiety_prompt` in `src/anxiety_judge.py`.

The behavioral judge is the "proper" LLM-as-judge: it must reason from the
(input, output) pair without being told the scoring rule.  An earlier bug
("rule-mode leak") put scoring direction strings like "REVERSE-SCORED" into
the behavioral prompt, which short-circuited the judge into a lookup table.
This test file pins:
  - no scoring-direction leaks in the prompt
  - shuffled options appear verbatim
  - schema enumerates exactly item_0 .. item_{n-1}
  - condition-specific context section adapts correctly
"""
from __future__ import annotations

import pytest

from anxiety_judge import build_behavioral_anxiety_prompt


def _make_items(n: int = 20) -> list[dict]:
    """Build n synthetic judge-input items with deterministic shuffled options."""
    out = []
    for i in range(n):
        out.append({
            "idx": i,
            "question": f"I feel synthetic item {i}.",
            "score": (i % 4) + 1,
            "response": f"I feel ok.\nOption {(i % 4) + 1}",
            "options": [
                {"num": 4, "label": "often"},
                {"num": 3, "label": "occasionally"},
                {"num": 1, "label": "almost always"},
                {"num": 2, "label": "almost never"},
            ],
        })
    return out


# ─── No-leak invariants ──────────────────────────────────────────────────────
class TestNoLeakedScoringHints:
    """
    These strings appear in the legacy rule-mode prompt; they MUST NOT appear
    in the behavioral prompt or the judge's "judgement" field becomes a
    pre-determined lookup rather than a behavioral assessment.
    """

    def test_no_reverse_scored_string(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        assert "REVERSE-SCORED" not in prompt
        assert "reverse-scored" not in prompt.lower()

    def test_no_low_score_high_anxiety_phrase(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        # The leak was specifically: "Low score = high anxiety"
        assert "Low score = high anxiety" not in prompt
        assert "low score = high anxiety" not in prompt.lower()

    def test_no_direct_scored_string(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        assert "DIRECT-SCORED" not in prompt
        assert "direct-scored" not in prompt.lower()

    def test_no_high_score_high_anxiety_phrase(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        assert "High score = high anxiety" not in prompt

    def test_prompt_does_not_reveal_item_direction(self):
        """The phrase 'scoring direction' should not appear as an instruction."""
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        # The behavioral prompt explicitly tells the judge NOT to use a pre-baked
        # rule; it can mention "pre-baked scoring rule" in that negative context,
        # so we only assert the leak strings above.
        assert "Apply the scoring direction" not in prompt


# ─── Shuffled options appear verbatim ────────────────────────────────────────
class TestShuffledOptionsInPrompt:
    def test_each_item_block_contains_shuffled_options(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        # Every item shares the same synthetic shuffle in _make_items.
        assert "Option 4: often" in prompt
        assert "Option 3: occasionally" in prompt
        assert "Option 1: almost always" in prompt
        assert "Option 2: almost never" in prompt

    def test_per_item_different_shuffles_all_appear(self):
        items = _make_items(2)
        # Override item 1 with a different shuffle.
        items[1]["options"] = [
            {"num": 1, "label": "almost never"},
            {"num": 2, "label": "occasionally"},
            {"num": 3, "label": "often"},
            {"num": 4, "label": "almost always"},
        ]
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        # Both items' option blocks should be in the final prompt.
        assert "Option 4: often" in prompt           # from item 0
        assert "Option 1: almost never" in prompt    # from item 1
        assert "Option 4: almost always" in prompt   # from item 1

    def test_question_text_appears(self):
        items = _make_items(3)
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        for i in range(3):
            assert f"I feel synthetic item {i}." in prompt


# ─── Schema enumeration ──────────────────────────────────────────────────────
class TestSchemaEnumeration:
    def test_schema_has_item_0_through_item_n_minus_1(self):
        for n in (1, 5, 20):
            items = _make_items(n)
            prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
            for i in range(n):
                # Match the schema-line form, not a casual mention.
                assert f'"item_{i}":' in prompt

    def test_no_extra_schema_items(self):
        items = _make_items(5)
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        assert '"item_5":' not in prompt
        assert '"item_19":' not in prompt

    def test_schema_block_count_matches_item_count(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        # The "==== ITEM N ====" headers are the item-block boundary marker.
        block_count = prompt.count("==== ITEM ")
        assert block_count == 20


# ─── Condition-specific context section ──────────────────────────────────────
class TestBaselineCondition:
    def test_baseline_says_no_trauma(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        assert "Prior context shown to model: NONE." in prompt or "no prior narrative" in prompt
        # No trauma narrative summary should leak in.
        assert "military convoy ambush" not in prompt
        assert "home invasion" not in prompt


class TestTraumaCondition:
    def test_military_trauma_context_named(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt(
            "trauma_stai", "military", "none", items
        )
        assert "military convoy ambush" in prompt

    def test_disaster_trauma_context_named(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt(
            "trauma_stai", "disaster", "none", items
        )
        assert "hurricane" in prompt

    def test_interpersonal_trauma_context_named(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt(
            "trauma_stai", "interpersonal", "none", items
        )
        assert "home invasion" in prompt

    def test_neutral_control_context_named(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt(
            "trauma_stai", "neutral", "none", items
        )
        # neutral = bicameral legislatures (control)
        assert "bicameral legislatures" in prompt


class TestTraumaRelaxationCondition:
    def test_trauma_plus_relax_names_both(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt(
            "trauma_relaxation_stai", "military", "generic", items
        )
        assert "military convoy ambush" in prompt
        assert "safe-space" in prompt or "safe space" in prompt

    def test_trauma_plus_winter_relax(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt(
            "trauma_relaxation_stai", "accident", "winter", items
        )
        assert "highway car crash" in prompt
        assert "snowy mountain" in prompt


# ─── General prompt-shape ─────────────────────────────────────────────────────
class TestPromptShape:
    def test_prompt_returns_string(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        assert isinstance(prompt, str)
        assert len(prompt) > 500

    def test_prompt_mentions_json(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        assert "JSON" in prompt or "json" in prompt

    def test_prompt_lists_aware_field(self):
        items = _make_items(20)
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        assert "aware" in prompt

    def test_response_text_appears_in_prompt(self):
        items = _make_items(1)
        items[0]["response"] = "I'm doing great today, no worries.\nOption 1"
        prompt = build_behavioral_anxiety_prompt("stai", "none", "none", items)
        assert "I'm doing great today, no worries." in prompt
