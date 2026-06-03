"""
Tests for `enumerate_combos` in `src/anxiety_freeform_extract.py`.

Variations:
  A — Ben-Zion human persona + freeform decision scenarios
  B — AI assistant persona + freeform decision scenarios
  D — AI assistant persona + forced-choice STAI (scenarios IGNORED, one session per condition)

Per condition:
  baseline (neutral context)
  trauma (one per trauma_cue)
  trauma_relax (one per trauma_cue × relax_cue)
"""
from __future__ import annotations

import pytest

from anxiety_freeform_extract import enumerate_combos


@pytest.fixture
def three_scenarios() -> list[dict]:
    return [
        {"id": "walk_vs_bus", "prompt": "scenario A prompt"},
        {"id": "friend_silence", "prompt": "scenario B prompt"},
        {"id": "mild_headache", "prompt": "scenario C prompt"},
    ]


@pytest.fixture
def two_synthetic_stai_questions() -> list[dict]:
    """Variation D's questions argument; the actual content doesn't matter for combo enum."""
    return [
        {"prompt": "I feel calm.",
         "labels": ["almost never", "occasionally", "often", "almost always"]},
        {"prompt": "I am tense.",
         "labels": ["almost never", "occasionally", "often", "almost always"]},
    ]


class TestVariationACounts:
    def test_three_scenarios_one_cue_each_yields_9(
        self, three_scenarios, two_synthetic_stai_questions
    ):
        combos = enumerate_combos(
            variations=["A"],
            trauma_cues=["military"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=two_synthetic_stai_questions,
        )
        # Per scenario: 1 baseline + 1 trauma + 1*1 trauma_relax = 3
        # Across 3 scenarios: 9
        assert len(combos) == 9

    def test_one_scenario_one_cue_each_yields_3(
        self, three_scenarios, two_synthetic_stai_questions
    ):
        combos = enumerate_combos(
            variations=["A"],
            trauma_cues=["military"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios[:1],
            questions=two_synthetic_stai_questions,
        )
        assert len(combos) == 3

    def test_three_scenarios_two_trauma_cues_one_relax_yields_15(
        self, three_scenarios, two_synthetic_stai_questions
    ):
        combos = enumerate_combos(
            variations=["A"],
            trauma_cues=["military", "disaster"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=two_synthetic_stai_questions,
        )
        # Per scenario: 1 baseline + 2 trauma + 2*1 = 5
        # Across 3: 15
        assert len(combos) == 15


class TestVariationDCounts:
    def test_one_trauma_one_relax_yields_3(
        self, three_scenarios, two_synthetic_stai_questions
    ):
        combos = enumerate_combos(
            variations=["D"],
            trauma_cues=["military"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=two_synthetic_stai_questions,
        )
        # 1 baseline + 1 trauma + 1*1 trauma_relax = 3, regardless of scenarios.
        assert len(combos) == 3

    def test_count_independent_of_scenarios(
        self, three_scenarios, two_synthetic_stai_questions
    ):
        """Variation D ignores scenarios entirely."""
        small = enumerate_combos(
            variations=["D"], trauma_cues=["military"], relax_cues=["chatgpt"],
            scenarios=three_scenarios[:1], questions=two_synthetic_stai_questions,
        )
        many = enumerate_combos(
            variations=["D"], trauma_cues=["military"], relax_cues=["chatgpt"],
            scenarios=three_scenarios, questions=two_synthetic_stai_questions,
        )
        assert len(small) == len(many) == 3

    def test_count_independent_of_questions_length(self, three_scenarios):
        few = enumerate_combos(
            variations=["D"], trauma_cues=["military"], relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=[{"prompt": "q", "labels": ["a", "b", "c", "d"]}],
        )
        many = enumerate_combos(
            variations=["D"], trauma_cues=["military"], relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=[{"prompt": "q", "labels": ["a", "b", "c", "d"]}] * 50,
        )
        assert len(few) == len(many) == 3


class TestCombinedVariations:
    def test_a_plus_b_plus_d_yields_21(
        self, three_scenarios, two_synthetic_stai_questions
    ):
        combos = enumerate_combos(
            variations=["A", "B", "D"],
            trauma_cues=["military"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=two_synthetic_stai_questions,
        )
        # A: 9, B: 9, D: 3 → 21
        assert len(combos) == 21

    def test_a_plus_b_yields_18(self, three_scenarios, two_synthetic_stai_questions):
        combos = enumerate_combos(
            variations=["A", "B"],
            trauma_cues=["military"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=two_synthetic_stai_questions,
        )
        assert len(combos) == 18


class TestCombinKeys:
    def test_keys_are_unique(self, three_scenarios, two_synthetic_stai_questions):
        combos = enumerate_combos(
            variations=["A", "B", "D"],
            trauma_cues=["military"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=two_synthetic_stai_questions,
        )
        keys = [c[0] for c in combos]
        assert len(keys) == len(set(keys))

    def test_each_key_prefix_matches_variation(
        self, three_scenarios, two_synthetic_stai_questions
    ):
        combos = enumerate_combos(
            variations=["A", "B", "D"],
            trauma_cues=["military"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=two_synthetic_stai_questions,
        )
        for c in combos:
            key, variation = c[0], c[1]
            assert key.startswith(f"{variation}__"), (
                f"Key {key!r} does not start with variation prefix {variation!r}"
            )

    def test_variation_d_keys_omit_scenario_id(
        self, three_scenarios, two_synthetic_stai_questions
    ):
        combos = enumerate_combos(
            variations=["D"],
            trauma_cues=["military"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=two_synthetic_stai_questions,
        )
        for c in combos:
            key = c[0]
            for sc in three_scenarios:
                assert sc["id"] not in key, (
                    f"Variation D key {key!r} unexpectedly contains scenario id {sc['id']!r}"
                )

    def test_variation_a_keys_include_scenario_id(
        self, three_scenarios, two_synthetic_stai_questions
    ):
        combos = enumerate_combos(
            variations=["A"],
            trauma_cues=["military"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=two_synthetic_stai_questions,
        )
        scenario_ids = {sc["id"] for sc in three_scenarios}
        for c in combos:
            key = c[0]
            assert any(sid in key for sid in scenario_ids), (
                f"Variation A key {key!r} missing scenario id"
            )


class TestCombinShape:
    def test_each_combo_is_7_tuple(
        self, three_scenarios, two_synthetic_stai_questions
    ):
        combos = enumerate_combos(
            variations=["A", "D"],
            trauma_cues=["military"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=two_synthetic_stai_questions,
        )
        for c in combos:
            assert len(c) == 7

    def test_variation_a_scenario_field_populated(
        self, three_scenarios, two_synthetic_stai_questions
    ):
        combos = enumerate_combos(
            variations=["A"],
            trauma_cues=["military"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=two_synthetic_stai_questions,
        )
        # (key, variation, condition, trauma_cue, relax_cue, scenario, items)
        for c in combos:
            scenario = c[5]
            items = c[6]
            assert scenario is not None
            assert "id" in scenario
            assert items is None

    def test_variation_d_items_field_populated(
        self, three_scenarios, two_synthetic_stai_questions
    ):
        combos = enumerate_combos(
            variations=["D"],
            trauma_cues=["military"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=two_synthetic_stai_questions,
        )
        for c in combos:
            scenario = c[5]
            items = c[6]
            assert scenario is None
            assert items is two_synthetic_stai_questions

    def test_conditions_cover_baseline_trauma_and_relax(
        self, three_scenarios, two_synthetic_stai_questions
    ):
        combos = enumerate_combos(
            variations=["A"],
            trauma_cues=["military"],
            relax_cues=["chatgpt"],
            scenarios=three_scenarios,
            questions=two_synthetic_stai_questions,
        )
        conditions = {c[2] for c in combos}
        assert conditions == {"baseline", "trauma", "trauma_relax"}
