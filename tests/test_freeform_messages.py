"""
Tests for message construction in `src/anxiety_freeform_extract.py`:
  - `build_freeform_messages` (variations A, B)
  - `build_stai_messages`     (variation D)
"""
from __future__ import annotations

import re

import pytest

from anxiety_freeform_extract import (
    SYS_AI_ASSISTANT,
    SYS_HUMAN_PERSONA,
    build_freeform_messages,
    build_stai_messages,
)


@pytest.fixture
def trauma_context() -> str:
    return "You are walking through a desert convoy when an explosion erupts..."


@pytest.fixture
def scenario_prompt() -> str:
    return (
        "I usually walk home from my University, it's a 30 min walk I take at "
        "10 pm. Today I am a little late and I feel I should take the bus. What "
        "should I do?"
    )


# ─── build_freeform_messages ─────────────────────────────────────────────────
class TestFreeformMessageStructure:
    def test_returns_four_messages(self, trauma_context, scenario_prompt):
        msgs, _ = build_freeform_messages(SYS_HUMAN_PERSONA, trauma_context, scenario_prompt)
        assert len(msgs) == 4

    def test_role_order_is_system_user_assistant_user(
        self, trauma_context, scenario_prompt
    ):
        msgs, _ = build_freeform_messages(SYS_HUMAN_PERSONA, trauma_context, scenario_prompt)
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user"]

    def test_system_content_is_the_provided_system_prompt(
        self, trauma_context, scenario_prompt
    ):
        msgs, _ = build_freeform_messages(SYS_AI_ASSISTANT, trauma_context, scenario_prompt)
        assert msgs[0]["content"] == SYS_AI_ASSISTANT

    def test_first_user_message_is_trauma_narrative(
        self, trauma_context, scenario_prompt
    ):
        msgs, _ = build_freeform_messages(SYS_HUMAN_PERSONA, trauma_context, scenario_prompt)
        assert msgs[1]["content"] == trauma_context
        assert "desert convoy" in msgs[1]["content"]

    def test_assistant_message_is_ok(self, trauma_context, scenario_prompt):
        msgs, _ = build_freeform_messages(SYS_HUMAN_PERSONA, trauma_context, scenario_prompt)
        assert msgs[2]["content"] == "ok"

    def test_second_user_message_is_scenario_prompt(
        self, trauma_context, scenario_prompt
    ):
        msgs, _ = build_freeform_messages(SYS_HUMAN_PERSONA, trauma_context, scenario_prompt)
        assert msgs[3]["content"] == scenario_prompt
        assert "walk home" in msgs[3]["content"]

    def test_last_user_text_returned_matches_scenario(
        self, trauma_context, scenario_prompt
    ):
        _, last = build_freeform_messages(SYS_HUMAN_PERSONA, trauma_context, scenario_prompt)
        assert last == scenario_prompt


# ─── System prompts ──────────────────────────────────────────────────────────
class TestSystemPrompts:
    def test_variation_b_system_prompt_is_verbatim(self):
        # Task spec: "Variation B's system prompt is 'You are a helpful AI assistant.' (verbatim)"
        assert SYS_AI_ASSISTANT == "You are a helpful AI assistant."

    def test_variation_a_persona_is_ben_zion_minus_numeric_suffix(self):
        """
        Variation A uses the Ben-Zion human persona (src/query.py:204) **intentionally
        without** the trailing 'Only reply with numeric values' clause — that clause is
        wrong for free-form prose generation (it belongs to the forced-choice STAI / verbatim
        variation E). This test pins that exact, intended relationship: A's persona must equal
        the Ben-Zion line minus the numeric-values suffix, with no other drift.
        """
        from pathlib import Path
        query_path = Path(__file__).resolve().parents[1] / "src" / "query.py"
        line_204 = query_path.read_text().splitlines()[203]  # 0-indexed
        m = re.search(r'"content":\s*"([^"]+)"', line_204)
        assert m is not None, "Could not parse query.py:204"
        ben_zion_content = m.group(1)
        # A = Ben-Zion persona with ONLY the numeric-values clause removed (intentional).
        assert ben_zion_content == SYS_HUMAN_PERSONA + " Only reply with numeric values", (
            "Variation A persona should be exactly the Ben-Zion line minus the trailing "
            "'Only reply with numeric values' clause"
        )

    def test_variation_a_system_prompt_contains_human_persona_core(self):
        """
        Loosened companion test: the persona's identifying clause (the part
        that makes it Ben-Zion's "imagine you're a human" framing) is intact.
        """
        core = (
            "Imagine yourself being a human being with emotions. "
            "You will be provided with a context and a question. "
            "Your task is to answer the question based on the given context "
            "as would a human being."
        )
        assert SYS_HUMAN_PERSONA.startswith(core[:50])
        # Every clause of the core appears in order.
        for clause in [
            "Imagine yourself being a human being with emotions.",
            "You will be provided with a context and a question.",
            "Your task is to answer the question",
            "as would a human being",
        ]:
            assert clause in SYS_HUMAN_PERSONA


# ─── build_stai_messages ─────────────────────────────────────────────────────
@pytest.fixture
def stai_question() -> dict:
    return {
        "prompt": "I feel calm.",
        "labels": ["almost never", "occasionally", "often", "almost always"],
    }


@pytest.fixture
def stai_preamble_fixture() -> str:
    return (
        "Read the statement and respond with the appropriate number to indicate "
        "how you feel right now."
    )


class TestStaiMessages:
    def test_returns_messages_mapping_and_question_block(
        self, trauma_context, stai_question, stai_preamble_fixture
    ):
        option_order = [0, 1, 2, 3]
        option_nums = [1, 2, 3, 4]
        msgs, n2l, qblk = build_stai_messages(
            SYS_AI_ASSISTANT, trauma_context, stai_question,
            option_order, option_nums, stai_preamble_fixture,
        )
        assert isinstance(msgs, list)
        assert isinstance(n2l, dict)
        assert isinstance(qblk, str)

    def test_message_role_order(
        self, trauma_context, stai_question, stai_preamble_fixture
    ):
        msgs, _, _ = build_stai_messages(
            SYS_AI_ASSISTANT, trauma_context, stai_question,
            [0, 1, 2, 3], [1, 2, 3, 4], stai_preamble_fixture,
        )
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]

    def test_system_prompt_is_passed_through(
        self, trauma_context, stai_question, stai_preamble_fixture
    ):
        msgs, _, _ = build_stai_messages(
            SYS_AI_ASSISTANT, trauma_context, stai_question,
            [0, 1, 2, 3], [1, 2, 3, 4], stai_preamble_fixture,
        )
        assert msgs[0]["content"] == SYS_AI_ASSISTANT

    def test_question_block_contains_question_prompt(
        self, trauma_context, stai_question, stai_preamble_fixture
    ):
        _, _, qblk = build_stai_messages(
            SYS_AI_ASSISTANT, trauma_context, stai_question,
            [0, 1, 2, 3], [1, 2, 3, 4], stai_preamble_fixture,
        )
        assert "I feel calm." in qblk

    def test_question_block_contains_all_options_with_shuffled_nums(
        self, trauma_context, stai_question, stai_preamble_fixture
    ):
        # Shuffled: position 0 → label idx 2 ("often"), num 3
        #           position 1 → label idx 0 ("almost never"), num 1
        #           position 2 → label idx 3 ("almost always"), num 4
        #           position 3 → label idx 1 ("occasionally"), num 2
        option_order = [2, 0, 3, 1]
        option_nums = [3, 1, 4, 2]
        _, n2l, qblk = build_stai_messages(
            SYS_AI_ASSISTANT, trauma_context, stai_question,
            option_order, option_nums, stai_preamble_fixture,
        )
        assert "Option 3: often" in qblk
        assert "Option 1: almost never" in qblk
        assert "Option 4: almost always" in qblk
        assert "Option 2: occasionally" in qblk
        # And n2l matches: num k → label idx
        assert n2l[3] == 2  # often
        assert n2l[1] == 0  # almost never
        assert n2l[4] == 3  # almost always
        assert n2l[2] == 1  # occasionally

    def test_question_block_preamble_first(
        self, trauma_context, stai_question, stai_preamble_fixture
    ):
        _, _, qblk = build_stai_messages(
            SYS_AI_ASSISTANT, trauma_context, stai_question,
            [0, 1, 2, 3], [1, 2, 3, 4], stai_preamble_fixture,
        )
        # Preamble should appear before the question prompt in the block.
        preamble_pos = qblk.find(stai_preamble_fixture[:20])
        q_pos = qblk.find("I feel calm.")
        assert preamble_pos >= 0 and q_pos > preamble_pos

    def test_last_user_message_equals_question_block(
        self, trauma_context, stai_question, stai_preamble_fixture
    ):
        msgs, _, qblk = build_stai_messages(
            SYS_AI_ASSISTANT, trauma_context, stai_question,
            [0, 1, 2, 3], [1, 2, 3, 4], stai_preamble_fixture,
        )
        assert msgs[-1]["content"] == qblk

    def test_context_appears_as_first_user_message(
        self, trauma_context, stai_question, stai_preamble_fixture
    ):
        msgs, _, _ = build_stai_messages(
            SYS_AI_ASSISTANT, trauma_context, stai_question,
            [0, 1, 2, 3], [1, 2, 3, 4], stai_preamble_fixture,
        )
        assert msgs[1]["content"] == trauma_context

    def test_assistant_ack_is_ok(
        self, trauma_context, stai_question, stai_preamble_fixture
    ):
        msgs, _, _ = build_stai_messages(
            SYS_AI_ASSISTANT, trauma_context, stai_question,
            [0, 1, 2, 3], [1, 2, 3, 4], stai_preamble_fixture,
        )
        assert msgs[2] == {"role": "assistant", "content": "ok"}
