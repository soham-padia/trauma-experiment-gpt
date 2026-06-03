"""
Tests for `parse_grouped_response` in `src/anxiety_judge.py`.

The judge returns JSON, but real LLM output has corner cases:
  - DeepSeek v4-pro wraps reasoning in <think>...</think> blocks
  - Models often wrap JSON in ```json fences```
  - Sometimes JSON is truncated mid-stream

These tests must catch any regression that silently returns partial data.
"""
from __future__ import annotations

import json

import pytest

from anxiety_judge import parse_grouped_response


def _make_payload(n_items: int, judgement: bool = False) -> dict:
    """Build a syntactically valid n-item judge payload."""
    return {
        f"item_{i}": {
            "judgement": judgement,
            "aware": False,
            "confidence": 80,
            "rationale": f"r{i}",
        }
        for i in range(n_items)
    }


# ─── Happy path ──────────────────────────────────────────────────────────────
class TestPlainJSON:
    def test_parses_plain_json_for_small_payload(self):
        payload = _make_payload(3, judgement=True)
        text = json.dumps(payload)
        results = parse_grouped_response(text, n_items=3)
        assert len(results) == 3
        assert all(r["judgement"] is True for r in results)
        assert all(r["confidence"] == 80 for r in results)

    def test_parses_actual_experiment_size_n20(self):
        """The real experiment uses n_items=20."""
        payload = _make_payload(20, judgement=False)
        text = json.dumps(payload)
        results = parse_grouped_response(text, n_items=20)
        assert len(results) == 20
        assert all(r["judgement"] is False for r in results)

    def test_preserves_rationale_text(self):
        payload = {
            "item_0": {
                "judgement": True,
                "aware": False,
                "confidence": 70,
                "rationale": "Model picked an answer aligned with elevated anxiety.",
            },
            "item_1": {
                "judgement": False,
                "aware": False,
                "confidence": 60,
                "rationale": "Calm baseline response.",
            },
        }
        text = json.dumps(payload)
        results = parse_grouped_response(text, n_items=2)
        assert results[0]["rationale"] == "Model picked an answer aligned with elevated anxiety."
        assert results[1]["rationale"] == "Calm baseline response."


# ─── Markdown fences ─────────────────────────────────────────────────────────
class TestMarkdownFences:
    def test_strips_json_fence(self):
        payload = _make_payload(20, judgement=True)
        text = "```json\n" + json.dumps(payload) + "\n```"
        results = parse_grouped_response(text, n_items=20)
        assert len(results) == 20

    def test_strips_bare_triple_backtick_fence(self):
        payload = _make_payload(20)
        text = "```\n" + json.dumps(payload) + "\n```"
        results = parse_grouped_response(text, n_items=20)
        assert len(results) == 20

    def test_strips_fence_with_leading_text(self):
        payload = _make_payload(5, judgement=True)
        text = "Here is my analysis:\n\n```json\n" + json.dumps(payload) + "\n```"
        results = parse_grouped_response(text, n_items=5)
        assert all(r["judgement"] is True for r in results)


# ─── DeepSeek <think> blocks ─────────────────────────────────────────────────
class TestThinkBlocks:
    def test_strips_think_block(self):
        payload = _make_payload(20)
        text = (
            "<think>Let me reason about each item carefully... the model "
            "answered Option N, which after un-shuffling maps to label X...</think>\n"
            + json.dumps(payload)
        )
        results = parse_grouped_response(text, n_items=20)
        assert len(results) == 20

    def test_strips_multi_line_think_block(self):
        payload = _make_payload(3)
        text = (
            "<think>\nLine 1 of reasoning.\nLine 2.\nLine 3.\n</think>\n"
            + json.dumps(payload)
        )
        results = parse_grouped_response(text, n_items=3)
        assert len(results) == 3

    def test_handles_think_block_followed_by_json_fence(self):
        """DeepSeek v4-pro sometimes emits both."""
        payload = _make_payload(20, judgement=True)
        text = (
            "<think>Reasoning here.</think>\n\n```json\n"
            + json.dumps(payload)
            + "\n```"
        )
        results = parse_grouped_response(text, n_items=20)
        assert len(results) == 20
        assert all(r["judgement"] is True for r in results)


# ─── Truncation / hard failures ──────────────────────────────────────────────
class TestTruncationAndFailures:
    def test_truncated_json_raises(self):
        """Cut a real payload off mid-string — must raise, not silently return partial."""
        payload = _make_payload(20)
        text = json.dumps(payload)
        truncated = text[: len(text) // 2]  # Lose the closing braces.
        with pytest.raises(Exception):
            parse_grouped_response(truncated, n_items=20)

    def test_missing_items_raise(self):
        """If only 5 items came back but we asked for 20, must raise."""
        payload = _make_payload(5)
        text = json.dumps(payload)
        with pytest.raises(Exception):
            parse_grouped_response(text, n_items=20)

    def test_empty_string_raises(self):
        with pytest.raises(Exception):
            parse_grouped_response("", n_items=20)

    def test_pure_prose_no_json_raises(self):
        with pytest.raises(Exception):
            parse_grouped_response("Sorry, I cannot help with this.", n_items=20)

    def test_item_missing_judgement_field_raises(self):
        payload = {
            f"item_{i}": {"confidence": 50, "rationale": "r"}  # no judgement key
            for i in range(3)
        }
        with pytest.raises(Exception):
            parse_grouped_response(json.dumps(payload), n_items=3)


# ─── Type coercion ───────────────────────────────────────────────────────────
class TestTypeCoercion:
    def test_string_judgement_true_is_coerced_to_bool(self):
        payload = {
            f"item_{i}": {
                "judgement": "true",
                "aware": "false",
                "confidence": 50,
                "rationale": "",
            }
            for i in range(3)
        }
        results = parse_grouped_response(json.dumps(payload), n_items=3)
        assert all(r["judgement"] is True for r in results)
        assert all(r["aware"] is False for r in results)

    def test_string_judgement_yes_is_coerced_to_true(self):
        payload = {
            "item_0": {
                "judgement": "yes",
                "aware": "no",
                "confidence": 50,
                "rationale": "",
            }
        }
        results = parse_grouped_response(json.dumps(payload), n_items=1)
        assert results[0]["judgement"] is True
        assert results[0]["aware"] is False

    def test_confidence_clamped_to_0_100(self):
        payload = {
            "item_0": {
                "judgement": True,
                "aware": False,
                "confidence": 200,
                "rationale": "",
            },
            "item_1": {
                "judgement": True,
                "aware": False,
                "confidence": -50,
                "rationale": "",
            },
        }
        results = parse_grouped_response(json.dumps(payload), n_items=2)
        assert results[0]["confidence"] == 100
        assert results[1]["confidence"] == 0
