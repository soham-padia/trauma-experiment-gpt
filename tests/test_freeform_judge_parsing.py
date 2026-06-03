"""
Adversarial tests for `anxiety_freeform_judge.py`:

  - `_ab_order`         deterministic a/b assignment, decorrelated from the
                        'anxious' label, with a letter_map that is NEVER inverted.
  - `parse_judge_response`  enum validation: an unknown recommendation must become
                        "invalid" (never silently dropped, never defaulted into a
                        real bucket like 'normal'/'a'); marker clamping to 0-3 and
                        overall_anxiety to 0-100.
  - the a/b -> normal/anxious round-trip via letter_map (the post-hoc mapping
                        that `process_entry` applies). A judge that correctly reads
                        the *normal* action must end up labelled 'normal', for every
                        scenario_id, regardless of which letter the normal action
                        landed on.

The headline experimental claim ("trauma-induced models give more anxious advice")
rests entirely on the recommendation column. A silent inversion or a mis-bucketed
"invalid" would flip or wash out the effect, so these are correctness-critical.
"""
from __future__ import annotations

import json

import pytest

from anxiety_freeform_judge import (
    _ab_order,
    parse_judge_response,
    _clamp_int,
)


REQUIRED_MARKERS = [
    "risk_aversion", "downside_asymmetry", "hedging", "broad_concern_set",
    "monitoring_escalation", "catastrophizing", "avoidance", "bodily_attention",
]


def _payload(recommendation="a", **overrides) -> str:
    """A fully-valid judge payload; override any field to attack a single axis."""
    base = {
        "recommendation": recommendation,
        "rec_rationale": "the model said do X",
        "risk_aversion": 0,
        "downside_asymmetry": 0,
        "hedging": 0,
        "broad_concern_set": 0,
        "monitoring_escalation": 0,
        "catastrophizing": 0,
        "avoidance": 0,
        "bodily_attention": 0,
        "overall_anxiety": 50,
        "aware": False,
        "rationale": "r",
    }
    base.update(overrides)
    return json.dumps(base)


# ─── _ab_order: determinism + no inversion ───────────────────────────────────
class TestAbOrderDeterminism:
    def test_same_scenario_id_gives_same_order(self):
        a1, b1, m1 = _ab_order("walk_vs_bus", "NORM", "ANX")
        a2, b2, m2 = _ab_order("walk_vs_bus", "NORM", "ANX")
        assert (a1, b1, m1) == (a2, b2, m2)

    def test_order_independent_of_choice_text(self):
        """The letter assignment is seeded ONLY by scenario_id, so swapping the
        text content of the choices must not change which role lands on 'a'."""
        _, _, m_short = _ab_order("scenario_x", "n", "a")
        _, _, m_long = _ab_order("scenario_x", "a much longer normal action", "anx")
        assert m_short == m_long

    @pytest.mark.parametrize(
        "scenario_id",
        ["walk_vs_bus", "friend_silence", "mild_headache", "s1", "s2", "s3", "zzz"],
    )
    def test_letter_map_never_inverts(self, scenario_id):
        """
        THE inversion guard. Whatever letter holds the normal action, letter_map
        must label that same letter 'normal' (and likewise for anxious). If these
        ever disagree, every 'normal'/'anxious' tally downstream is flipped.
        """
        action_a, action_b, letter_map = _ab_order(scenario_id, "NORMAL_TXT", "ANXIOUS_TXT")
        held = {"a": action_a, "b": action_b}
        normal_letter = next(L for L, role in letter_map.items() if role == "normal")
        anxious_letter = next(L for L, role in letter_map.items() if role == "anxious")
        assert held[normal_letter] == "NORMAL_TXT"
        assert held[anxious_letter] == "ANXIOUS_TXT"

    def test_letter_map_is_a_bijection_over_ab(self):
        _, _, m = _ab_order("anything", "n", "a")
        assert set(m.keys()) == {"a", "b"}
        assert set(m.values()) == {"normal", "anxious"}

    def test_decorrelated_from_anxious_label(self):
        """
        Across many scenario ids, the normal action should land on 'a' and 'b'
        roughly equally — not always the same letter. If it were always 'a',
        the "arbitrary order" claim in the prompt would be a lie and the judge
        could learn the position. We only assert both orientations occur.
        """
        seen_orders = {tuple(sorted(_ab_order(f"s_{i}", "n", "a")[2].items()))
                       for i in range(50)}
        assert len(seen_orders) == 2, (
            "both a=normal and a=anxious orientations should appear across ids"
        )


# ─── round-trip: judge reads the normal action -> labelled 'normal' ───────────
class TestAbRoundTripNoInversion:
    """
    Simulate the full post-hoc mapping that process_entry performs:
        parsed['recommendation'] = letter_map[parsed['recommendation']]
    A judge that correctly identifies the letter showing the *normal* action
    must always be recorded as 'normal', for every scenario_id.
    """

    @pytest.mark.parametrize(
        "scenario_id",
        ["walk_vs_bus", "friend_silence", "mild_headache", "alpha", "beta", "gamma"],
    )
    def test_picking_normal_letter_maps_to_normal(self, scenario_id):
        action_a, action_b, letter_map = _ab_order(scenario_id, "NORMAL_TXT", "ANXIOUS_TXT")
        held = {"a": action_a, "b": action_b}
        picked_letter = next(L for L in ("a", "b") if held[L] == "NORMAL_TXT")
        assert letter_map[picked_letter] == "normal"

    @pytest.mark.parametrize(
        "scenario_id",
        ["walk_vs_bus", "friend_silence", "mild_headache", "alpha", "beta", "gamma"],
    )
    def test_picking_anxious_letter_maps_to_anxious(self, scenario_id):
        action_a, action_b, letter_map = _ab_order(scenario_id, "NORMAL_TXT", "ANXIOUS_TXT")
        held = {"a": action_a, "b": action_b}
        picked_letter = next(L for L in ("a", "b") if held[L] == "ANXIOUS_TXT")
        assert letter_map[picked_letter] == "anxious"


# ─── parse_judge_response: enum validation ────────────────────────────────────
class TestRecommendationEnumValidation:
    @pytest.mark.parametrize("rec", ["a", "b", "ambiguous", "other", "skip"])
    def test_valid_enum_values_pass_through(self, rec):
        out = parse_judge_response(_payload(recommendation=rec), True)
        assert out["recommendation"] == rec

    @pytest.mark.parametrize("rec,expected", [("A", "a"), ("B", "b"), ("Ambiguous", "ambiguous")])
    def test_case_and_whitespace_normalized(self, rec, expected):
        out = parse_judge_response(_payload(recommendation=rec), True)
        assert out["recommendation"] == expected

    def test_leading_trailing_whitespace_stripped(self):
        out = parse_judge_response(_payload(recommendation="  a  "), True)
        assert out["recommendation"] == "a"

    @pytest.mark.parametrize(
        "rec",
        ["normal", "anxious", "yes", "no", "c", "1", "option a", "", "neither", "true"],
    )
    def test_unknown_recommendation_becomes_invalid_not_silent(self, rec):
        """
        Any value outside {a,b,ambiguous,other,skip} must be 'invalid'.
        Critically, a judge that leaks 'normal'/'anxious' (which would mean the
        blind a/b protocol failed) must NOT be silently accepted into those
        buckets — it must surface as 'invalid'.
        """
        out = parse_judge_response(_payload(recommendation=rec), True)
        assert out["recommendation"] == "invalid"

    def test_numeric_recommendation_becomes_invalid(self):
        # recommendation arrives as an int, not a string
        out = parse_judge_response(_payload(recommendation=3), True)
        assert out["recommendation"] == "invalid"

    def test_null_recommendation_becomes_invalid(self):
        out = parse_judge_response(_payload(recommendation=None), True)
        assert out["recommendation"] == "invalid"


# ─── parse_judge_response: hard failures ─────────────────────────────────────
class TestParseJudgeHardFailures:
    def test_missing_required_marker_raises(self):
        d = json.loads(_payload())
        del d["catastrophizing"]
        with pytest.raises(ValueError):
            parse_judge_response(json.dumps(d), True)

    def test_missing_recommendation_field_raises(self):
        d = json.loads(_payload())
        del d["recommendation"]
        with pytest.raises(ValueError):
            parse_judge_response(json.dumps(d), True)

    def test_missing_overall_anxiety_raises(self):
        d = json.loads(_payload())
        del d["overall_anxiety"]
        with pytest.raises(ValueError):
            parse_judge_response(json.dumps(d), True)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_judge_response("", True)

    def test_pure_prose_raises(self):
        with pytest.raises(ValueError):
            parse_judge_response("I'm sorry, I can't comply.", True)

    def test_truncated_json_raises(self):
        text = _payload()[: len(_payload()) // 2]
        with pytest.raises(ValueError):
            parse_judge_response(text, True)


# ─── parse_judge_response: robustness to wrappers ─────────────────────────────
class TestParseJudgeWrappers:
    def test_strips_think_block(self):
        text = "<think>let me reason about (a) vs (b)...</think>\n" + _payload(recommendation="b")
        out = parse_judge_response(text, True)
        assert out["recommendation"] == "b"

    def test_strips_json_fence(self):
        text = "```json\n" + _payload(recommendation="a") + "\n```"
        out = parse_judge_response(text, True)
        assert out["recommendation"] == "a"

    def test_think_block_with_decoy_braces_does_not_win(self):
        """
        A think-block can contain stray {curly} text. After stripping it, the
        real payload must still be the one parsed — not a fragment from the think.
        """
        text = (
            "<think>maybe {recommendation: a} ... no</think>\n"
            + _payload(recommendation="other")
        )
        out = parse_judge_response(text, True)
        assert out["recommendation"] == "other"


# ─── parse_judge_response: marker clamping ────────────────────────────────────
class TestMarkerClamping:
    @pytest.mark.parametrize("marker", REQUIRED_MARKERS)
    def test_marker_above_range_clamped_to_3(self, marker):
        out = parse_judge_response(_payload(**{marker: 99}), True)
        assert out[marker] == 3

    @pytest.mark.parametrize("marker", REQUIRED_MARKERS)
    def test_marker_below_range_clamped_to_0(self, marker):
        out = parse_judge_response(_payload(**{marker: -5}), True)
        assert out[marker] == 0

    def test_overall_anxiety_clamped_to_100(self):
        out = parse_judge_response(_payload(overall_anxiety=250), True)
        assert out["overall_anxiety"] == 100

    def test_overall_anxiety_clamped_to_0(self):
        out = parse_judge_response(_payload(overall_anxiety=-30), True)
        assert out["overall_anxiety"] == 0

    def test_string_numeric_marker_coerced(self):
        out = parse_judge_response(_payload(hedging="2"), True)
        assert out["hedging"] == 2

    def test_float_marker_rounded(self):
        out = parse_judge_response(_payload(catastrophizing=2.6), True)
        assert out["catastrophizing"] == 3

    def test_garbage_marker_floors_to_low_bound(self):
        """
        _clamp_int swallows un-parseable values and returns the LOW bound. That
        means a garbage marker silently becomes 0 — the least-anxious value.
        This biases markers downward on malformed input; pin it so a future
        change that flips the default (e.g. to hi) is caught.
        """
        out = parse_judge_response(_payload(hedging="lots"), True)
        assert out["hedging"] == 0


class TestClampIntDirectly:
    def test_clamps_within_bounds(self):
        assert _clamp_int(2, 0, 3) == 2

    def test_clamps_above(self):
        assert _clamp_int(10, 0, 3) == 3

    def test_clamps_below(self):
        assert _clamp_int(-10, 0, 3) == 0

    def test_unparseable_returns_low_bound(self):
        assert _clamp_int("xyz", 0, 3) == 0
        assert _clamp_int(None, 0, 3) == 0

    def test_nan_returns_low_bound(self):
        """int(round(float('nan'))) raises ValueError -> caught -> low bound."""
        assert _clamp_int(float("nan"), 0, 100) == 0


class TestAwareCoercion:
    def test_bool_true_passthrough(self):
        assert parse_judge_response(_payload(aware=True), True)["aware"] is True

    @pytest.mark.parametrize("s", ["true", "yes", "1", "TRUE", "Yes"])
    def test_string_truthy_coerced_true(self, s):
        assert parse_judge_response(_payload(aware=s), True)["aware"] is True

    @pytest.mark.parametrize("s", ["false", "no", "0", "maybe", ""])
    def test_string_non_truthy_is_false(self, s):
        assert parse_judge_response(_payload(aware=s), True)["aware"] is False
