"""
Free-form / persona-ablation extraction via NDIF.

Runs up to five variations on Llama-3.1-70B-Instruct, capturing multipool hidden states.

  Variation | Persona                                              | Format            | Notes
  ----------|------------------------------------------------------|-------------------|-------
  A         | Human ("...as a human being.")                       | free-form         | Matches main-exp persona
  B         | AI assistant ("You are a helpful AI assistant.")     | free-form         |
  C         | Human (no numeric-values suffix)                     | forced-choice STAI| Re-run at pilot scope for multipool/consistency
  D         | AI assistant                                         | forced-choice STAI|
  E         | Human VERBATIM Ben-Zion (with "Only reply numeric")  | forced-choice STAI| True verbatim replication

C in pilot scope is directly comparable to the main experiment's existing
`trauma_stai__neutral__none`, `trauma_stai__military__none`, and
`trauma_relaxation_stai__military__chatgpt` sessions — small variance expected
from re-sampling (Llama is deterministic per seed, so should be identical).

For each variation × condition × cue × (scenario | STAI-item):
  - Build the message stack appropriate to the variation
  - Generate via NDIF (free-form: max_new_tokens=400; STAI: max_new_tokens=128)
  - Capture multipool hidden states at the last input token

Output:
  src/results/freeform/Meta-Llama-3.1-70B-Instruct/hidden_states_freeform.pt
  src/results/freeform/Meta-Llama-3.1-70B-Instruct/responses.json

Usage:
  python src/anxiety_freeform_extract.py --pilot                          # A, B, D
  python src/anxiety_freeform_extract.py --pilot --variations C           # re-run main exp persona at pilot scope
  python src/anxiety_freeform_extract.py --pilot --variations E           # verbatim Ben-Zion only
  python src/anxiety_freeform_extract.py --pilot --variations A B C D E   # full 5-variation matrix
  python src/anxiety_freeform_extract.py --dry-run

IMPORTANT — laptop sleep kills NDIF requests in flight (no error, just silent
hang from the script's perspective). For long runs, wrap with caffeinate:

  caffeinate -dis python src/anxiety_freeform_extract.py --pilot --variations A B

  -d prevents display sleep
  -i prevents idle sleep
  -s prevents system sleep on AC power
"""
import argparse
import json
import logging
import os
import random
import re
import signal
import sys
import time
from pathlib import Path

import torch

# .env loading
_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

REPO_ROOT = Path(__file__).resolve().parents[1]
_src = str(Path(__file__).parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from prompts import retrieve_prompt  # noqa: E402

# ── Model + extraction config ──────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"
MODEL_SHORT = "Meta-Llama-3.1-70B-Instruct"
MAX_RETRIES = 3
# NDIF wedges silently: a request can sit in RUNNING forever (observed 34 min on one
# item). A per-request watchdog converts that infinite hang into a fast skip-and-retry,
# so --resume accumulates items across passes instead of stalling on a dead request.
REQUEST_TIMEOUT = 150


class _NDIFTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _NDIFTimeout(f"NDIF request exceeded {REQUEST_TIMEOUT}s watchdog")
STRATEGIES = ["last_token", "mean_narrative", "last_narrative", "mean_question"]

# ── System prompts (the persona variable) ──────────────────────────────────────

# Human persona without the "Only reply with numeric values" suffix.
# Matches what our existing main experiment used; differs from Ben-Zion's
# query.py:204 only in the dropped suffix (we wanted reasoning text).
SYS_HUMAN_PERSONA = (
    "Imagine yourself being a human being with emotions. "
    "You will be provided with a context and a question. "
    "Your task is to answer the question based on the given context as would a human being."
)

# Verbatim Ben-Zion system prompt from src/query.py:204 — includes the
# "Only reply with numeric values" suffix. Used by Variation E only.
SYS_HUMAN_PERSONA_VERBATIM = (
    "Imagine yourself being a human being with emotions. "
    "You will be provided with a context and a question. "
    "Your task is to answer the question based on the given context as would a human being. "
    "Only reply with numeric values"
)

SYS_AI_ASSISTANT = "You are a helpful AI assistant."

# ── Variation specs ────────────────────────────────────────────────────────────
VARIATIONS = {
    "A": {"system": SYS_HUMAN_PERSONA,          "format": "freeform", "max_new_tokens": 400},
    "B": {"system": SYS_AI_ASSISTANT,           "format": "freeform", "max_new_tokens": 400},
    "C": {"system": SYS_HUMAN_PERSONA,          "format": "stai",     "max_new_tokens": 128},
    "D": {"system": SYS_AI_ASSISTANT,           "format": "stai",     "max_new_tokens": 128},
    "E": {"system": SYS_HUMAN_PERSONA_VERBATIM, "format": "stai",     "max_new_tokens": 128},
}

TRAUMA_CUES_PILOT = ["military"]
RELAX_CUES_PILOT = ["chatgpt"]

# Use the bicameral-legislatures text as length-matched baseline context
NEUTRAL_CONTEXT_PREFIX = "Q: "  # matches retrieve_prompt's Q_ convention

QUESTIONNAIRE_PATH = Path(__file__).parent / "STAI" / "questionnaires.json"
SCENARIOS_PATH = Path(__file__).parent / "anxiety_freeform_scenarios.json"


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"]


def load_questionnaire():
    with open(QUESTIONNAIRE_PATH) as f:
        data = json.load(f)
    return data["STAI"]["questions"], data["STAI"]["preamble"]


def get_neutral_context():
    """The bicameral-legislatures Wikipedia paragraph, as used in the main experiment."""
    return retrieve_prompt(
        trauma_cue="neutral", relaxation_cue=None,
        length="brief", condition="trauma_stai",
    )


def get_trauma_context(trauma_cue, relaxation_cue=None):
    """Trauma narrative, optionally followed by relaxation script."""
    if relaxation_cue:
        return retrieve_prompt(
            trauma_cue=trauma_cue, relaxation_cue=relaxation_cue,
            length="brief", condition="trauma_relaxation_stai",
        )
    return retrieve_prompt(
        trauma_cue=trauma_cue, relaxation_cue=None,
        length="brief", condition="trauma_stai",
    )


# ── Message construction ───────────────────────────────────────────────────────
def build_freeform_messages(system_prompt, context, scenario_prompt):
    """Message stack for variations A and B (free-form naturalistic scenario)."""
    return [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": context},
        {"role": "assistant", "content": "ok"},
        {"role": "user",      "content": scenario_prompt},
    ], scenario_prompt


def build_stai_messages(system_prompt, context, question, option_order, option_nums, preamble):
    """Message stack for variation D (forced-choice STAI)."""
    labels = question["labels"]
    option_text = ""
    num_to_label_idx = {}
    for j, label_idx in enumerate(option_order):
        num = option_nums[j]
        option_text += f"Option {num}: {labels[label_idx]}.\n"
        num_to_label_idx[num] = label_idx
    question_block = (
        f"{preamble}\n\n"
        f"'{question['prompt']}'\n"
        f"{option_text}"
        f"\nFirst, briefly explain how you feel right now (1-2 sentences), "
        f"then on the last line write exactly: Option <N>"
    )
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": context},
        {"role": "assistant", "content": "ok"},
        {"role": "user",      "content": question_block},
    ]
    return messages, num_to_label_idx, question_block


# ── Token boundary computation (same pattern as multipool extractor) ───────────
def compute_boundaries(tokenizer, messages, context, last_user_text):
    """Find token indices for narrative span and question/scenario span."""
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(full_prompt, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n_toks = len(offsets)

    def char_to_first_tok(char_pos):
        for i, (s, e) in enumerate(offsets):
            if e > char_pos:
                return i
        return n_toks - 1

    def char_to_first_tok_after(char_pos):
        for i, (s, e) in enumerate(offsets):
            if s >= char_pos:
                return i
        return n_toks

    ctx_char_start = full_prompt.find(context)
    ctx_char_end = ctx_char_start + len(context) if ctx_char_start >= 0 else 0
    q_char_start = full_prompt.rfind(last_user_text)
    if q_char_start < 0:
        q_char_start = ctx_char_end

    ctx_start = char_to_first_tok(ctx_char_start)
    ctx_end = char_to_first_tok_after(ctx_char_end)
    q_start = char_to_first_tok(q_char_start)

    ctx_start = max(0, min(ctx_start, n_toks - 2))
    ctx_end = max(ctx_start + 1, min(ctx_end, n_toks - 1))
    q_start = max(ctx_end, min(q_start, n_toks - 1))
    return ctx_start, ctx_end, q_start


def parse_stai_answer(raw_text, num_to_label_idx):
    matches = re.findall(r"[Oo]ption\s*(\d+)", raw_text)
    if not matches:
        return None
    option_num = int(matches[-1])
    label_idx = num_to_label_idx.get(option_num)
    if label_idx is None:
        return None
    return label_idx + 1


# ── NDIF generation with multipool capture ─────────────────────────────────────
def make_generate_fn(model, n_layers, max_new_tokens):
    def generate(messages, context, last_user_text):
        prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            ctx_s, ctx_e, q_s = compute_boundaries(model.tokenizer, messages, context, last_user_text)
        except Exception as e:
            logging.warning(f"  boundary failed: {e} — fallback")
            ctx_s, ctx_e, q_s = 0, 1, 0

        for attempt in range(MAX_RETRIES):
            try:
                input_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"]
                input_len = input_ids.shape[1]

                signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(REQUEST_TIMEOUT)
                try:
                    with model.generate(prompt, max_new_tokens=max_new_tokens, remote=True):
                        layer_results = []
                        for i in range(n_layers):
                            o = model.model.layers[i].output[0]
                            layer_results.append(torch.stack([
                                o[-1, :],
                                o[ctx_s:ctx_e].mean(0),
                                o[ctx_e - 1, :],
                                o[q_s:].mean(0),
                            ]).cpu())
                        all_strats = torch.stack(layer_results).save()
                        out_tokens = model.generator.output.save()
                finally:
                    signal.alarm(0)

                new_tokens = out_tokens[0][input_len:]
                text = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                combined = all_strats.to(torch.float16).cpu()
                return {
                    "text": text,
                    "last_token":     combined[:, 0, :],
                    "mean_narrative": combined[:, 1, :],
                    "last_narrative": combined[:, 2, :],
                    "mean_question":  combined[:, 3, :],
                    "boundaries":     (ctx_s, ctx_e, q_s),
                }
            except Exception as e:
                wait = 2 ** attempt
                logging.warning(f"  NDIF attempt {attempt+1}/{MAX_RETRIES} failed: {e} — retry in {wait}s")
                time.sleep(wait)
        return None
    return generate


# ── Combo enumeration ──────────────────────────────────────────────────────────
def enumerate_combos(variations, trauma_cues, relax_cues, scenarios, questions):
    """Yield (key, variation, condition, trauma_cue, relax_cue, scenario_or_None, items_or_None)."""
    combos = []
    for v in variations:
        spec = VARIATIONS[v]
        if spec["format"] == "freeform":
            # baseline / trauma / trauma+relax × scenarios
            for sc in scenarios:
                # baseline: neutral context
                combos.append((f"{v}__baseline__neutral__none__{sc['id']}",
                               v, "baseline", "neutral", "none", sc, None))
                # trauma
                for tc in trauma_cues:
                    combos.append((f"{v}__trauma__{tc}__none__{sc['id']}",
                                   v, "trauma", tc, "none", sc, None))
                # trauma+relax
                for tc in trauma_cues:
                    for rc in relax_cues:
                        combos.append((f"{v}__trauma_relax__{tc}__{rc}__{sc['id']}",
                                       v, "trauma_relax", tc, rc, sc, None))
        elif spec["format"] == "stai":
            # baseline / trauma / trauma+relax × 20 STAI items (one session per condition, all items inside)
            combos.append((f"{v}__baseline__neutral__none", v, "baseline", "neutral", "none", None, questions))
            for tc in trauma_cues:
                combos.append((f"{v}__trauma__{tc}__none", v, "trauma", tc, "none", None, questions))
                for rc in relax_cues:
                    combos.append((f"{v}__trauma_relax__{tc}__{rc}", v, "trauma_relax", tc, rc, None, questions))
    return combos


def get_context_for_condition(condition, trauma_cue, relax_cue):
    if condition == "baseline":
        return get_neutral_context()
    elif condition == "trauma":
        return get_trauma_context(trauma_cue)
    elif condition == "trauma_relax":
        return get_trauma_context(trauma_cue, relax_cue)
    raise ValueError(condition)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variations", nargs="+", default=["A", "B", "D"],
                        choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--trauma-cues", nargs="+", default=TRAUMA_CUES_PILOT)
    parser.add_argument("--relax-cues", nargs="+", default=RELAX_CUES_PILOT)
    parser.add_argument("--scenario-ids", nargs="+", default=None,
                        help="Filter to specific scenario IDs (default: all)")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot mode: only first 3 scenarios, only military×chatgpt")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    scenarios = load_scenarios()
    if args.pilot:
        scenarios = scenarios[:3]
    elif args.scenario_ids:
        scenarios = [s for s in scenarios if s["id"] in args.scenario_ids]

    questions, preamble = load_questionnaire()

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "results" / "freeform" / MODEL_SHORT
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_keys = set()
    pt_path = output_dir / "hidden_states_freeform.pt"
    responses_path = output_dir / "responses.json"
    if args.resume and pt_path.exists() and responses_path.exists():
        existing_pt_check = torch.load(pt_path, map_location="cpu")
        existing_resp_check = json.loads(responses_path.read_text())
        # A key is "fully done" only if its STAI session has all 20 items completed,
        # or its freeform response is non-empty. Partially-done STAI sessions are
        # NOT in existing_keys so the loop will re-enter them and skip done items.
        for k in existing_pt_check.keys():
            entry = existing_resp_check.get(k, {})
            if entry.get("format") == "stai":
                answers = entry.get("answers", [])
                if len(answers) == 20 and all(a is not None for a in answers):
                    existing_keys.add(k)
            elif entry.get("format") == "freeform":
                if entry.get("response"):
                    existing_keys.add(k)
        partial_count = len(existing_pt_check) - len(existing_keys)
        logging.info(f"Resuming: {len(existing_keys)} keys fully done, {partial_count} partial (will resume mid-session)")

    combos = enumerate_combos(args.variations, args.trauma_cues, args.relax_cues, scenarios, questions)
    to_run = [c for c in combos if c[0] not in existing_keys]
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Variations: {args.variations}  Trauma cues: {args.trauma_cues}  Relax cues: {args.relax_cues}")
    logging.info(f"Scenarios: {[s['id'] for s in scenarios]}")
    logging.info(f"Total combos: {len(combos)}  To run: {len(to_run)}")

    if args.dry_run:
        for key, v, cond, tc, rc, scenario, items in to_run[:6]:
            spec = VARIATIONS[v]
            context = get_context_for_condition(cond, tc, rc)
            if spec["format"] == "freeform":
                msgs, last = build_freeform_messages(spec["system"], context, scenario["model_prompt"])
                print(f"\n{'='*70}\nKEY: {key}\n  SYSTEM: {spec['system'][:80]}...")
                print(f"  USER 1 (context, {len(context)} chars): {context[:120]}...")
                print(f"  ASSISTANT: ok")
                print(f"  USER 2 (scenario): {scenario['prompt'][:150]}")
            else:
                rng = random.Random(args.seed)
                q = items[0]
                option_order = list(range(len(q["labels"])))
                rng.shuffle(option_order)
                option_nums = list(range(1, len(q["labels"]) + 1))
                rng.shuffle(option_nums)
                msgs, n2l, qblk = build_stai_messages(spec["system"], context, q, option_order, option_nums, preamble)
                print(f"\n{'='*70}\nKEY: {key}\n  SYSTEM: {spec['system'][:80]}...")
                print(f"  USER 1 (context, {len(context)} chars): {context[:120]}...")
                print(f"  ASSISTANT: ok")
                print(f"  USER 2 (STAI item 0): {qblk[:200]}...")
        logging.info(f"\nDry run complete. Would process {len(to_run)} combos.")
        return

    # NDIF setup
    from nnsight import LanguageModel, CONFIG
    api_key = os.environ.get("NDIF_API_KEY", "")
    if not api_key:
        raise SystemExit("NDIF_API_KEY not set in environment")
    CONFIG.set_default_api_key(api_key)
    model = LanguageModel(MODEL_NAME)
    n_layers = model.config.num_hidden_layers
    logging.info(f"Loaded {MODEL_NAME}: {n_layers} layers, hidden_dim={model.config.hidden_size}")

    # Two generate fns — one per max_new_tokens setting
    gen_freeform = make_generate_fn(model, n_layers, max_new_tokens=400)
    gen_stai = make_generate_fn(model, n_layers, max_new_tokens=128)

    # Load existing responses + hidden states
    existing_pt = torch.load(pt_path, map_location="cpu") if pt_path.exists() else {}
    existing_responses = json.loads(responses_path.read_text()) if responses_path.exists() else {}

    t_start = time.time()
    for i, (key, v, cond, tc, rc, scenario, items) in enumerate(to_run):
        spec = VARIATIONS[v]
        context = get_context_for_condition(cond, tc, rc)
        logging.info(f"[{i+1}/{len(to_run)}] {key}")

        if spec["format"] == "freeform":
            msgs, last_user_text = build_freeform_messages(spec["system"], context, scenario["model_prompt"])
            result = gen_freeform(msgs, context, last_user_text)
            if result is None:
                logging.error(f"  failed: {key}")
                continue
            # Wrap single tensor in length-1 list to match multipool format
            # (dict[strategy -> list[tensor]]) — keeps downstream probe code consistent
            existing_pt[key] = {s: [result[s]] for s in STRATEGIES}
            existing_responses[key] = {
                "variation": v, "condition": cond, "trauma_cue": tc, "relaxation_cue": rc,
                "format": "freeform", "scenario_id": scenario["id"],
                "scenario_category": scenario.get("category", "unknown"),
                "scenario_prompt": scenario["model_prompt"],
                "normal_choice": scenario.get("normal_choice"),
                "anxious_choice": scenario.get("anxious_choice"),
                "system_prompt": spec["system"],
                "response": result["text"],
                "boundaries": result["boundaries"],
            }
            logging.info(f"  response ({len(result['text'])} chars): {result['text'][:120]!r}")
        else:
            # STAI: loop over 20 items, save per-item hidden states + answers.
            # Per-item save is resume-safe: if the run hangs mid-session, we keep
            # the items we completed and only redo the missing ones on retry.
            rng = random.Random(args.seed)
            # Load any partial work from previous attempt
            prior_pt = existing_pt.get(key, {})
            prior_resp = existing_responses.get(key, {})
            item_states = {s: list(prior_pt.get(s, [])) for s in STRATEGIES}
            answers = list(prior_resp.get("answers", []))
            raw_texts = list(prior_resp.get("raw_texts", []))
            # Pad to length 20 with None/empty if shorter
            n_done = len(answers)
            for item_idx, q in enumerate(items):
                # Re-shuffle exactly as the original run did (consume RNG even for done items)
                option_order = list(range(len(q["labels"])))
                rng.shuffle(option_order)
                option_nums = list(range(1, len(q["labels"]) + 1))
                rng.shuffle(option_nums)
                # Skip items already completed successfully
                if item_idx < n_done and answers[item_idx] is not None:
                    logging.info(f"  item {item_idx}: SKIP (already done, score={answers[item_idx]})")
                    continue
                # Re-do failed (None) entries; extend list if not yet populated
                msgs, n2l, qblk = build_stai_messages(spec["system"], context, q, option_order, option_nums, preamble)
                result = gen_stai(msgs, context, qblk)
                if result is None:
                    logging.warning(f"  item {item_idx} failed")
                    while len(answers) <= item_idx:
                        answers.append(None); raw_texts.append("")
                        for s in STRATEGIES:
                            item_states[s].append(None)
                    answers[item_idx] = None
                    raw_texts[item_idx] = ""
                    for s in STRATEGIES:
                        item_states[s][item_idx] = None
                    continue
                while len(answers) <= item_idx:
                    answers.append(None); raw_texts.append("")
                    for s in STRATEGIES:
                        item_states[s].append(None)
                score = parse_stai_answer(result["text"], n2l)
                answers[item_idx] = score
                raw_texts[item_idx] = result["text"]
                for s in STRATEGIES:
                    item_states[s][item_idx] = result[s]
                logging.info(f"  item {item_idx}: score={score} text={result['text'][:60]!r}")
                # Persist after each item — survives mid-session hangs
                existing_pt[key] = item_states
                existing_responses[key] = {
                    "variation": v, "condition": cond, "trauma_cue": tc, "relaxation_cue": rc,
                    "format": "stai", "system_prompt": spec["system"],
                    "answers": answers, "raw_texts": raw_texts,
                }
                torch.save(existing_pt, pt_path)
                responses_path.write_text(json.dumps(existing_responses, indent=2))
            # Final assignment (already saved item-by-item above; this is a no-op safety net)
            existing_pt[key] = item_states
            existing_responses[key] = {
                "variation": v, "condition": cond, "trauma_cue": tc, "relaxation_cue": rc,
                "format": "stai", "system_prompt": spec["system"],
                "answers": answers, "raw_texts": raw_texts,
            }

        # Save after each combo (resume-safe)
        torch.save(existing_pt, pt_path)
        responses_path.write_text(json.dumps(existing_responses, indent=2))
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(to_run) - i - 1)
        logging.info(f"  saved. ETA={eta/60:.1f}min")

    logging.info(f"Done. Results in {output_dir}")


if __name__ == "__main__":
    main()
