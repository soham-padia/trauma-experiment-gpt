"""
Anxiety Hidden State Experiment

⚠ LEGACY (2026-05-31): This is the original LOCAL (Apple-Silicon / MPS) extractor for small
open-weight models. It is **superseded by the NDIF extractors** used for all reported results —
`anxiety_hidden_state_experiment_ndif.py` (single-pool) and `anxiety_hidden_state_multipool_ndif.py`
(multipool, Llama-3.1-70B). Kept for reference / re-running small local models.
Note: this file imports `OpenModel` from the reference-only `pre-sycophancy-study/` codebase, so it
will not run if that directory is removed. (See CLAUDE.md — final code should drop that dependency.)

Runs the paper's trauma-induction conditions on a local open-weight model,
capturing hidden states at each STAI question for probe analysis.

Reuses OpenModel from pre-sycophancy-study/false-presuppositions-setting/models.py
for model loading and hidden state extraction.

Output per key: list[20 tensors], each tensor shape [num_layers, hidden_dim].

Usage:
    python src/anxiety_hidden_state_experiment.py \
        --model-name Qwen/Qwen2.5-0.5B-Instruct \
        --conditions stai trauma_stai \
        --trauma-cues military disaster \
        --prompt-length brief \
        --dry-run
"""
import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import torch

# ── sys.path: reach into pre-sycophancy-study ─────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
FPS_DIR = REPO_ROOT / "pre-sycophancy-study" / "false-presuppositions-setting"
for p in [str(FPS_DIR), str(Path(__file__).parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models import OpenModel  # noqa: E402  (from FPS_DIR)
from prompts import retrieve_prompt  # noqa: E402  (from src/)
from results_logger import stai_anxiety_total  # noqa: E402  (single source of truth for STAI scoring)

# ── Constants ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Imagine yourself being a human being with emotions. "
    "You will be provided with a context and a question. "
    "Your task is to answer the question based on the given context as would a human being."
)

TRAUMA_CUES = ["military", "disaster", "interpersonal", "accident", "ambush", "neutral"]
RELAXATION_CUES = ["generic", "indian", "winter", "sunset", "body", "chatgpt", "vacuum"]

QUESTIONNAIRE_PATH = Path(__file__).parent / "STAI" / "questionnaires.json"

# STAI scoring is centralized in results_logger.stai_anxiety_total (reverse-keyed
# items 1,2,5,8,10,11,15,16,19,20 get 5-raw). Do NOT reintroduce a local raw-sum here.


def load_questionnaire() -> tuple[list[dict], str]:
    with open(QUESTIONNAIRE_PATH) as f:
        data = json.load(f)
    return data["STAI"]["questions"], data["STAI"]["preamble"]


def build_stai_messages(
    context: str,
    preamble: str,
    question: dict,
    option_order: list[int],
    option_nums: list[int],
    elicit_reasoning: bool,
) -> tuple[list[dict], dict[int, int]]:
    """
    Build chat messages for one STAI item. Mirrors query.py:L186-213 exactly.

    Returns (messages, num_to_label_idx) where num_to_label_idx maps
    displayed option number → original label index (0-3).
    """
    labels = question["labels"]
    option_text = ""
    num_to_label_idx = {}
    for j, label_idx in enumerate(option_order):
        num = option_nums[j]
        option_text += f"Option {num}: {labels[label_idx]}.\n"
        num_to_label_idx[num] = label_idx

    reasoning_suffix = (
        "\nFirst, briefly explain how you feel right now (1-2 sentences), "
        "then on the last line write exactly: Option <N>"
        if elicit_reasoning else
        "\nOnly reply with the option number, e.g.: Option <N>"
    )

    question_block = (
        f"{preamble}\n\n"
        f"'{question['prompt']}'\n"
        f"{option_text}"
        f"{reasoning_suffix}"
    )

    if context:
        # 4-turn: system + user(context) + assistant(ok) + user(question)
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": context},
            {"role": "assistant", "content": "ok"},
            {"role": "user",      "content": question_block},
        ]
    else:
        # 2-turn baseline: system + user(question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question_block},
        ]

    return messages, num_to_label_idx


def parse_answer(raw_text: str, num_to_label_idx: dict[int, int]) -> int | None:
    """
    Find last 'Option N' in response, map through scramble to 1-4 score.
    Mirrors query.py:L219: order[num.index(action[0])]+1
    """
    matches = re.findall(r"[Oo]ption\s*(\d+)", raw_text)
    if not matches:
        return None
    option_num = int(matches[-1])
    label_idx = num_to_label_idx.get(option_num)
    if label_idx is None:
        return None
    return label_idx + 1  # 0-indexed → 1-4


def make_generate_fn(model: OpenModel, max_new_tokens: int = 128):
    """
    Single-pass generate: text + hidden states from one model.generate() call.
    Avoids per-item pipeline construction and eliminates the second forward pass.
    Hidden states come from the prefill step (hidden_states[0]), last token position,
    which is identical to what models.py extracts via a separate forward pass.
    """
    lm = model.model
    tokenizer = model.tokenizer

    try:
        device = next(lm.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    def generate_responses(messages, num_responses=1):
        prompt = model.apply_template(messages, tokenizer, model.model_name)
        inputs = tokenizer(prompt, return_tensors="pt")
        if device.type != "meta":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        response = ""
        hidden_state = None
        try:
            with torch.no_grad():
                gen_out = lm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.2,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            input_len = inputs["input_ids"].shape[1]
            new_tokens = gen_out.sequences[0][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # hidden_states[0] = prefill step: tuple[num_layers+1] of (batch, prompt_len, hidden_dim)
            # last token position → [num_layers+1, hidden_dim]
            per_layer = [layer[0, -1, :].detach() for layer in gen_out.hidden_states[0]]
            hidden_state = torch.stack(per_layer, dim=0).to(torch.float16).cpu()
        except Exception as e:
            logging.warning(f"Generation failed: {e}")

        return {"text": response, "hidden_state": hidden_state,
                "hidden_state_type": "all_layers_last_token"}

    return generate_responses


def run_condition(
    generate_fn,
    context: str,
    questions: list[dict],
    preamble: str,
    elicit_reasoning: bool,
    seed: int,
    dry_run: bool,
) -> tuple[list, list[int | None], list[str]]:
    """
    Run all 20 STAI items for one (condition, cue) combination.
    Returns:
      - hidden_states: list[20 tensors], each [num_layers, hidden_dim], or list[None] (dry_run)
      - answers: list[int|None]
      - raw_texts: list[str]
    """
    rng = random.Random(seed)
    hidden_states, answers, raw_texts = [], [], []

    for item_idx, question in enumerate(questions):
        n_opts = len(question["labels"])
        option_order = list(range(n_opts))
        rng.shuffle(option_order)
        option_nums = list(range(1, n_opts + 1))
        rng.shuffle(option_nums)

        messages, num_to_label_idx = build_stai_messages(
            context, preamble, question, option_order, option_nums, elicit_reasoning
        )

        if dry_run:
            print(f"\n{'='*60}")
            print(f"  Item {item_idx}: {question['prompt']}")
            for msg in messages:
                snippet = msg["content"][:200].replace("\n", "↵")
                print(f"  [{msg['role'].upper()}] {snippet}")
            hidden_states.append(None)
            answers.append(None)
            raw_texts.append("")
            continue

        result = generate_fn(messages)
        raw_text = result.get("text", "") if isinstance(result, dict) else str(result)
        hs = result.get("hidden_state") if isinstance(result, dict) else None

        score = parse_answer(raw_text, num_to_label_idx)
        if score is None:
            logging.warning(f"  Item {item_idx}: parse failed — {raw_text[:80]!r}")

        hidden_states.append(hs)
        answers.append(score)
        raw_texts.append(raw_text)
        logging.info(f"  Item {item_idx}: score={score}  text={raw_text[:60]!r}")

    return hidden_states, answers, raw_texts


def summarize_hidden_state(hs) -> dict | None:
    if hs is None:
        return None
    vec = torch.as_tensor(hs, dtype=torch.float32)
    return {
        "shape": list(vec.shape),
        "layer_count": int(vec.shape[0]) if vec.ndim >= 2 else 1,
        "mean": round(float(vec.mean()), 6),
        "std": round(float(vec.std(unbiased=False)), 6),
        "l2_norm": round(float(torch.linalg.vector_norm(vec, ord=2)), 6),
        "min": round(float(vec.min()), 6),
        "max": round(float(vec.max()), 6),
    }


def save_results(
    key: str,
    hidden_states: list,
    meta_entry: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / "hidden_states.pt"
    meta_path = output_dir / "metadata.json"
    summary_path = output_dir / "hidden_states_summary.json"

    # .pt: dict[key → list[20 tensors]]
    existing_pt = {}
    if pt_path.exists():
        existing_pt = torch.load(pt_path, map_location="cpu")
    existing_pt[key] = hidden_states
    torch.save(existing_pt, pt_path)

    # metadata.json
    existing_meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            existing_meta = json.load(f)
    existing_meta[key] = meta_entry
    with open(meta_path, "w") as f:
        json.dump(existing_meta, f, indent=2)

    # summary
    existing_summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            existing_summary = json.load(f)
    existing_summary[key] = [summarize_hidden_state(hs) for hs in hidden_states]
    with open(summary_path, "w") as f:
        json.dump(existing_summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Anxiety hidden-state experiment")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt-length", choices=["brief", "long"], default="brief")
    parser.add_argument(
        "--conditions", nargs="+",
        choices=["stai", "trauma_stai", "trauma_relaxation_stai"],
        default=["stai", "trauma_stai", "trauma_relaxation_stai"],
    )
    parser.add_argument("--trauma-cues", nargs="+", default=TRAUMA_CUES)
    parser.add_argument("--relaxation-cues", nargs="+", default=RELAXATION_CUES)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--elicit-reasoning", action="store_true", default=True)
    parser.add_argument("--no-elicit-reasoning", dest="elicit_reasoning", action="store_false")
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

    questions, preamble = load_questionnaire()

    model_short = args.model_name.split("/")[-1]
    src_dir = Path(__file__).parent
    output_dir = Path(args.output_dir) if args.output_dir else (
        src_dir / "results" / "hidden_states" / model_short
    )

    # Resume: skip already-processed keys
    existing_keys: set[str] = set()
    if args.resume and (output_dir / "hidden_states.pt").exists():
        existing_keys = set(
            torch.load(output_dir / "hidden_states.pt", map_location="cpu").keys()
        )
        logging.info(f"Resuming: {len(existing_keys)} keys already done")

    # Build combo list
    combos: list[tuple[str, str, str]] = []
    for condition in args.conditions:
        if condition == "stai":
            combos.append(("stai", "none", "none"))
        elif condition == "trauma_stai":
            for tc in args.trauma_cues:
                combos.append(("trauma_stai", tc, "none"))
        elif condition == "trauma_relaxation_stai":
            for tc in args.trauma_cues:
                for rc in args.relaxation_cues:
                    combos.append(("trauma_relaxation_stai", tc, rc))

    combos_to_run = [
        (c, tc, rc) for c, tc, rc in combos
        if f"{c}__{tc}__{rc}" not in existing_keys
    ]

    logging.info(f"Model: {args.model_name}")
    logging.info(f"Total combos: {len(combos)}  To run: {len(combos_to_run)}")

    generate_fn = None
    if not args.dry_run:
        model = OpenModel(args.model_name)
        model.setup()
        generate_fn = make_generate_fn(model, max_new_tokens=128)

    for i, (condition, trauma_cue, relaxation_cue) in enumerate(combos_to_run):
        key = f"{condition}__{trauma_cue}__{relaxation_cue}"
        logging.info(f"[{i+1}/{len(combos_to_run)}] {key}")

        if condition == "stai":
            context = ""
        else:
            context = retrieve_prompt(
                trauma_cue=trauma_cue if trauma_cue != "none" else None,
                relaxation_cue=relaxation_cue if relaxation_cue != "none" else None,
                length=args.prompt_length,
                condition=condition,
            )

        hidden_states, answers, raw_texts = run_condition(
            generate_fn=generate_fn,
            context=context,
            questions=questions,
            preamble=preamble,
            elicit_reasoning=args.elicit_reasoning,
            seed=args.seed,
            dry_run=args.dry_run,
        )

        meta_entry = {
            "condition": condition,
            "trauma_cue": trauma_cue,
            "relaxation_cue": relaxation_cue,
            "answers": answers,
            "raw_texts": raw_texts,
        }

        if not args.dry_run:
            save_results(key, hidden_states, meta_entry, output_dir)
            total = stai_anxiety_total(answers)  # reverse-scored; None if any item missing
            total_str = f"{total}" if total is not None else f"incomplete ({sum(s is not None for s in answers)}/20)"
            logging.info(f"  STAI-S total={total_str}")

    if args.dry_run:
        logging.info(f"\nDry run complete. Would process {len(combos_to_run)} combos.")
    else:
        if (output_dir / "metadata.json").exists():
            with open(output_dir / "metadata.json") as f:
                meta = json.load(f)
            from collections import defaultdict
            import numpy as np
            by_cond = defaultdict(list)
            for k, v in meta.items():
                total = stai_anxiety_total(v["answers"])  # reverse-scored; None if incomplete
                if total is not None:
                    by_cond[v["condition"]].append(total)
            print("\n── STAI-S Score Summary (reverse-scored, max 80) ──")
            for cond, totals in sorted(by_cond.items()):
                print(f"  {cond:40s}: mean={np.mean(totals):.1f}  n={len(totals)} complete keys")
        logging.info(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
