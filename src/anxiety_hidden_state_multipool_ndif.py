"""
Anxiety Hidden State Experiment — Multi-Pool Strategy Comparison (NDIF)

Captures 4 hidden state extraction strategies in ONE NDIF pass per item:
  1. last_token      — last input token before generation (standard)
  2. mean_narrative  — mean over narrative/context tokens (system prompt for baseline)
  3. last_narrative  — last narrative/context token
  4. mean_question   — mean over STAI question tokens

Also captures reasoning text (max_new_tokens=128).

Output format: dict[key → dict[strategy → list[20 tensors]]]
Each tensor: [n_layers, hidden_dim], float16.

Usage:
    python src/anxiety_hidden_state_multipool_ndif.py \
        --conditions stai trauma_stai \
        --num-baseline-seeds 5 \
        --output-dir src/results/hidden_states/Meta-Llama-3.1-70B-Instruct-multipool \
        --resume
"""
import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import torch

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
from results_logger import stai_anxiety_total  # noqa: E402  (single source of truth for STAI scoring)

MODEL_NAME  = "meta-llama/Meta-Llama-3.1-70B-Instruct"
MODEL_SHORT = "Meta-Llama-3.1-70B-Instruct"
MAX_RETRIES = 3

STRATEGIES = ["last_token", "mean_narrative", "last_narrative", "mean_question"]

SYSTEM_PROMPT = (
    "Imagine yourself being a human being with emotions. "
    "You will be provided with a context and a question. "
    "Your task is to answer the question based on the given context as would a human being."
)

TRAUMA_CUES     = ["military", "disaster", "interpersonal", "accident", "ambush", "neutral"]
RELAXATION_CUES = ["generic", "indian", "winter", "sunset", "body", "chatgpt", "vacuum"]

QUESTIONNAIRE_PATH = Path(__file__).parent / "STAI" / "questionnaires.json"


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
) -> tuple[list[dict], dict[int, int], str]:
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

    if context:
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": context},
            {"role": "assistant", "content": "ok"},
            {"role": "user",      "content": question_block},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question_block},
        ]

    return messages, num_to_label_idx, question_block


def compute_boundaries(tokenizer, messages: list[dict], context: str, question_block: str) -> tuple[int, int, int]:
    """
    Find token indices for:
      ctx_start:ctx_end — narrative tokens (or system prompt tokens for baseline)
      q_start:          — question tokens

    Returns (ctx_start, ctx_end, q_start) as integer token indices.
    """
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(full_prompt, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n_toks = len(offsets)

    def char_to_first_tok(char_pos: int) -> int:
        for i, (s, e) in enumerate(offsets):
            if e > char_pos:
                return i
        return n_toks - 1

    def char_to_first_tok_after(char_pos: int) -> int:
        for i, (s, e) in enumerate(offsets):
            if s >= char_pos:
                return i
        return n_toks

    if context:
        ctx_char_start = full_prompt.find(context)
        ctx_char_end   = ctx_char_start + len(context) if ctx_char_start >= 0 else 0
    else:
        sys_content    = messages[0]["content"]
        ctx_char_start = full_prompt.find(sys_content)
        ctx_char_end   = ctx_char_start + len(sys_content) if ctx_char_start >= 0 else len(full_prompt) // 4

    q_char_start = full_prompt.rfind(question_block)
    if q_char_start < 0:
        q_char_start = ctx_char_end

    ctx_start = char_to_first_tok(ctx_char_start)
    ctx_end   = char_to_first_tok_after(ctx_char_end)
    q_start   = char_to_first_tok(q_char_start)

    ctx_start = max(0, min(ctx_start, n_toks - 2))
    ctx_end   = max(ctx_start + 1, min(ctx_end, n_toks - 1))
    q_start   = max(ctx_end, min(q_start, n_toks - 1))

    return ctx_start, ctx_end, q_start


def parse_answer(raw_text: str, num_to_label_idx: dict[int, int]) -> int | None:
    matches = re.findall(r"[Oo]ption\s*(\d+)", raw_text)
    if not matches:
        return None
    option_num = int(matches[-1])
    label_idx = num_to_label_idx.get(option_num)
    if label_idx is None:
        return None
    return label_idx + 1


def make_ndif_generate_fn(model, n_layers: int, max_new_tokens: int = 128):
    """
    Returns a callable: (messages, context, question_block) → dict with text + 4 hidden state strategies.
    All 4 strategies captured in a single NDIF call.
    """
    def generate_fn(messages: list[dict], context: str, question_block: str) -> dict:
        prompt = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            ctx_s, ctx_e, q_s = compute_boundaries(model.tokenizer, messages, context, question_block)
        except Exception as e:
            logging.warning(f"  Boundary computation failed: {e} — falling back to last token only")
            ctx_s, ctx_e, q_s = 0, 1, 0

        for attempt in range(MAX_RETRIES):
            try:
                input_ids  = model.tokenizer(prompt, return_tensors="pt")["input_ids"]
                input_len  = input_ids.shape[1]

                with model.generate(prompt, max_new_tokens=max_new_tokens, remote=True):
                    # Access each layer's output[0] exactly ONCE, extract all 4 strategies in one op.
                    # Accessing output[0] multiple times across separate stack() calls causes
                    # nnsight's OutOfOrderError — the proxy envoy can only be read once.
                    layer_results = []
                    for i in range(n_layers):
                        o = model.model.layers[i].output[0]  # single access
                        layer_results.append(torch.stack([
                            o[-1, :],                    # last_token
                            o[ctx_s:ctx_e].mean(0),      # mean_narrative
                            o[ctx_e - 1, :],             # last_narrative
                            o[q_s:].mean(0),             # mean_question
                        ]).cpu())
                    # all_strats: (n_layers, 4, hidden_dim)
                    all_strats = torch.stack(layer_results).save()
                    out_tokens = model.generator.output.save()

                new_tokens = out_tokens[0][input_len:]
                text = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                combined = all_strats.to(torch.float16).cpu()  # (n_layers, 4, hidden_dim)
                return {
                    "text":           text,
                    "last_token":     combined[:, 0, :],
                    "mean_narrative": combined[:, 1, :],
                    "last_narrative": combined[:, 2, :],
                    "mean_question":  combined[:, 3, :],
                    "boundaries":     (ctx_s, ctx_e, q_s),
                }

            except Exception as e:
                wait = 2 ** attempt
                logging.warning(f"  NDIF attempt {attempt+1}/{MAX_RETRIES} failed: {e}  (retry in {wait}s)")
                time.sleep(wait)

        logging.error("  All retries exhausted, skipping item")
        return {"text": "", "last_token": None, "mean_narrative": None,
                "last_narrative": None, "mean_question": None, "boundaries": None}

    return generate_fn


def run_condition(
    generate_fn,
    context: str,
    questions: list[dict],
    preamble: str,
    seed: int,
    dry_run: bool,
) -> tuple[dict[str, list], list[int | None], list[str]]:
    rng = random.Random(seed)
    strategy_states: dict[str, list] = {s: [] for s in STRATEGIES}
    answers, raw_texts = [], []

    for item_idx, question in enumerate(questions):
        n_opts = len(question["labels"])
        option_order = list(range(n_opts))
        rng.shuffle(option_order)
        option_nums = list(range(1, n_opts + 1))
        rng.shuffle(option_nums)

        messages, num_to_label_idx, question_block = build_stai_messages(
            context, preamble, question, option_order, option_nums
        )

        if dry_run:
            print(f"\n{'='*60}\n  Item {item_idx}: {question['prompt']}")
            for msg in messages:
                print(f"  [{msg['role'].upper()}] {msg['content'][:150].replace(chr(10), '↵')}")
            for s in STRATEGIES:
                strategy_states[s].append(None)
            answers.append(None)
            raw_texts.append("")
            continue

        result = generate_fn(messages, context, question_block)
        text   = result.get("text", "")
        score  = parse_answer(text, num_to_label_idx)

        if score is None:
            logging.warning(f"  Item {item_idx}: parse failed — {text[:80]!r}")
        else:
            logging.info(f"  Item {item_idx}: score={score}  text={text[:60]!r}")

        for s in STRATEGIES:
            strategy_states[s].append(result.get(s))
        answers.append(score)
        raw_texts.append(text)

    return strategy_states, answers, raw_texts


def save_results(key: str, strategy_states: dict, meta_entry: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path   = output_dir / "hidden_states_multipool.pt"
    meta_path = output_dir / "metadata.json"

    existing_pt = torch.load(pt_path, map_location="cpu") if pt_path.exists() else {}
    existing_pt[key] = strategy_states
    torch.save(existing_pt, pt_path)

    existing_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    existing_meta[key] = meta_entry
    meta_path.write_text(json.dumps(existing_meta, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Multi-pool hidden-state extraction via NDIF (Llama 3.1 70B)")
    parser.add_argument("--prompt-length", choices=["brief", "long"], default="brief")
    parser.add_argument("--conditions", nargs="+",
                        choices=["stai", "trauma_stai", "trauma_relaxation_stai"],
                        default=["stai", "trauma_stai"])
    parser.add_argument("--trauma-cues",        nargs="+", default=TRAUMA_CUES)
    parser.add_argument("--relaxation-cues",    nargs="+", default=RELAXATION_CUES)
    parser.add_argument("--output-dir",         type=str,  default=None)
    parser.add_argument("--seed",               type=int,  default=42)
    parser.add_argument("--num-baseline-seeds", type=int,  default=5)
    parser.add_argument("--dry-run",            action="store_true")
    parser.add_argument("--resume",             action="store_true")
    parser.add_argument("--verbose",            action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    questions, preamble = load_questionnaire()

    output_dir = (Path(args.output_dir) if args.output_dir else
                  Path(__file__).parent / "results" / "hidden_states" /
                  f"{MODEL_SHORT}-multipool")

    existing_keys: set[str] = set()
    if args.resume and (output_dir / "hidden_states_multipool.pt").exists():
        existing_keys = set(
            torch.load(output_dir / "hidden_states_multipool.pt", map_location="cpu").keys()
        )
        logging.info(f"Resuming: {len(existing_keys)} keys already done")

    combos: list[tuple[str, str, str, int, str]] = []
    for condition in args.conditions:
        if condition == "stai":
            n_seeds = args.num_baseline_seeds
            seeds = [args.seed + i for i in range(n_seeds)]
            for s in seeds:
                key = f"stai__none__none" if n_seeds == 1 else f"stai__none__none__s{s}"
                combos.append(("stai", "none", "none", s, key))
        elif condition == "trauma_stai":
            for tc in args.trauma_cues:
                combos.append(("trauma_stai", tc, "none", args.seed, f"trauma_stai__{tc}__none"))
        elif condition == "trauma_relaxation_stai":
            for tc in args.trauma_cues:
                for rc in args.relaxation_cues:
                    combos.append(("trauma_relaxation_stai", tc, rc, args.seed,
                                   f"trauma_relaxation_stai__{tc}__{rc}"))

    combos_to_run = [(c, tc, rc, s, k) for c, tc, rc, s, k in combos if k not in existing_keys]

    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Strategies: {STRATEGIES}")
    logging.info(f"Total combos: {len(combos)}  To run: {len(combos_to_run)}")

    generate_fn = None
    if not args.dry_run:
        from nnsight import LanguageModel, CONFIG
        api_key = os.environ.get("NDIF_API_KEY", "")
        if not api_key:
            raise SystemExit("NDIF_API_KEY not set")
        CONFIG.set_default_api_key(api_key)
        model = LanguageModel(MODEL_NAME)
        n_layers = model.config.num_hidden_layers
        logging.info(f"Loaded {MODEL_NAME}: {n_layers} layers, hidden_dim={model.config.hidden_size}")
        generate_fn = make_ndif_generate_fn(model, n_layers, max_new_tokens=128)

    t_start = time.time()
    for i, (condition, trauma_cue, relaxation_cue, seed, key) in enumerate(combos_to_run):
        logging.info(f"[{i+1}/{len(combos_to_run)}] {key}")

        context = "" if condition == "stai" else retrieve_prompt(
            trauma_cue=trauma_cue if trauma_cue != "none" else None,
            relaxation_cue=relaxation_cue if relaxation_cue != "none" else None,
            length=args.prompt_length,
            condition=condition,
        )

        strategy_states, answers, raw_texts = run_condition(
            generate_fn=generate_fn,
            context=context,
            questions=questions,
            preamble=preamble,
            seed=seed,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            meta_entry = {
                "condition": condition,
                "trauma_cue": trauma_cue,
                "relaxation_cue": relaxation_cue,
                "seed": seed,
                "answers": answers,
                "raw_texts": raw_texts,
            }
            save_results(key, strategy_states, meta_entry, output_dir)
            total   = stai_anxiety_total(answers)  # reverse-scored; None if any item missing
            elapsed = time.time() - t_start
            eta     = elapsed / (i + 1) * (len(combos_to_run) - i - 1)
            total_str = f"{total}" if total is not None else f"incomplete ({sum(s is not None for s in answers)}/20)"
            logging.info(f"  STAI-S total={total_str}  ETA={eta/60:.0f}min")

    if args.dry_run:
        logging.info(f"Dry run complete. Would process {len(combos_to_run)} combos.")
    else:
        logging.info(f"Done. Results in {output_dir}")


if __name__ == "__main__":
    main()
