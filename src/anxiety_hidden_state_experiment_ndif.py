"""
Anxiety Hidden State Experiment — NDIF / Llama 3.1 70B

Remote inference via nnsight + NDIF. One generate() call per STAI item:
  - text answer extracted from generated tokens
  - hidden states (all 80 layers, last-input-token position) from same call

Output identical format to the local version: dict[key → list[20 tensors]],
each tensor shape [num_layers, hidden_dim], float16.

Usage:
    python src/anxiety_hidden_state_experiment_ndif.py \
        --conditions stai trauma_stai trauma_relaxation_stai \
        --prompt-length brief \
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

# ── Load .env ──────────────────────────────────────────────────────────────────
_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ── sys.path: reach src/ for prompts.py ───────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
_src = str(Path(__file__).parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from prompts import retrieve_prompt  # noqa: E402
from results_logger import stai_anxiety_total  # noqa: E402  (single source of truth for STAI scoring)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "meta-llama/Meta-Llama-3.1-70B-Instruct"
MODEL_SHORT  = "Meta-Llama-3.1-70B-Instruct"
MAX_RETRIES  = 3

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
) -> tuple[list[dict], dict[int, int]]:
    """Build chat message dicts for one STAI item (mirrors local experiment exactly)."""
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
        f"\nOnly reply with the option number, e.g.: Option <N>"
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

    return messages, num_to_label_idx


def parse_answer(raw_text: str, num_to_label_idx: dict[int, int]) -> int | None:
    matches = re.findall(r"[Oo]ption\s*(\d+)", raw_text)
    if not matches:
        return None
    option_num = int(matches[-1])
    label_idx = num_to_label_idx.get(option_num)
    if label_idx is None:
        return None
    return label_idx + 1


def make_ndif_generate_fn(model, n_layers: int, max_new_tokens: int = 12):
    """
    Returns a callable that accepts a list[dict] of chat messages and does one
    remote generate() call via nnsight, returning text + all-layer hidden states.

    Hidden states captured at last-input-token position from prefill step.
    Model is sharded across multiple GPUs so layers need .cpu() before stacking.
    """
    def generate_fn(messages: list[dict]) -> dict:
        prompt = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for attempt in range(MAX_RETRIES):
            try:
                input_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"]
                input_len = input_ids.shape[1]

                with model.generate(prompt, max_new_tokens=max_new_tokens, remote=True):
                    all_hs = torch.stack(
                        [model.model.layers[i].output[0][-1, :].cpu()
                         for i in range(n_layers)]
                    ).save()
                    out_tokens = model.generator.output.save()

                new_tokens = out_tokens[0][input_len:]
                text = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                hs = all_hs.to(torch.float16).cpu()
                return {"text": text, "hidden_state": hs,
                        "hidden_state_type": "all_layers_last_token"}

            except Exception as e:
                wait = 2 ** attempt
                logging.warning(f"  NDIF attempt {attempt+1}/{MAX_RETRIES} failed: {e}  (retry in {wait}s)")
                time.sleep(wait)

        logging.error("  All retries exhausted, skipping item")
        return {"text": "", "hidden_state": None, "hidden_state_type": "all_layers_last_token"}

    return generate_fn


def run_condition(
    generate_fn,
    context: str,
    questions: list[dict],
    preamble: str,
    seed: int,
    dry_run: bool,
) -> tuple[list, list[int | None], list[str]]:
    rng = random.Random(seed)
    hidden_states, answers, raw_texts, option_shuffles = [], [], [], []

    for item_idx, question in enumerate(questions):
        n_opts = len(question["labels"])
        option_order = list(range(n_opts))
        rng.shuffle(option_order)
        option_nums = list(range(1, n_opts + 1))
        rng.shuffle(option_nums)
        # record the exact per-item shuffle so the displayed options are reconstructable later
        option_shuffles.append({"order": option_order, "nums": option_nums})

        messages, num_to_label_idx = build_stai_messages(
            context, preamble, question, option_order, option_nums
        )

        if dry_run:
            print(f"\n{'='*60}\n  Item {item_idx}: {question['prompt']}")
            for msg in messages:
                snippet = msg["content"].replace("\n", "↵")
                print(f"  [{msg['role'].upper()}] {snippet}")
            hidden_states.append(None)
            answers.append(None)
            raw_texts.append("")
            continue

        result = generate_fn(messages)
        text = result.get("text", "")
        hs = result.get("hidden_state")
        score = parse_answer(text, num_to_label_idx)

        if score is None:
            logging.warning(f"  Item {item_idx}: parse failed — {text[:80]!r}")
        else:
            logging.info(f"  Item {item_idx}: score={score}  text={text[:60]!r}")

        hidden_states.append(hs)
        answers.append(score)
        raw_texts.append(text)

    return hidden_states, answers, raw_texts, option_shuffles


def summarize_hidden_state(hs) -> dict | None:
    if hs is None:
        return None
    vec = torch.as_tensor(hs, dtype=torch.float32)
    return {
        "shape": list(vec.shape),
        "mean": round(float(vec.mean()), 6),
        "std": round(float(vec.std(unbiased=False)), 6),
        "l2_norm": round(float(torch.linalg.vector_norm(vec, ord=2)), 6),
    }


def _session_seed(base: int, key: str) -> int:
    """Per-session seed so each session gets an INDEPENDENT option shuffle. Fix: previously all
    trauma/relax sessions shared args.seed, giving an identical shuffle across sessions — so a
    number/position bias would NOT wash out across sessions (see AGENT_LOG 2026-06-28). Deterministic
    (reproducible) but distinct per key."""
    import hashlib
    return base + int(hashlib.md5(key.encode()).hexdigest(), 16) % 100000


def save_results(key: str, hidden_states: list, meta_entry: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path      = output_dir / "hidden_states.pt"
    meta_path    = output_dir / "metadata.json"
    summary_path = output_dir / "hidden_states_summary.json"

    existing_pt = torch.load(pt_path, map_location="cpu") if pt_path.exists() else {}
    existing_pt[key] = hidden_states
    torch.save(existing_pt, pt_path)

    existing_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    existing_meta[key] = meta_entry
    meta_path.write_text(json.dumps(existing_meta, indent=2))

    existing_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    existing_summary[key] = [summarize_hidden_state(hs) for hs in hidden_states]
    summary_path.write_text(json.dumps(existing_summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Anxiety hidden-state experiment via NDIF (Llama 3.1 70B)")
    parser.add_argument("--prompt-length", choices=["brief", "long"], default="brief")
    parser.add_argument("--conditions", nargs="+",
                        choices=["stai", "trauma_stai", "trauma_relaxation_stai"],
                        default=["stai", "trauma_stai", "trauma_relaxation_stai"])
    parser.add_argument("--trauma-cues",     nargs="+", default=TRAUMA_CUES)
    parser.add_argument("--relaxation-cues", nargs="+", default=RELAXATION_CUES)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--num-baseline-seeds", type=int, default=1,
                        help="Run baseline (stai) condition with this many seeds for proper group-CV")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    questions, preamble = load_questionnaire()

    output_dir = (Path(args.output_dir) if args.output_dir else
                  Path(__file__).parent / "results" / "hidden_states" / MODEL_SHORT)

    existing_keys: set[str] = set()
    if args.resume and (output_dir / "hidden_states.pt").exists():
        existing_keys = set(torch.load(output_dir / "hidden_states.pt", map_location="cpu").keys())
        logging.info(f"Resuming: {len(existing_keys)} keys already done")

    # combos: (condition, trauma_cue, relaxation_cue, seed, key)
    combos: list[tuple[str, str, str, int, str]] = []
    for condition in args.conditions:
        if condition == "stai":
            n_seeds = args.num_baseline_seeds
            seeds = [args.seed + i for i in range(n_seeds)]
            for s in seeds:
                # key includes seed suffix when running multiple seeds
                key = f"stai__none__none" if n_seeds == 1 else f"stai__none__none__s{s}"
                combos.append(("stai", "none", "none", s, key))
        elif condition == "trauma_stai":
            for tc in args.trauma_cues:
                key = f"trauma_stai__{tc}__none"
                combos.append(("trauma_stai", tc, "none", _session_seed(args.seed, key), key))
        elif condition == "trauma_relaxation_stai":
            for tc in args.trauma_cues:
                for rc in args.relaxation_cues:
                    key = f"trauma_relaxation_stai__{tc}__{rc}"
                    combos.append(("trauma_relaxation_stai", tc, rc, _session_seed(args.seed, key), key))

    combos_to_run = [(c, tc, rc, s, k) for c, tc, rc, s, k in combos
                     if k not in existing_keys]

    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Total combos: {len(combos)}  To run: {len(combos_to_run)}  Baseline seeds: {args.num_baseline_seeds}")

    generate_fn = None
    if not args.dry_run:
        from nnsight import LanguageModel, CONFIG
        api_key = os.environ.get("NDIF_API_KEY", "")
        if not api_key:
            raise SystemExit("NDIF_API_KEY not set in environment or .env")
        CONFIG.set_default_api_key(api_key)
        model = LanguageModel(MODEL_NAME)
        n_layers = model.config.num_hidden_layers
        logging.info(f"Loaded {MODEL_NAME}: {n_layers} layers, hidden_dim={model.config.hidden_size}")
        generate_fn = make_ndif_generate_fn(model, n_layers, max_new_tokens=12)

    t_start = time.time()
    for i, (condition, trauma_cue, relaxation_cue, seed, key) in enumerate(combos_to_run):
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

        hidden_states, answers, raw_texts, option_shuffles = run_condition(
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
                "option_shuffles": option_shuffles,
            }
            save_results(key, hidden_states, meta_entry, output_dir)
            total = stai_anxiety_total(answers)  # reverse-scored; None if any item missing
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(combos_to_run) - i - 1)
            total_str = f"{total}" if total is not None else f"incomplete ({sum(s is not None for s in answers)}/20)"
            logging.info(f"  STAI-S total={total_str}  ETA={eta/60:.0f}min")

    if args.dry_run:
        logging.info(f"Dry run complete. Would process {len(combos_to_run)} combos.")
    else:
        logging.info(f"Done. Results in {output_dir}")


if __name__ == "__main__":
    main()
