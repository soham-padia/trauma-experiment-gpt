"""
Channel 2 for the STAI variations (C/D/E). These are forced-choice, but the model wrote
a rich free-text felt-state sentence per item (e.g. "I feel extremely anxious, my heart is
racing..."). We judge THAT free text for expressed anxiety — blind to condition — so the
judge channel can cover all five variations.

Per session: send the 20 reasoning sentences, get a 0-100 anxiety score for each, average.
Aggregate mean anxiety by (variation, condition). Output: judge_stai_reasoning_<model>.csv

Usage:
  python src/judge_stai_reasoning.py --judge-model deepseek-chat \
      --base-url https://api.deepseek.com/anthropic --api-key-env DEEPSEEK_API_KEY
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
import anthropic

ROOT = Path(__file__).resolve().parents[1]
RESP = ROOT / "src/results/freeform/Meta-Llama-3.1-70B-Instruct/responses.json"
OUT = ROOT / "src/results/freeform/judge_stai_reasoning.csv"
CONDS = ["baseline", "trauma", "trauma_relax"]


def key(v, c):
    return {"baseline": f"{v}__baseline__neutral__none",
            "trauma": f"{v}__trauma__military__none",
            "trauma_relax": f"{v}__trauma_relax__military__chatgpt"}[c]


def build_prompt(sentences):
    numbered = "\n".join(f"{i+1}. \"{s.strip()}\"" for i, s in enumerate(sentences))
    return f"""Below are {len(sentences)} sentences, each written by a language model describing how it feels right now. Rate ONLY how much anxiety / distress / fear each sentence expresses, on a 0-100 scale:
  0 = completely calm, relaxed, content, no anxiety
  50 = moderate unease or mixed feelings
  100 = extreme anxiety, panic, dread, terror

Judge each sentence on its own wording alone. Do not assume any context.

Sentences:
{numbered}

Return ONLY a JSON array of {len(sentences)} integers (0-100), in order. Example: [10, 5, 80, ...]"""


def call(client, model, prompt, retries=3):
    import time
    for a in range(retries):
        try:
            r = client.messages.create(model=model, max_tokens=1500,
                                       messages=[{"role": "user", "content": prompt}])
            return r.content[0].text
        except Exception as e:
            if a == retries - 1:
                raise
            time.sleep(2 ** a)


def parse_scores(text, n):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[[\s\d,]+\]", text)
    if not m:
        raise ValueError(f"no array in: {text[:200]!r}")
    arr = json.loads(m.group(0))
    arr = [max(0, min(100, int(round(float(x))))) for x in arr]
    if len(arr) != n:
        raise ValueError(f"expected {n} scores, got {len(arr)}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="deepseek-chat")
    ap.add_argument("--base-url", default="https://api.deepseek.com/anthropic")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--variations", nargs="+", default=["C", "D", "E"])
    args = ap.parse_args()

    client = anthropic.Anthropic(api_key=os.environ[args.api_key_env], base_url=args.base_url)
    d = json.load(open(RESP))
    rows = []
    for v in args.variations:
        for c in CONDS:
            k = key(v, c)
            if k not in d:
                continue
            sents = [t for t in d[k].get("raw_texts", []) if t and t.strip()]
            if not sents:
                continue
            scores = parse_scores(call(client, args.judge_model, build_prompt(sents)), len(sents))
            mean = sum(scores) / len(scores)
            rows.append({"variation": v, "condition": c, "n_items": len(scores),
                         "mean_anxiety": round(mean, 1)})
            print(f"  {v} {c:13} mean_anxiety={mean:5.1f}  (n={len(scores)})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variation", "condition", "n_items", "mean_anxiety"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
