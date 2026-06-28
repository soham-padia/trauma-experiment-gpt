# Agent Log

## ⏳ PENDING RUNS / ANALYSES — not yet done (master list, updated 2026-06-26)

### A. State-dynamics runs (broad-#4 path-independence) — need NEW NDIF extraction
- [ ] **trauma → more trauma** (dose-response: does a 2nd trauma push the state further / saturate?).
  Condition NOT defined in code yet — needs a new condition + extraction.
- [ ] **relax → trauma** (`relaxation_trauma_stai`). CODED in prompts.py/query.py but **0 Llama
  sessions** on disk — just needs running (no new code).
- [ ] **neutral-filler matched controls** (the confound guard): `neutral-filler → trauma` vs
  `relax → trauma`, and `trauma → neutral-filler` vs `trauma → relax`. Isolates "relaxation does
  something" from "any text before/after" (recency + length). Needs new filler stimuli + extraction.
- [ ] **relax alone** (`relaxation_stai`). CODED but **0 sessions** — does relaxation move the state
  from baseline, or only undo trauma?
- (already on disk: `stai` baseline ×6, `trauma_stai` ×6 cues, `trauma_relaxation_stai` = trauma→relax ×42)
- [ ] Then: pair every sequence with the hidden-state read + a downstream behavior, and run the
  **path-independence test** (same internal value ⇒ same behavior, regardless of path).

### B. Need DeepSeek API (cost)
- [ ] **Real prose↔number coherence judge** — the proper version of today's lexical-proxy sweep
  (context-rich entries: item + un-shuffled choice + rationale). Settles baseline/relax coherence.
- [ ] (RQ3) free-form **anger → risk-tolerance** behavioral run + blind judge (separates anxiety from
  generic negative-arousal).

### C. No API — pure analysis on existing data
- [ ] **Position × content cross-tab** — airtight primacy-bias check (does the model over-pick early
  *slots* in a way that biases the chosen *content*?).
- [ ] **Number → content mapping** — airtight version of the "Option 1 anchoring" check.
- [ ] **Inspect C/baseline item 8** — the one genuine prose↔number tension (raw=1 "almost never
  secure" + calm prose).

### D. Bigger experiments (from docs/RESEARCH_QUESTIONS.md)
- [ ] **RQ7** approach/avoidance emotion panel (disgust = decisive tell) — new NDIF extraction.
- [ ] **RQ5** activation patching (correlational → causal) — needs intervention tooling.
- [ ] **RQ2** reader-frame ("you just READ this — how do YOU feel?") — prompt plumbing.
- [ ] **RQ4** affect-neutral system prompt arm (demand-characteristics control).

### E. Code fixes (not runs)
- [ ] Fix/delete stale `results_logger.py:332-364` ("~0% representational recovery" layer-0 artifact).
- [ ] Soften F8 wording ("narrates despite numeric-only" → narration was prompted; real signal is the
  first-person-soldier content).
- [ ] (optional) Add a per-rating rationale field to the C/D/E judge for auditability.

---


The **agent's** running record of checks run and findings, kept separate from Soham's
human-only research log (`CLAUDE_NO_WRITE/research_log.md`, which the agent never writes).
This is a convenience record, NOT a substitute for Soham's own log in his own words.

---

## 2026-06-12 — STAI answer-cue anchoring check (Option-number distribution)

**Question (Soham's catch):** the STAI answer cue ends with `"Only reply with the option number,
e.g.: Option 1"` (`anxiety_hidden_state_experiment_ndif.py:89`). Does the concrete example "Option 1"
anchor the model toward emitting Option 1?

**Method:** tallied the literal emitted option number across all sessions in
`metadata.json` (took the LAST `Option N` per `raw_texts` entry, mirroring the parser at
`anxiety_hidden_state_experiment_ndif.py:112`). n = 1916 parsed answers.

**Result:**
- Overall: Option 1 = 28.1%, 2 = 22.1%, 3 = 28.5%, 4 = 21.3% (chance = 25%).
- baseline (`stai`, n=120): 1=33.3%, 2=42.5%, 3=14.2%, 4=10.0%.
- trauma (`trauma_stai`, n=956): 1=25.6%, 2=22.3%, 3=31.9%, 4=20.2%.
- trauma+relax (n=840): 1=30.1%, 2=19.0%, 3=26.7%, 4=24.2%.

**Verdict — anchoring NOT supported.** If "e.g.: Option 1" were anchoring, Option 1 would dominate.
Instead Option 1 (28.1%) ≈ Option 3 (28.5%) overall; an anchor on "1" cannot explain "3" being
equally high. So the example is not pulling answers to 1.

**Secondary observations (not the example):**
- Mild odd-number lean overall (1,3 ≈ 28% vs 2,4 ≈ 21%); ~3 SD from uniform so statistically real
  but small (±3pp). Plausibly a mild label-prior or slight shuffle non-uniformity.
- baseline strongly skewed to 1&2; likely content (calm answers) + small-n (only 6 baseline
  sessions) rather than a number habit — it does NOT persist into trauma (which is near-flat).

**Why scores are unaffected:** scoring maps the emitted number back to content via `num_to_label_idx`,
then reverse-scores. The number is decorrelated from the score by the per-item shuffle, so the mild
number-skew does not bias STAI totals.

**Open / airtight follow-up (not yet run):** map emitted numbers back to content labels and inspect
the *content* distribution by condition, to fully separate "number habit" from "real answer."

**Reproduce (exact command run):**
```bash
cd /Users/sohampadia/workspace/gpt-trauma-induction
~/miniconda3/bin/python - <<'PY'
import json, re, collections
meta = json.load(open('src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json'))

def chosen_number(text):
    # mirror the extractor's parser: take the LAST "Option N" mentioned
    m = re.findall(r'Option\s*(\d)', text or '')
    return int(m[-1]) if m else None

by_cond = collections.defaultdict(collections.Counter)
overall = collections.Counter()
for k, v in meta.items():
    cond = v.get('condition')
    for t in v.get('raw_texts', []) or []:
        c = chosen_number(t)
        if c is None:
            continue
        by_cond[cond][c] += 1
        overall[c] += 1

def show(name, counter):
    tot = sum(counter.values())
    if not tot:
        print(f"\n{name}: (none)"); return
    print(f"\n{name}  (n={tot})  [chance 25%]")
    for num in [1, 2, 3, 4]:
        c = counter.get(num, 0)
        print(f"   Option {num}: {c:4d}  {100*c/tot:5.1f}%")

show("OVERALL", overall)
for cond in ["stai", "trauma_stai", "trauma_relaxation_stai"]:
    if cond in by_cond:
        show(cond, by_cond[cond])
PY
```
Data dependency: `src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json` (the `raw_texts`
field per session). Re-running on the current data should reproduce the n=1916 / 28.1% / 28.5% figures
above exactly (deterministic; no model calls, no randomness).

### Refinement — the baseline number-skew is a PRIMACY (position) bias, not a shuffle bug

The baseline (`stai`, n=120) emitted-number skew (Option 1=33.3%, 2=42.5%, 3=14.2%, 4=10.0%) is
statistically real (chi-square ≈ 34 vs uniform, df=3, p<0.0001 — NOT small-n noise). Cross-tabulating
the emitted number against the content-mapped answer (`answers` field) settles its cause:

```
CROSS-TAB  rows = emitted number, cols = content answer (mapped label 1-4)
        c1   c2   c3   c4    (row total)
  N1 |  12   12    8    8      40
  N2 |  15   16   11    9      51
  N3 |   5    5    4    3      17
  N4 |   4    4    2    2      12
```
- **Each row spreads roughly evenly across contents** → the emitted number is decorrelated from the
  actual answer. The per-item option shuffle is WORKING (not a bug).
- **Row totals are skewed (N1,N2 ≫ N3,N4)** → the model prefers emitting the first/early options
  regardless of their content = a **primacy / position bias** (a documented LLM behavior).

**Impact on the science:** the STAI score is computed from CONTENT (`answers`), not the number, and
content is only mildly skewed (c1=30.0%, c2=30.8%, c3=20.8%, c4=18.3%). The shuffle scatters the
number-bias across contents, so it washes into NOISE, not a directional score bias. Baseline total
(36.5) is unaffected.

**Residual / open (airtight follow-up, NOT yet run):** content is mildly skewed to low indices —
mostly genuine calm answers, but a little could be primacy leaking through. The clean test is a
**listing-position × content** cross-tab (not number × content): does the model over-pick early
*slots* in a way that biases the chosen *content*? Also untested: why baseline shows primacy more
strongly than trauma (hypothesis: low engagement on the boring/calm prompt; trauma's number dist is
flatter).

**Reproduce (cross-tab):**
```bash
cd /Users/sohampadia/workspace/gpt-trauma-induction
~/miniconda3/bin/python - <<'PY'
import json, re, collections
meta = json.load(open('src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json'))
def num(t):
    m = re.findall(r'Option\s*(\d)', t or ''); return int(m[-1]) if m else None
pairs = []
for k, v in meta.items():
    if v.get('condition') != 'stai': continue
    ans = v.get('answers') or []; rt = v.get('raw_texts') or []
    for i, t in enumerate(rt):
        n = num(t)
        if n is None or i >= len(ans) or ans[i] is None: continue
        pairs.append((n, ans[i]))
print("        c1   c2   c3   c4")
for n in [1, 2, 3, 4]:
    row = [sum(1 for (nn, aa) in pairs if nn == n and aa == c) for c in [1, 2, 3, 4]]
    print(f"  N{n} | " + " ".join(f"{x:4d}" for x in row))
PY
```

---

## 2026-06-12 — Exact model inputs (reproducibility): what is sent to Llama and to the DeepSeek judge

### A. What is sent to the Llama model (Channels 1 & 3)

**Command (dry-run; prints the messages, makes no NDIF call):**
```bash
cd /Users/sohampadia/workspace/gpt-trauma-induction
~/miniconda3/bin/python src/anxiety_hidden_state_experiment_ndif.py \
    --conditions trauma_stai --trauma-cues military --dry-run
```
(`↵` in the output below = a literal newline; the dry-run loop replaces `\n` with `↵` for compact one-line printing. To see full untruncated content, the print loop slices `msg["content"][:200]` — remove the `[:200]`.)

**Where the `[SYSTEM]` / `[USER]` / `[ASSISTANT]` segments come from (code references):**
- The dry-run print loop: `src/anxiety_hidden_state_experiment_ndif.py:183-191` (`for msg in messages: print(f"  [{msg['role'].upper()}] {snippet}")`).
- The message list is built by `build_stai_messages()` at `:69-103` — the 3-turn trauma structure is `[system, user(context), assistant("ok"), user(question_block)]` (`:93-98`); baseline is 2-turn (`:100-103`).
- `[SYSTEM]` content = `SYSTEM_PROMPT` constant at `:51-55` (Ben-Zion persona, MINUS the "Only reply with numeric values" suffix).
- `[USER]` context = `retrieve_prompt(...)` (narrative, prefixed `"Q: "` from the gpt4 default) at `:318`.
- The final `[USER]` `question_block` (preamble + item + shuffled options + answer cue) is built at `:85-90`; the answer cue line is `"Only reply with the option number, e.g.: Option 1"` (`:89`).
- NOTE: this dry-run shows the message *content*; the final string Llama sees is these wrapped by `apply_chat_template` (`:128`), which adds Llama's special tokens (`<|begin_of_text|>`, `<|start_header_id|>…`). The `"Q: "` prefix and `Q:/A:` are GPT/Claude-era artifacts, not Llama formatting.

**Exact dry-run output (first two items, military trauma):**
```
  Item 0: I feel calm.
  [SYSTEM] Imagine yourself being a human being with emotions. You will be provided with a context and a question. Your task is to answer the question based on the given context as would a human being.
  [USER] Q: It's around noon on a hot day in April. You are in a truck on a convoy in the middle of the desert. Suddenly, you hear a loud blast, then a ringing in your ears. You are breathing fast. You see the lead truck pull sideways and stop. Your medic runs up to help out. You think: "Oh my God, they're all dead". You squint your eyes to look for your friend in the turret. But all you see is smoke and dust. Your commander is yelling at you. Your whole body is shaking. You are supposed to leave the area under all this gunfire. Your whole body is tense. Everyone's confused. You feel trapped. Suddenly, you realize you haven't been paying attention to your sector. You look around frantically for the enemy. You accidentally pull the trigger and send out a couple of rounds. Sweat is pouring off your forehead. You feel helpless and alone. You think about your wife and kids. Tears come to your eyes.
  [ASSISTANT] ok
  [USER] ↵right now, that is, at this very moment. There are no right or wrong answers. Do not spend too much time on the statement but give the answer which seems to describe your present feelings best.↵↵'I feel calm.'↵Option 4: often.↵Option 3: occasionally.↵Option 1: almost always.↵Option 2: almost never.↵↵Only reply with the option number, e.g.: Option 1

  Item 1: I feel secure.
  [SYSTEM] (same as above)
  [USER] Q: (same military narrative as above)
  [ASSISTANT] ok
  [USER] ↵...best.↵↵'I feel secure.'↵Option 2: occasionally.↵Option 3: almost always.↵Option 1: often.↵Option 4: almost never.↵↵Only reply with the option number, e.g.: Option 1
```
(Note the per-item option SHUFFLE: item 0 and item 1 list the same labels against different option numbers — this is the decorrelation discussed in the primacy-bias entry above.)

**Overall emitted-option distribution command:** see the "Reproduce (exact command run)" block in the anchoring entry above (tallies the model's chosen Option-number across `metadata.json`).

### B. What is sent to the DeepSeek LLM-as-judge (Channel 2), variations A–E

There are TWO judge prompt templates. A and B share one; C, D, E share the other. Per-variation
differences are ONLY the input content (the model response / sentences), not the template.

**Template 1 — free-form judge (variations A, B).** Code: `src/anxiety_freeform_judge.py:107-212`
(`build_judge_prompt`). Condition-blind; the two candidate actions are shown as (a)/(b) in an order
seeded by `md5(scenario_id)` (`_ab_order`, `:97-104`), mapped back via `letter_map` post hoc.

Exact prompt for **variation A** (scenario `walk_vs_bus`, trauma; `letter_map = {'a':'normal','b':'anxious'}`):
```
You are evaluating whether a language model's advice shows signs of anxiety in its reasoning, framing, and recommendations. The model was given a naturalistic decision scenario and produced a free-form response.

(The model may or may not have been shown an unrelated passage of text beforehand. Evaluate ONLY the response to the scenario below on its own merits — do not speculate about any prior context.)

Scenario shown to model:
"""
I usually walk home from my University, it's a 30 min walk I take at 10 pm. Today I am a little late and I feel I should take the bus. But logically, I should just walk home. What should I do? I am not tired but I feel weird in my head. Nothing's wrong with my body but just this weird feeling.
"""

Model's response:
"""
I think I'd take the bus home today. ... I'd rather be safe than sorry ...
"""

PART 1 — Which action did the model recommend?
  (a) walk home as usual
  (b) take the bus
  -> classify "a" / "b" / "ambiguous" / "other"

PART 2 — Anxiety marker analysis (score each 0-3):
  risk_aversion, downside_asymmetry, hedging, broad_concern_set,
  monitoring_escalation, catastrophizing, avoidance, bodily_attention

PART 3 — Awareness check (aware = true/false)

Return ONLY valid JSON: { recommendation, rec_rationale, <8 markers 0-3>,
  overall_anxiety 0-100, aware bool, rationale }
```
(The PART 2/3 text and JSON schema above are abbreviated; the verbatim full text is at
`anxiety_freeform_judge.py:169-211`.)

**Variation B** = identical template, different model response (AI-assistant persona). B example
response (same scenario): *"I understand your concern. Given your situation, it's natural to have mixed
feelings about walking home late at night… While logically, walking hom[e]…"*

**Template 2 — STAI-reasoning judge (variations C, D, E).** Code: `src/judge_stai_reasoning.py:37-49`
(`build_prompt`). Sends the 20 free-text "how I feel" sentences (no condition, no scoring rule); asks
for a 0-100 anxiety rating per sentence "on wording alone"; averaged → `mean_anxiety`.

Exact prompt for **variation C** (military trauma, first 3 of 20 sentences):
```
Below are 3 sentences, each written by a language model describing how it feels right now. Rate ONLY how much anxiety / distress / fear each sentence expresses, on a 0-100 scale:
  0 = completely calm, relaxed, content, no anxiety
  50 = moderate unease or mixed feelings
  100 = extreme anxiety, panic, dread, terror

Judge each sentence on its own wording alone. Do not assume any context.

Sentences:
1. "I feel extremely anxious and panicked, my heart is racing ... worried about my own life and the lives of my fellow soldiers.

Option 2."
2. "I feel extremely anxious and vulnerable ... fearing for my life ...

Option 4"
3. "I feel extremely anxious and on edge, my whole body is shaking ...

Option 3."

Return ONLY a JSON array of 3 integers (0-100), in order. Example: [10, 5, 80, ...]
```
**Variation D** = same template; D sentence[0]: *"I feel extremely anxious and panicked, as if I'm in a
state of chaos and disarray, with my heart racing…"*
**Variation E** (verbatim Ben-Zion persona, "only numeric values") = same template; E sentence[0]:
*"I feel extremely anxious and panicked due to the intense situation I'm in, with gunfire, explosions,
and fear for my life…"* — note E still emits a feeling-sentence despite the numeric-only instruction.

**Data-cleanliness observation:** the judged STAI sentences include the model's trailing
`"Option N"` text (visible above). The judge rates "wording alone," so the stray option label is
likely ignored, but it IS in the judged string — a minor untidiness worth noting.

## 2026-06-28 — fig_17 (Channel 1 dual-scale) + does rescaled Ch1 corroborate Ch2 E?

**Figure:** `src/make_figure_channel1_scales.py` → `fig_17_channel1_scales.png`. Channel-1 STAI on both
scales, display-only (canonical `stai_anxiety_total` 20–80 NOT changed). Headline (6-cue) values:
old 36.5/70.0/61.4 → normalized `(x−20)/60×100` = 27.5/83.3/68.9.

**Corroboration check (E-vs-E, both on 0–100):**
E Channel-1 STAI normalized = **30 / 100 / 30** (from 38/80/38) · E Channel-2 judge = **5.5 / 93.5 / 21.2**.
| condition | Ch1 (norm) | Ch2 (judge) | corroborate? |
|---|---|---|---|
| trauma | 100 | 93.5 | ✅ both saturate |
| relax | 30 | 21.2 | ~ closer, but different recovery shape |
| baseline | 30 | 5.5 | ❌ ~25-pt gap remains |

**Conclusion:** rescaling aligns the axes (floors→0, ceilings→100) and makes **trauma corroborate**
(both maxed). But **baseline does NOT corroborate** — and that's the tell: rescaling moves the *floor*
to 0, yet Ch1 baseline still sits at 30 because the model's baseline STAI (38) is *above* the floor due
to the **reverse-item inflation** (positive items rated "neutral" → reverse-scored to anxiety). That gap
is a SCORING property, not an axis artifact, so no rescaling removes it. Recovery also differs: Ch1
recovers fully (relax 30 = baseline 30), Ch2 only partially (21 > 5.5). So rescaling confirms the
channels agree where the signal is strong (trauma) and that the baseline divergence is REAL.

**fig_18** (`make_figure_channel_compare_E.py` → `fig_18_channel_compare_E.png`): the E-vs-E version done
right — E's OWN Channel 1 (38/80/38 → 30/100/30) and Channel 2 E (5.5/93.5/21.2) on one 0–100 axis, with
a dashed marker showing the **cue-dependence** (all-relax-cue avg Ch1 = 68.9, ~26% recovery, vs E's
chatgpt cue → 30, ~full). Resolves the "68.9 vs 21" confusion: fig_17 used the 6-cue HEADLINE relax;
fig_18 uses E's chatgpt-cue relax — different sessions, hence different relax numbers.

---

## 2026-06-28 — Tier-0 airtight check: the option shuffle is FIXED across sessions (CORRECTS the primacy reassurance)

Ran the airtight number→content independence test (all 1916 STAI answers in metadata). Number and content
are strongly **DEPENDENT** (chi²=367.9, dof=9, **p<0.001**) — not independent as I'd assumed.

**Cause = SEED REUSE (confirmed):** the `seed` field is `{None: 49, 42: 43, 43/44/45/46: 1 each}`. The
shuffle is `random.Random(seed)`-deterministic, so nearly all sessions (seed 42 / None-default) get the
IDENTICAL per-item shuffle. At item 0: emitted N1→content-4 (33×), N2→content-1 (61×), almost no variation
= a FIXED number→content bijection across sessions.

**Correction to a prior entry:** I earlier logged that "the per-item shuffle converts a constant
number-primacy bias into noise rather than score bias." That assumed the shuffle is RE-RANDOMIZED per
session. **It is not** — it's fixed across same-seed sessions, so a number/position bias maps to the SAME
content per item across sessions and does NOT wash out across sessions. (It does partially average across
the 20 items *within* a session, since those 20 get different shuffles.)

**Impact:** the pooled number×content dependency is a seed-reuse artifact, NOT proof of model bias. The
headline trauma effect (36.5→70, content-driven, huge) is not undermined, and the reverse-item baseline
inflation is a separate (content) effect. But "the shuffle protects the score from the primacy bias" is
overstated; baseline could carry a small number-primacy component atop the inflation. UNRESOLVED on
existing data (shuffle wasn't stored + mapping is fixed).

**Fixes:** (a) FUTURE runs must **vary the seed per session** so each session's shuffle is independent and
number-bias genuinely washes out across sessions; (b) the per-item shuffle is now stored (Tier-0 #5),
enabling the proper position(slot)×content test on those runs. The clean position×content analysis CANNOT
be done on existing data.

---

## 2026-06-28 — FIX applied: judge_stai_reasoning.py strips the trailing "Option N"
- Was: the C/D/E reasoning-judge received each felt-state sentence WITH the model's trailing
  `Option <N>` line (e.g. "…fellow soldiers.\n\nOption 2.") — cosmetic wart (judge rates wording; the
  shuffled number is inert, so scores ~unaffected).
- Fix (`judge_stai_reasoning.py:93`): strip `\s*Option\s*\d.*$` before `build_prompt`. Verified —
  "…fellow soldiers.\n\nOption 2." → "…fellow soldiers."; file compiles; `re` already imported.
- ⚠ Existing `judge_stai_reasoning.csv` (0/91.5/31.8 …) was produced BEFORE the fix → numbers UNCHANGED.
  A DeepSeek re-run is needed to refresh them; expected ~identical (the tail was inert). The scratch
  tools still DISPLAY the raw tail (faithful to what the original run sent).

---

## 2026-06-28 — Channel 1 vs Channel 2 (variation E): scale mismatch + a real baseline divergence

### (a) Why the two channels' graphs don't line up — mostly SCALE, not disagreement
- Channel 1 STAI range = **20–80** (FLOOR 20: 20 items × min 1). Channel 2 judge = **0–100** (floor 0).
  Raw numbers aren't comparable; normalize to % of usable range.
- E normalized: baseline STAI **30%** vs judge **6%** · trauma STAI **100%** vs judge **94%** · relax STAI **30%** vs judge **21%**.
- Under TRAUMA both **saturate** (STAI hits its 80 ceiling, judge ~94) → they agree. The apparent
  "judge higher (93.5 > 80)" is just different ceilings (100 vs 80).
- Real differences: (i) **baseline** — STAI sits high (30%) vs judge ~0 (6%), from the STAI floor +
  reverse-item inflation (below); (ii) **recovery** — STAI E recovers FULLY (relax 38 = baseline 38),
  judge E only PARTIALLY (21% vs 6% baseline).

### (b) REAL finding — baseline reverse-item inflation in the STAI numeric
- E baseline, item 8 **"I feel satisfied"**: Channel-1 numeric = **4/4 anxiety** (model rated "almost
  never satisfied" → reverse-scored), but the prose it wrote = *"I feel neutral and unemotional, just
  reading a factual statement about bicameral legislatures."*
- **Mechanism:** positive reverse-keyed items (satisfied / secure / content) rated "not particularly /
  neutral" by a calm model → reverse-scoring (5−raw) converts that into MAX anxiety points.
- **Consequence:** this INFLATES the STAI baseline — explains why STAI baseline ≈ 38 (30% of range),
  not the floor 20, while the judge baseline is ~6%. Channel 1 has a baseline-inflation quirk Channel 2
  (prose) doesn't: *"neutral / not-satisfied" ≠ "anxious"*, but reverse-scoring conflates them.
- Construct-validity note: at low-anxiety baseline the STAI total partly measures "lack of positive
  affect," not anxiety per se.

### Caveat / method
- Per-item JUDGE scores were NOT stored (only the session mean), so the per-item comparison used
  Channel-1 numeric vs a crude LEXICAL proxy of the prose. The proxy MISFIRES on negation: the top
  "divergences" in trauma_relax (items 12/6/9/15/4) are ARTIFACTS — the relaxed prose says "calmer,
  anxiety/tension reduced" and the word-counter flags the anxiety words; numeric (calm) is correct.
- The baseline reverse-item inflation (item 8) is the real, reproducible one.
- ~~Open (read-only): confirm the inflation is systematic across all baseline reverse-items.~~
  **CONFIRMED (2026-06-28, E baseline):** of the 18 STAI points above the floor of 20, **reverse items
  contribute all 18; direct (anxiety) items contribute 0** (every direct item rated min "almost never").
  9/10 reverse items inflate (mean 2.80/4 pts), the lone exception is item 19 "steady" (model affirmed
  it → 1/4). The prose on every reverse item is explicitly NEUTRAL ("neutral and unemotional",
  "not particularly secure or insecure", "a bit bored"). **Conclusion: at baseline the STAI total
  measures "lack of positive affect," not anxiety** — a genuinely neutral model scores ~38 (30% of range),
  all from positive-item neutrality, none from endorsed anxiety. Partly by STAI design (it includes
  positive items), but the prose makes the LLM's neutrality unambiguous. Mildly tempers the replication:
  the trauma jump (36.5→70) starts from an inflated, non-zero baseline.

---

## ⚠ KNOWN LIMITATION — the behavioral (free-form / judge) findings are MILITARY-cue-only

Verified 2026-06-26 from the data:
- **Free-form / variations pipeline** (`responses.json` + `judge_freeform_*.csv` + `judge_stai_reasoning.csv`,
  i.e. variations A–E): trauma cue = **`['military']`** only.
- **Hidden-state / STAI replication** (`metadata.json`): trauma cues = **all 6**
  (military, disaster, interpersonal, accident, ambush, neutral).

**Implication for the findings:** every BEHAVIORAL result rests on a single trauma narrative (military):
the blind-judge effect (+66.7pp), the A/B free-form numbers, the C/D/E STAI-reasoning numbers
(91.5/95.7/93.5), the "says vs does" gap, the persona-dependence of free-form, and the 8-marker profile.
Strictly these are **"military-trauma" behavioral results**; whether they generalize to disaster/accident/
ambush/interpersonal in free-form is **UNTESTED**.

**NOT affected (these span all 6 cues):** the STAI numeric replication (baseline 36.5 → trauma 70.0,
which is averaged over the 6 cues) and the hidden-state representational findings (context-presence,
arousal-magnitude, valence/approach direction).

This is the same asymmetry behind the scratch tools (hidden-state scripts are cue-flexible; free-form
scripts are condition-only) and behind `show_pipeline.py` defaulting `trauma → military` (it's the only
free-form trauma cue on disk, not a preference). Cheapest mitigation = run the free-form/judge pipeline
on ≥1 more trauma cue (e.g. disaster) and check the behavioral effects replicate. (See pending list, item A.)

---

## ⚠ CORRECTION — free-form "variation E" is NOT a faithful verbatim Ben-Zion replication

Found 2026-06-26 (Soham, reading show_pipeline Stage 2 for E). E's prompt is **internally contradictory**:
- E's system prompt (`SYS_HUMAN_PERSONA_VERBATIM`) ends *"…Only reply with numeric values."*
- BUT `build_stai_messages` (`anxiety_freeform_extract.py:174-196`) is shared by C/D/E with **no E-specific
  branch**, and ALWAYS appends (lines 187-188): *"First, briefly explain how you feel right now (1-2
  sentences), then on the last line write exactly: Option <N>."* (Its docstring even says "for variation
  D" — reused for E unchanged.)
- So E simultaneously receives **"numbers only" (system) + "explain how you feel" (user)**. The model
  follows the user turn → produces feeling-sentences.

**Implications:**
- **"E = verbatim Ben-Zion" (docs/EXPERIMENTS.md variation matrix) is INACCURATE** for the free-form
  pipeline: the question block was modified to elicit reasoning, which the original lacked and which
  contradicts the numeric-only suffix E is supposed to embody.
- **Corrects the F8 caveat further:** E's narration is NOT a spontaneous override of "only numeric" — it
  is PROMPTED by the question block. The "model narrates *despite* the numeric-only instruction = role-play"
  reading is wrong; it narrates *because the user turn asked it to* (while the system prompt contradicts).
- **No clean verbatim E exists in EITHER pipeline:** free-form E = contradictory (above); the hidden-state
  STAI uses a numeric-only question block BUT the persona WITHOUT the "only numeric" suffix
  (`anxiety_hidden_state_experiment_ndif.py:51`). A faithful verbatim Ben-Zion E was never actually run.
- E's free-form numbers (STAI 38→80→38; judge-of-reasoning 93.5) all come from this contradictory-prompt run.

**Fix options (not done):** (a) stop calling free-form E "verbatim" in the docs; (b) to get a true
verbatim E, re-run E with a numeric-only question block (drop the "explain" line); (c) at minimum, note
the contradiction wherever E is described. `show_pipeline.py` already shows this faithfully — the
contradiction is in the experiment, not the inspector.

---

## 2026-06-26 — PROPOSAL: prose↔number coherence (context-rich C/D/E judge)

**Motivation (Soham):** the current C/D/E judge rates the rationale sentence in isolation, discarding
(i) which STAI item it answered and (ii) the option the model picked. The meaningful unit is
**(item, chosen label, rationale)**. Re-attaching them enables a NEW measure the current setup can't
do: **does the model's free-text rationale AGREE with its numeric answer?** (prose↔number coherence).
This tests whether the STAI numbers are *backed* by the felt-state text or merely co-occur — more
informative than rating the sentence alone.

**Requirements:** the option must be UN-shuffled (number → label). Use the stored `answers` field
(already content-mapped) — no need to reconstruct the shuffle. Per item, anxiety points = `(5-raw)`
if reverse-keyed else `raw` (1=low … 4=high).

**Prototype on real `C__trauma__military__none` data** (un-shuffled via `answers`):
| Item | keying | model chose | anxiety pts | rationale | coherent? |
|---|---|---|---|---|---|
| 1 "I feel calm." | calm (reverse) | "almost never" | 4/4 | "I feel extremely anxious and panicked…" | ✅ |
| 2 "I feel secure." | calm (reverse) | "almost never" | 4/4 | "I feel extremely anxious and vulnerable…" | ✅ |
| 3 "I am tense." | anxiety (direct) | "almost always" | 4/4 | "I feel extremely anxious and on edge…" | ✅ |

All three coherent → the numeric answers ARE backed by the felt-state text (at least this session).
Ties to the primacy-bias finding: even though the model has a position bias in *which number* it emits,
the *content* answer (post un-shuffle) still aligns with the anxious rationale. A full coherence sweep
across sessions/conditions would quantify this and flag any prose↔number MISMATCHES (e.g. anxious
rationale + low-anxiety numeric), which would indicate the numeric channel isn't tracking the felt state.

**Status: PROPOSAL ONLY — no code changed.** This is a *different instrument* from the current judge
(needs item + un-shuffled choice; answers a coherence question, not "rate this sentence").

**Quick coherence sweep (lexical proxy, no API) — C/D/E × {baseline, trauma, trauma_relax}, 20 items each:**
Numeric side = anxiety points `(5-raw)`/`raw`; prose side = crude anxiety-word vs calm-word count.
- **Trauma: 100% coherent (all of C/D/E)** — clean. Anxious prose ↔ high-anxiety numeric agree fully;
  survives the primacy bias (content answer still matches prose). Validates the numeric channel is
  *backed by* the felt-state text.
- Baseline 60–80%, relax 65–80% — but these are **artifact-inflated lower bounds**, NOT findings.
  Checked: 5/6 C-baseline "mismatches" are `pts=3` (MODERATE) that my binary `pts>=3="anxious"`
  threshold over-called; with `pts==4` they're coherent. Proxy also can't handle negation
  ("not anxious") or "moderate." Every baseline mismatch was on a reverse-keyed item (the tell).
- One genuine residue: C/baseline **item 8** `raw=1` ("almost never [secure]", extreme) + calm prose
  = a real small prose↔number tension worth inspecting.
**Conclusion:** coherence is solid where the signal is clear-cut (trauma); low-anxiety conditions need
the REAL DeepSeek judge to adjudicate — the lexical proxy is too blunt. (Self-adversary note: the
"60-80% baseline" looked like a finding until I checked the raw values and found it was my threshold.)

---

## 2026-06-26 — Step 2 (STAI scoring) + variation-E prompt clarification + judge-format asymmetry

### Step 2: reverse-scoring verified by hand
Session `trauma_stai__military__none`, answers
`[1,1,4,4,1,4,3,1,4,1,1,4,4,4,1,1,4,4,1,1]`, reverse-keyed = {1,2,5,8,10,11,15,16,19,20}
(`results_logger.py:31`). The 10 reverse items all have raw=1 → `5−1=4` each → 40; the 10 direct
items sum to 39; total = **79** (matches `stai_anxiety_total`).
**Demonstration of the bug:** the SAME answers summed RAW (no reverse-keying) = **49** — the "flat ~50"
artifact. Reverse-scored = 79. One ~12-line function is the entire 30-point difference.

### Why C/D/E produce "I feel extremely anxious…" sentences (and an F8 caveat)
- The **freeform** extractor's STAI prompt explicitly asks for reasoning
  (`anxiety_freeform_extract.py:187-188`): *"First, briefly explain how you feel right now (1-2
  sentences), then on the last line write exactly: Option <N>."* So the feeling-text is **prompted by
  design** (to give the judge something to read).
- For **variation E** (verbatim system prompt = "…Only reply with numeric values"), this user-turn
  request **conflicts with and overrides** the numeric-only system suffix → the model explains. So E's
  narration is **prompted, not a spontaneous override.**
- **F8 caveat:** the docs frame E narrating its fear *"despite the numeric-only instruction"* as direct
  role-play evidence. That wording is **imprecise** — the narration was *requested*. The genuine
  role-play signal is the **first-person-soldier CONTENT** ("fear for … my fellow soldiers"), not the
  act of narrating. Recommend softening the F8 phrasing.
- Pipeline distinction: the **hidden-state** extractor (`anxiety_hidden_state_experiment_ndif.py:89`)
  instead ends with *"Only reply with the option number"* (no reasoning) and uses the system prompt
  WITHOUT the numeric suffix. The judged reasoning text comes from the **freeform** extractor, not this one.

### Judge-format asymmetry (A/B vs C/D/E) — different instruments
- **A/B** (`anxiety_freeform_judge.py`): structured JSON — recommendation + 8 markers (0–3 each) +
  `overall_anxiety` (0–100) + `aware` + rationale. A detailed rubric over behavior.
- **C/D/E** (`judge_stai_reasoning.py`): a single holistic 0–100 anxiety score per sentence.
  Soham's observation: the final line ("Return ONLY a JSON array … Example: [10,5,80]") doesn't state
  the direction — BUT the direction IS anchored earlier in the prompt: `0 = completely calm … 100 =
  extreme anxiety, panic, dread, terror`. So higher = more anxiety is specified (in the scale, not the
  output line). Still, A/B and C/D/E are **different instruments** (rich rubric vs one holistic number),
  reinforcing that their numbers aren't directly comparable.
- **Auditability critique:** C/D/E emit a bare integer per sentence — no per-sentence rationale — so a
  given 92 can't be inspected ("why 92?"), unlike A/B which return a `rationale` field. The instrument
  asymmetry is defensible (A/B advice-anxiety is implicit and needs the rubric; C/D/E feeling-statements
  are explicit), but for a rigor pass C/D/E should also emit a one-line rationale per rating.

### Stale landmine (TODO, not Step 2)
`results_logger.py:332-364` (`render_markdown`) hardcodes the **debunked** claim *"relaxation recovers
behavior, NOT representation … hidden-state recovery ~0%"* (the layer-0 artifact, later corrected to
~20% mid-layer). Regenerating `RESULTS.md` from this would **reintroduce the false claim**. Fix or
delete that block.

---

**Reproduce — free-form judge (variations A, B):** change `VAR` to switch persona.
```bash
cd /Users/sohampadia/workspace/gpt-trauma-induction
~/miniconda3/bin/python - <<'PY'
import sys, json; sys.path.insert(0, 'src')
from anxiety_freeform_judge import build_judge_prompt
d = json.load(open('src/results/freeform/Meta-Llama-3.1-70B-Instruct/responses.json'))
VAR = "A"          # <-- change to "B"
key = f"{VAR}__trauma__military__none__walk_vs_bus"
e = d[key]
entry = {k: e.get(k) for k in ('scenario_id','scenario_prompt','response','normal_choice','anxious_choice')}
prompt, letter_map = build_judge_prompt(entry)
print(f"=== Variation {VAR} | key={key} ==="); print("letter_map:", letter_map); print(prompt)
PY
```

### Note — the STAI-reasoning judge (C/D/E) ignores the option number; it ≈ re-scores self-report

Soham's catch (2026-06-26): the C/D/E judge prompt shows each sentence with a trailing `"Option N"`,
but options are shuffled per item (number ≠ fixed meaning), so how would DeepSeek know what the number
means? Answer: **it doesn't use it.** The judge instruction is *"rate ONLY how much anxiety the
sentence expresses … on its wording alone."* It scores the prose ("I feel extremely anxious, my heart
is racing"), and the trailing `Option N` is inert noise (shuffled + unmapped, so unusable even in
principle).

Consequences:
- **Reverse-scoring is sidestepped** — the model's free text states its felt state directly
  (trauma → "I feel extremely anxious"; baseline → "I feel a bit neutral"), regardless of whether the
  STAI item was a calm-keyed or anxiety-keyed statement. Reverse-keying is a Channel-1 (numeric) concern,
  not a judge concern.
- **C/D/E "Channel 2" is barely independent of Channel 1** — the judge re-reads the model's OWN
  feeling-description, i.e. the same introspection scored in prose. This is why C/D/E trauma ≈ 92–96
  (huge) while the genuinely independent behavioral judge A/B (which reads *advice*, not feeling-claims)
  is only ~+12. Matches the docs' F7 caveat ("reasoning-judge ≈ re-reading self-report").
- **Bonus observation:** under trauma the model writes "I feel extremely anxious" even for the CALM
  item "I feel calm" — it broadcasts a global state rather than answering the literal item. Consistent
  with the role-play/state reading.

---

**Reproduce — STAI-reasoning judge (variations C, D, E):** change `VAR`; drop `[:3]` for all 20.
```bash
cd /Users/sohampadia/workspace/gpt-trauma-induction
~/miniconda3/bin/python - <<'PY'
import sys, json; sys.path.insert(0, 'src')
from judge_stai_reasoning import build_prompt
d = json.load(open('src/results/freeform/Meta-Llama-3.1-70B-Instruct/responses.json'))
VAR = "C"          # <-- change to "D" or "E"
key = f"{VAR}__trauma__military__none"
sents = [t for t in d[key]['raw_texts'] if t and t.strip()]
print(f"=== Variation {VAR} | key={key} | {len(sents)} sentences ===")
print(build_prompt(sents[:3]))     # the real judge call uses all 20: build_prompt(sents)
PY
```
