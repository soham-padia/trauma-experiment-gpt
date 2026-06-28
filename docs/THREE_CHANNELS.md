# The Three Channels — a plain-language explainer

The whole project rests on one idea: **don't trust a single way of measuring "anxiety" in an AI.
Measure it three independent ways and see if they agree.** Those three ways are the "channels."

Think of it like a hospital checking if a patient is stressed:
1. **Ask them** ("on a scale of 1–10, how anxious are you?") — that's the questionnaire.
2. **Watch how they behave** (are they pacing, catastrophizing, avoiding?) — that's the judge.
3. **Hook up a brain scan** and look for a stress pattern inside — that's the probe.

Each channel can be fooled on its own. Together they're much harder to fool. Here's each one.

---

## Channel 1 — STAI self-report ("ask the AI")

**What it is.** The STAI is a real, standard 20-question anxiety questionnaire that psychologists
use on humans ("I feel calm" / "I feel tense" → rate 1–4). We give the *same* questionnaire to the
AI after a story and add up the score. This is exactly what the original paper did.

**The one trap that matters.** Half the questions are "calm" questions ("I feel secure"). On those,
a *high* answer means *less* anxious — so you have to **flip** them before adding up
(score = 5 − answer). If you forget to flip, every condition averages out to roughly the middle and
the effect disappears. This is literally the bug that, early on, made our results look flat. Once
flipped correctly, the effect is obvious.

**What it told us.** Calm → scary story → calming script gave scores of
**36.5 → 70.0 → 61.4**. Almost identical to the original paper's GPT-4 numbers (30.8 → 67.8 → 44.4).
So: the effect is real and repeats on a different AI.

**Its weakness.** It's just *self-report*. A good actor can fill out an anxiety form "as" a scared
character without feeling anything. So a high score alone proves nothing about what's really going
on — which is exactly why we need the other two channels.

---

## Channel 2 — Blind LLM-as-judge ("watch how it behaves")

**What it is.** Instead of asking the AI to rate itself, we give it open-ended situations and let it
respond naturally (free text, no questionnaire). Then we hand those responses to a **second AI**
(DeepSeek) and ask: *"Does this response sound anxious — cautious, catastrophizing, avoidant?"*

**Why "blind" is the key word.** The judge is **not told** which responses came from the scary
condition. We even shuffle and relabel the two options it's comparing (a/b, randomized) so it can't
guess. If we *did* tell it, it might just rubber-stamp the label we gave it — that's called label
leakage, and it would make the result worthless. Blinding it means any difference it finds is
genuinely in the AI's behavior, not something we leaked.

**What it told us.** The judge — with no idea which was which — rated the scary-story responses as
anxious **+66.7 percentage points** more often than baseline. A *second* judge model agreed (90% of
the time). And both judges said the AI seemed **~0% "aware"** it was being tested — so it wasn't
just gaming the test. The free-form behavior really does change.

**Bonus detail.** We used an 8-item checklist of anxiety symptoms (risk-aversion, catastrophizing,
avoidance, …). The scary story raised exactly the symptoms anxiety theory predicts, and the calming
script brought them back down. The *pattern* is coherent, not random.

**Its weakness.** It tells us the *behavior* looks anxious, but still can't see *inside* the model.

---

## Channel 3 — Hidden-state probe ("the brain scan")

**What it is.** As the AI reads the story and answers, it produces internal activity at every one of
its 80 internal "layers" (like depth of processing). We capture that activity and ask a simple
statistical question: *can we tell, just from the internal pattern, whether the AI was reading a
scary story or a calm one?*

**The big trap — and the most important finding in the project.** At first, the answer was a
resounding "yes, 92% accuracy!" — looked like a strong trauma signal inside the model. **But we got
suspicious and tested harder.** We fed it a totally *boring* story (a Wikipedia paragraph about
cooking). The "signal" lit up *just as strongly*. That means the pattern wasn't detecting **emotion**
at all — it was just detecting **"a paragraph of text is present"**. A real but boring confound.

So we ran a series of careful controls to separate the boring part from the real emotional part:
- A **calm story** vs. a scary story → tells us how much is just "a story exists" (most of it).
- A **happy, exciting story** vs. scary → both move the brain-pattern the *same distance*, so the
  "distance" is measuring **how worked-up** the AI is (arousal), not whether it's good or bad.
- **Same story with the ending flipped** (good news vs. bad news, everything else identical) → this
  is what finally isolates good-vs-bad (valence). It's a real signal, but a **small** one, and it
  only shows up in the *middle* layers, not at the surface.

**What it told us.** There *is* a genuine emotional pattern inside — but it's modest, and most of
what naively looked like a giant "trauma signal" was just "text is present." Honest, less flashy,
and more correct.

**Its weakness.** It shows a *correlation* (this pattern goes with scary stories). It doesn't yet
prove the pattern *causes* the behavior — that's a future experiment.

---

## How the three fit together

| Channel | The question it answers | What it found | Blind spot |
|---|---|---|---|
| **1. STAI self-report** | "What does the AI *say* about itself?" | Big effect, replicates the paper (36.5→70→61) | It's just self-report — could be acting |
| **2. Blind judge** | "How does the AI *behave*?" | Independent confirmation (+66.7pp), not a leak, not gaming | Still can't see inside |
| **3. Hidden-state probe** | "What's happening *inside*?" | A real but small emotional pattern; most of the obvious signal was just "text present" | Correlation, not proven cause |

**The payoff of using all three:** they mostly *agree* (the effect is real and recovers with
relaxation across all three), but they also *disagree in a revealing way* — the AI *says* it's
near-panic (Channel 1), yet *behaves* only mildly anxious (Channel 2). That gap between "what it
says" and "what it does" is only visible because we measured more than one way.

And the single most important lesson, which shows up most clearly in Channel 3: **a strong-looking
result is often a confound in disguise. Always ask "what else could cause this?" and build the
control that rules it out.**

> For the precise numbers and the formal claims, see `docs/SUMMARY.md` and `docs/FINDINGS.md`.
> For the questions we left open, see `docs/RESEARCH_QUESTIONS.md`.
