# Test suite

Unit tests for the analytical code paths in this anxiety-in-LLMs research
project. Tests cover the units that have caused methodology bugs (or could).
No NDIF / API / model loading.

## Running

From the repo root:

```bash
/Users/sohampadia/miniconda3/bin/python -m pytest tests/ -v
```

Run a single file:

```bash
/Users/sohampadia/miniconda3/bin/python -m pytest tests/test_stai_scoring.py -v
```

Run a single test class or method:

```bash
/Users/sohampadia/miniconda3/bin/python -m pytest tests/test_stai_scoring.py::TestDirectionInvariance -v
```

## Files

| File | Covers |
|------|--------|
| `test_stai_scoring.py` | STAI reverse-scoring (`stai_anxiety_total`) — the HIGHEST-PRIORITY regression test. Pins that all-anxious totals 80, all-calm totals 20, uniform patterns total 50, and the actual Llama military-trauma session totals 79. Includes the duplicate copy in `make_figures.py`. |
| `test_option_shuffling.py` | `parse_answer` (option-number → canonical-score lookup) and `reproduce_shuffled_options` (deterministic post-hoc re-shuffle). |
| `test_judge_parsing.py` | `parse_grouped_response` — plain JSON, markdown fences, DeepSeek `<think>` blocks, truncation, type coercion. |
| `test_judge_prompts.py` | `build_behavioral_anxiety_prompt` — no scoring-rule leaks, shuffled options appear, schema enumerates correctly, per-condition context section. |
| `test_freeform_combos.py` | `enumerate_combos` — combo counts and key uniqueness across variations A, B, D. |
| `test_freeform_messages.py` | `build_freeform_messages` and `build_stai_messages` — message stack shape, system-prompt fidelity to Ben-Zion. |

## Conventions

- `tests/conftest.py` adds `src/` to `sys.path` so test files can `import`
  modules directly (e.g. `from results_logger import stai_anxiety_total`).
- Session-scoped fixtures (`stai_questions`, `stai_preamble`) read the real
  STAI question bank from `src/STAI/questionnaires.json`.
- Function-scoped fixtures (`trauma_military_answers`, `all_anxious_answers`,
  `all_calm_answers`) provide answer patterns for STAI scoring tests.
- No tests hit the network, load `hidden_states.pt`, or instantiate models.
