# Who did the math?

**When an LLM does the analysis, it invents the numbers. When it only narrates numbers you computed, it doesn't.** This repo is the measurement.

Ten questions about a synthetic 10,000-household vehicle-ownership panel, each answered three ways by the same model with the same settings. The only thing that changes is what the model is given. A verifier then checks every number in every answer against ground truth.

| | ungrounded | naive-grounded | pipeline-grounded |
|---|---:|---:|---:|
| what the model receives | schema only | ~200 raw records, pre-filtered | aggregated statistics |
| numbers verified | 50% | 61% | **99.7%** |
| questions where every number matches | 34 / 80 | 13 / 80 | 77 / 80 |
| unanswerable questions refused | 7 / 16 | 10 / 16 | 16 / 16 |
| figures invented on unanswerable questions | 22 | 13 | 0 |
| input tokens per question | 531 | 244,930 | 975 |

Pooled over eight runs and six model configurations — Claude Sonnet 5 ×3 (adaptive thinking, no sampling parameters), Gemini 3.1 Pro, GPT-5.4 with and without reasoning, Grok 4.6, Qwen 3.8 — all through the same runner, same prompts, same questions. Per-model and per-run breakdown in [`eval/AGGREGATE.md`](eval/AGGREGATE.md). One run in full detail, claim by claim, in [`eval/RESULTS.md`](eval/RESULTS.md); every prompt and answer is committed under [`eval/runs/`](eval/runs/). The interactive version is the [site](https://glee2dev.github.io/vehicle-intelligence/).

## The findings

**Reading records is fine; arithmetic across them is not.** Every individual household the naive-grounded model quoted was exact. Every aggregate it computed was wrong — 57 households where there were 73, 43 where there were 44 — and on three questions it thought for two minutes over 285k tokens and produced nothing. Qwen 3.8 did that on nine of ten.

**The run decides the number; the mode decides whether it is true.** Sonnet 5 run three times moved between 62.9% and 79.2% naive-grounded — a wider spread than the one between five different labs, 48.7% to 62.2%. Pipeline-grounded did not move: 100% on every model but Sonnet, which lost the same two claims in all three runs.

**More thinking does not buy arithmetic.** GPT-5.4 at its provider default spends zero reasoning tokens and verified at 48.7%; at medium effort it took ten times as long and verified at 62.2%. Both were 100% pipeline-grounded.

**Ungrounded declines, then does it anyway.** "I can't report an exact figure… nationally EVs are roughly 7–9% per Kelley Blue Book." Three sources, three fabricated numbers, on a question with no external answer.

**The false premise is the realistic one.** Asked why empty nesters spent more in their 40s than their 50s — they didn't — the grounded answer rejected the premise with the numbers. The ungrounded answer explained a pattern that does not exist.

## Architecture

```
question ─► structured query ─► pipeline (filter · aggregate · denominators, or "not answerable")
                                      │
        ─────────── raw records never cross this line ───────────
                                      │
                                model narrates ─► verifier matches every number back
```

Two servers. The **pipeline** (`pipeline/`) owns the data and never calls a model. The **orchestrator** (`middleware/`) builds the three contexts, calls the model, and never touches raw data. The **verifier** (`eval/verifier.py`) extracts every numeric claim from an answer and matches it to the pipeline output: verified, derived (arithmetic on given figures, or a record read correctly), or unverified.

The pattern comes from analytics work on real customer panels. This is an independent build from scratch, on synthetic data, with the argument made measurable.

## The dataset is a set of lives

Every household is generated from one timeline (`tools/generate_dataset.py`): life events → career → residence → household → vehicle transactions, each carrying the event that triggered it → annual spend → derived intelligence. A promotion at 38 raises the income bracket, which raises the price band of the next purchase; a second child triggers the minivan. Households holding an SUV or minivan the year before a birth: 38%. Within two years after: 87%. Schema in [`SCHEMA.md`](SCHEMA.md).

## Fairness decisions

- Same model, same settings, same question. Only the context differs.
- Naive-grounded gets the *filtering* for free: the pipeline pre-selects the segment before sampling, so it is judged on the math alone. Two-cohort questions interleave records from both cohorts.
- Naive-grounded is scored against ground truth recomputed on the exact records it received, not the population. Record citations (`$94,844, HH-9032, 2015`) are checked against the record.
- The verifier is strict — integer counts exact, percentages within 1.5%, a number restated in one answer counted once — and was calibrated to 52/52 on a template narrator that prints pipeline values only, then hand-checked on every unverified claim across all eight runs.
- No mode sees the generator's archetype labels.

## Run it

```bash
pip install -r requirements.txt
python tools/generate_dataset.py            # 10,000 households, ~75 MB, deterministic (seed 42)
python -m pytest tests -q                   # 25 tests

# servers (two terminals)
PYTHONPATH=. python pipeline/server.py      # :8000  data, no LLM
LLM_BACKEND=mock python middleware/server.py # :3000  orchestration; mock narrates mode 3 from a template

# the experiment, in-process, real model
LLM_BACKEND=anthropic ANTHROPIC_API_KEY=... python eval/run_modes.py
python eval/score_run.py eval/runs/<run>.json
python eval/build_replay.py && python site/build.py   # -> docs/index.html
```

Mode 2 costs about a dollar and two minutes per question; the ten-question run is roughly $12. The site replays the recorded run and can rerun the two cheap modes live with your own key.

## Layout

```
tools/generate_dataset.py   timeline-first synthetic panel
pipeline/engine.py          filters, group_by, nine metrics with explicit denominators, unanswerable as a result
pipeline/server.py          /query /records /schema
middleware/modes.py         the three context builders
middleware/orchestrator.py  runs a question through the modes concurrently
middleware/llm_client.py    mock + Anthropic backends (httpx, async)
eval/queries.json           the ten questions and what each one exposes
eval/verifier.py            claim extraction, matching, external-knowledge and refusal detection
eval/score_run.py           scoreboard
eval/aggregate.py           pooled table across every scored run
eval/build_replay.py        bundle for the static site
site/                       single-page site; built to docs/
tests/                      engine, modes, verifier
```

## Caveats

Ten curated questions. The headline table is one run of one model; the aggregate is eight runs across five model families, still ten questions each. A synthetic panel has cleaner causal structure than a real one. The comparison is between modes, not between this data and yours.

—

Gyunpyo Lee · [glee2dev](https://github.com/glee2dev)
