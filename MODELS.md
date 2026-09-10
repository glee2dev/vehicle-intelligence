# Running the experiment on other models

The claim is about the architecture, not about one model. Every run below uses the same ten
questions, the same three modes, and the same prompts; only `LLM_MODEL` changes. Results for
all of them: [`eval/AGGREGATE.md`](eval/AGGREGATE.md).

## Anthropic (direct)

```bash
LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... LLM_MODEL=claude-sonnet-5 \
  python eval/run_modes.py --concurrency 3
```

## OpenRouter (any lab, OpenAI-compatible)

```bash
export LLM_BACKEND=openai OPENROUTER_API_KEY=sk-or-...
LLM_MODEL=google/gemini-3.1-pro-preview  python eval/run_modes.py --concurrency 3
LLM_MODEL=x-ai/grok-4.6                  python eval/run_modes.py --concurrency 3
LLM_MODEL=openai/gpt-5.4                 python eval/run_modes.py --concurrency 3
LLM_MODEL=qwen/qwen3.8-2.4t-a95b         python eval/run_modes.py --concurrency 3
```

Model ids are whatever OpenRouter lists at the time (`curl https://openrouter.ai/api/v1/models`);
the ids above are the ones the committed runs used, September 2026.

Options: `LLM_REASONING=high|medium|low|none` sets reasoning effort where the model supports it
(unset = provider default). Provider default is not the same across labs: Gemini, Grok and Qwen
reason by default; **GPT-5.4 does not** (`reasoning_tokens: 0`), so it is committed twice, once at
provider default and once at `LLM_REASONING=medium`. `LLM_TEMPERATURE=0.1` opts in to a temperature —
many reasoning models reject it, so it is off unless you set it. `LLM_MAX_TOKENS` (default 20000)
caps output including reasoning; naive-grounded needs the room and some models (Sonnet on 3–5
questions, Qwen on 9) spend all of it thinking and return nothing. That is a scored outcome
(`empty_truncated`), not an error.

## Claude repeats through OpenRouter

The Anthropic backend accepts a base URL, so repeats can go through OpenRouter's Anthropic-compatible
endpoint with adaptive thinking intact (BYOK if you have it set up there):

```bash
LLM_BACKEND=anthropic ANTHROPIC_BASE_URL=https://openrouter.ai/api ANTHROPIC_API_KEY=sk-or-... \
  LLM_MODEL=anthropic/claude-sonnet-5 python eval/run_modes.py --concurrency 3 \
  --out eval/runs/anthropic_sonnet5_fleet_analyst_v3.json
```

`v2` and `v3` were run this way; `v1` went direct. Settings and `thinking_blocks` match in the run files.

## Things that will bite

- **Naming and resume.** Pass `--out` explicitly so the file name says which model. Runs save
  incrementally to `<out>.partial.jsonl` and resume from it when rerun with the same `--out`;
  a question whose outputs carry an `error` is treated as finished, so strip those lines before
  resuming.
- **OpenRouter 402 `in_flight_budget_exhausted`.** OpenRouter reserves credit for every in-flight
  request at `max_tokens`, three modes at 20k each, so a run stops well before the balance hits
  zero. Keep ~$15 headroom per model.
- **ZDR.** If your account enforces zero-data-retention, models hosted only by their own lab
  (Qwen Max, Kimi) return 404 "0 endpoints available". Pick an open-weights id served by a
  ZDR provider — that is why the Qwen run is 3.8 2.4T-A95B via Together, not 3.8 Max.
- **Cost.** Naive-grounded dominates: ~200–285k input tokens per question depending on tokenizer.
  At $2/M input a full run is $10–15.

## Scoring and the aggregate

```bash
python eval/score_run.py eval/runs/<run>.json     # scoreboard + <run>.scored.json
python eval/aggregate.py                          # eval/aggregate.json + eval/AGGREGATE.md over every scored run
python eval/build_replay.py --analyst <file>      # which run the site replays (default: Sonnet v1)
python site/build.py
```
