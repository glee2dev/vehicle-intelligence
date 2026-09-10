# Running the experiment on other models

The claim is about the architecture, not about one model. Every run below uses the same ten
questions, the same three modes, and the same prompts; only `LLM_MODEL` changes.

## Anthropic (direct)

```bash
LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... LLM_MODEL=claude-sonnet-5 \
  python eval/run_modes.py --concurrency 3
```

## OpenRouter (any other lab)

```bash
export LLM_BACKEND=openai OPENROUTER_API_KEY=sk-or-...
LLM_MODEL=openai/gpt-5           python eval/run_modes.py --concurrency 3
LLM_MODEL=google/gemini-2.5-pro  python eval/run_modes.py --concurrency 3
LLM_MODEL=x-ai/grok-4            python eval/run_modes.py --concurrency 3
LLM_MODEL=qwen/qwen3-235b-a22b   python eval/run_modes.py --concurrency 3
```

Model ids are whatever OpenRouter lists at the time; check https://openrouter.ai/models.
Options: `LLM_REASONING=high|medium|low|none` sets reasoning effort where the model supports it
(unset = provider default). `LLM_TEMPERATURE=0.1` opts in to a temperature — many reasoning
models reject it, so it is off unless you set it. `LLM_MAX_TOKENS` (default 20000) caps output
including reasoning; naive-grounded needs the room.

Cost per full run is dominated by naive-grounded: ~285k input tokens per question. Budget
roughly ten times the model's price per million input tokens times three.

## Repeats

Run the same command three times; each run gets its own timestamped file. Runs save
incrementally (`.partial.jsonl`) and resume if interrupted.

## Scoring and the site

```bash
python eval/score_run.py eval/runs/<run>.json        # scoreboard + <run>.scored.json
python eval/build_replay.py                           # site bundle (currently reads one analyst run)
python site/build.py
```

Multi-model aggregation across runs (`eval/aggregate.py`) is the next piece; until it lands,
`build_replay.py --analyst <file>` picks which run the site shows.
