# Aggregate across models and repeats

8 runs, 6 model configurations, ten questions each, same three modes, same prompts. Grounding rate = (verified + derived) / all numeric claims, pooled over the run's claims. A model with repeats shows the pooled figure with the per-run range.

## Pooled over everything

| | ungrounded | naive-grounded | pipeline-grounded |
|---|---:|---:|---:|
| claims checked | 206 | 928 | 1474 |
| numbers verified | 52.9% | 59.3% | 99.8% |
| questions fully grounded | 33/80 | 6/80 | 77/80 |
| empty or truncated answers | 0 | 21 | 0 |
| external-knowledge phrases | 29 | 6 | 1 |
| refused the unanswerable | 7/16 | 10/16 | 16/16 |
| fabricated on unanswerable | 26 | 23 | 0 |
| input tokens per question | 531 | 244,930 | 975 |

## Per model

| model | runs | ungrounded | naive-grounded | pipeline-grounded | naive empty | refused unanswerable (naive / pipeline) |
|---|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 5 (adaptive thinking) | 3 | 40.2% (33.3%–50.0%) | 64.1% (52.6%–79.3%) | 99.5% (99.5%–99.5%) | 12/30 | 6/6 / 6/6 |
| Gemini 3.1 Pro | 1 | 68.0% | 60.2% | 100.0% | 0/10 | 1/2 / 2/2 |
| GPT-5.4 (reasoning=medium) | 1 | 77.8% | 54.0% | 100.0% | 0/10 | 1/2 / 2/2 |
| GPT-5.4 (provider default: no reasoning) | 1 | 57.7% | 55.9% | 100.0% | 0/10 | 1/2 / 2/2 |
| Qwen 3.8 2.4T-A95B | 1 | 71.4% | 92.3% | 100.0% | 9/10 | 0/2 / 2/2 |
| Grok 4.6 | 1 | 42.9% | 50.0% | 100.0% | 0/10 | 1/2 / 2/2 |

Grounding rate counts claims in answers that were produced. A model whose naive-grounded answers came back empty (it spent the 20k-token budget thinking) has few claims to check there; read the naive column next to the empty column.

## Every run

| run | ungrounded | naive-grounded | pipeline-grounded |
|---|---:|---:|---:|
| `anthropic_sonnet5_fleet_analyst_v1.scored.json` | 13/39 | 92/144 | 218/219 |
| `anthropic_sonnet5_fleet_analyst_v2.scored.json` | 17/34 | 60/114 | 191/192 |
| `anthropic_sonnet5_fleet_analyst_v3.scored.json` | 13/34 | 69/87 | 194/195 |
| `openai_google-gemini-3-1-pro-preview_fleet_analyst_v1.scored.json` | 17/25 | 77/128 | 155/155 |
| `openai_openai-gpt-5-4-reasoning-medium_fleet_analyst_v1.scored.json` | 21/27 | 74/137 | 208/208 |
| `openai_openai-gpt-5-4_fleet_analyst_v1.scored.json` | 15/26 | 128/229 | 219/219 |
| `openai_qwen-qwen3-8-2-4t-a95b_fleet_analyst_v1.scored.json` | 10/14 | 12/13 | 169/169 |
| `openai_x-ai-grok-4-6_fleet_analyst_v1.scored.json` | 3/7 | 38/76 | 117/117 |

Rerun with `python eval/aggregate.py` after scoring a new run file.
