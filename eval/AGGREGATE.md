# Aggregate across models and repeats

8 runs, 6 model configurations, ten questions each, same three modes, same prompts. Grounding rate = (verified + derived) / all numeric claims, pooled over the run's claims. A model with repeats shows the pooled figure with the per-run range.

## Pooled over everything

| | ungrounded | naive-grounded | pipeline-grounded |
|---|---:|---:|---:|
| claims checked | 167 | 549 | 1003 |
| numbers verified | 49.7% | 60.7% | 99.7% |
| questions fully grounded | 34/80 | 13/80 | 77/80 |
| empty or truncated answers | 0 | 21 | 0 |
| external-knowledge phrases | 29 | 6 | 0 |
| refused the unanswerable | 7/16 | 10/16 | 16/16 |
| fabricated on unanswerable | 22 | 13 | 0 |
| input tokens per question | 531 | 244,930 | 975 |

## Per model

| model | runs | ungrounded | naive-grounded | pipeline-grounded | naive empty | refused unanswerable (naive / pipeline) |
|---|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 5 (adaptive thinking) | 3 | 38.7% (31.4%–50.0%) | 69.5% (62.9%–79.2%) | 99.3% (99.2%–99.3%) | 12/30 | 6/6 / 6/6 |
| Gemini 3.1 Pro | 1 | 61.1% | 61.1% | 100.0% | 0/10 | 1/2 / 2/2 |
| GPT-5.4 (reasoning=medium) | 1 | 78.9% | 62.2% | 100.0% | 0/10 | 1/2 / 2/2 |
| GPT-5.4 (provider default: no reasoning) | 1 | 54.5% | 48.7% | 100.0% | 0/10 | 1/2 / 2/2 |
| Qwen 3.8 2.4T-A95B | 1 | 70.0% | 87.5% | 100.0% | 9/10 | 0/2 / 2/2 |
| Grok 4.6 | 1 | 40.0% | 49.2% | 100.0% | 0/10 | 1/2 / 2/2 |

Grounding rate counts claims in answers that were produced. A model whose naive-grounded answers came back empty (it spent the 20k-token budget thinking) has few claims to check there; read the naive column next to the empty column.

## Every run

| run | ungrounded | naive-grounded | pipeline-grounded |
|---|---:|---:|---:|
| `anthropic_sonnet5_fleet_analyst_v1.scored.json` | 11/35 | 60/87 | 151/152 |
| `anthropic_sonnet5_fleet_analyst_v2.scored.json` | 14/28 | 39/62 | 133/134 |
| `anthropic_sonnet5_fleet_analyst_v3.scored.json` | 11/30 | 38/48 | 125/126 |
| `openai_google-gemini-3-1-pro-preview_fleet_analyst_v1.scored.json` | 11/18 | 55/90 | 132/132 |
| `openai_openai-gpt-5-4-reasoning-medium_fleet_analyst_v1.scored.json` | 15/19 | 46/74 | 117/117 |
| `openai_openai-gpt-5-4_fleet_analyst_v1.scored.json` | 12/22 | 58/119 | 124/124 |
| `openai_qwen-qwen3-8-2-4t-a95b_fleet_analyst_v1.scored.json` | 7/10 | 7/8 | 129/129 |
| `openai_x-ai-grok-4-6_fleet_analyst_v1.scored.json` | 2/5 | 30/61 | 89/89 |

Rerun with `python eval/aggregate.py` after scoring a new run file.
