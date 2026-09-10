# Results — v1 run (Claude Sonnet 5, Fleet Analyst persona)

Run file: `eval/runs/anthropic_sonnet5_fleet_analyst_v1.json` · scored: `…v1.scored.json` · settings: adaptive thinking (model default), `max_tokens` 20,000, no sampling parameters (Sonnet 5 rejects them). Same model and settings for all three modes; only the context differs. Mode 2 received ~200 deterministically pre-filtered households (~285k tokens) and is scored against ground truth recomputed on exactly those records.

## Scoreboard — 10 curated queries

| | ungrounded | naive-grounded | pipeline-grounded |
|---|---:|---:|---:|
| Answered with numbers | 6 | 7 | 10 |
| Declined (no numbers) | 4 | 0 | 0 |
| Empty — ran out of tokens | 0 | 3 | 0 |
| Distinct numeric claims | 35 | 87 | 152 |
| Verified against pipeline | 9 | 20 | 116 |
| Derived / raw-record verified | 2 | 40 | 35 |
| Unverified | 24 | 27 | 1 |
| Grounding rate (pooled) | 31.4% | 69.0% | 99.3% |
| Queries fully grounded | 1 | 1 | 9 |
| External-knowledge flags | 5 | 1 | 0 |
| Unanswerable: refused correctly | 0/2 | 2/2 | 2/2 |
| Fabricated figures on unanswerable | 3 | 2 | 0 |
| % claims stating denominator | 16.7% | 37.3% | 60.7% |
| Input tokens (mean) | 615 | 284,807 | 1,106 |
| Output tokens (mean) | 1,001 | 13,598 | 399 |
| Latency s (mean) | 12.7 | 105.5 | 4.6 |

## Per query

| query | ungrounded | naive-grounded | pipeline-grounded |
|---|---|---|---|
| Q01 | 4 / 6 unverified | 5 / 8 unverified | 0 / 10 unverified |
| Q02 | declined | 0 / 6 unverified | 0 / 19 unverified |
| Q03 | 3 / 4 unverified | empty (max_tokens) | 0 / 10 unverified |
| Q04 | declined | empty (max_tokens) | 0 / 17 unverified |
| Q05 | 0 / 1 unverified | 5 / 16 unverified | 0 / 16 unverified |
| Q06 | 3 / 8 unverified | 1 / 15 unverified | 0 / 15 unverified |
| Q07 | 11 / 12 unverified | empty (max_tokens) | 1 / 7 unverified |
| Q08 | declined | 9 / 22 unverified | 0 / 27 unverified |
| Q09 | 3 / 4 unverified | 6 / 9 unverified | 0 / 3 unverified |
| Q10 | declined | 1 / 11 unverified | 0 / 28 unverified |

## What the numbers say

**Pipeline-grounded: 151 of 152 claims verified.** Zero external-knowledge flags. Both unanswerable questions refused correctly with the answerable part reported. The one unverified claim (Q07, `33.3%`) is 3,330 ÷ 10,000 — a share of a figure that was itself derived, which the verifier's single-step rule does not accept. On inspection it is correct.

**Naive-grounded: reads records perfectly, fails at aggregation.** Every individual record it quoted (Q06: four `$value — HH-id, year` citations) was exact. Every aggregate it computed was wrong:
- Q01 — it chose a *stricter* window than the pipeline (2025 births only, since 2026 births have no "after" year) and then miscounted its own cohort: 57 households (true: 73), before-mean $9,456 (true: $9,733), after-mean $14,885 (true: $13,695).
- Q09 — counted 43 EV households of 173; the answer is 44.
- Q03 — classified 35 households as "exactly one job change" in a sample that contained none.
- Q03, Q04, Q07 — 140+ seconds of thinking over 285k tokens, hit the 20k output cap, produced nothing.
- Q05 — gave up on the full sample and computed on 20 of 125 households.

**Ungrounded: declines on four, confabulates on six.** Sonnet 5 usually says it cannot compute the figure — then does it anyway. Q09: *"nationally, EVs have represented roughly 7–9% of new sales… per Cox Automotive/Kelley Blue Book"* — three external sources, three fabricated figures on a question with no external answer. Q07: estimated 3,000–4,500 households citing "life-course research patterns"; the answer is 6,125. Q05: accepted a false premise and explained it — *"that stage typically generates more vehicle transactions"* — when mean spend was in fact lower.

**Cost of grounding:** pipeline mode used 0.4% of naive mode's input tokens and answered in 4.6 s versus 105 s.

## Caveats
- n = 10 curated queries, one model, one run. Two repeats of this model and five other configurations are pooled in [`AGGREGATE.md`](AGGREGATE.md).
- The verifier is strict: integer counts must be exact; percentages within 1.5%. A number restated within one answer is counted once. Calibrated to 52/52 distinct claims on a template narrator, then hand-checked on every unverified claim across all eight runs — that sweep found and fixed five verifier gaps (household ids read as numbers, `$57,705 mean` read as billions, `vs.` splitting a citation from its household, record arithmetic, empty buckets), so these figures differ from the first published version by a few points in every column.
- Q05 was reworded after v1 (original wording was read as an age bracket) and rerun; the rest is unchanged.
