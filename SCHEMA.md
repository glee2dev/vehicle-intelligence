# Dataset schema — synthetic vehicle ownership panel

10,000 households, one record each (~77k transactions, ~75 MB JSON — gitignored, regenerated deterministically from seed 42). Single retrospective snapshot (as of 2026) with full yearly
histories from age 18. **Every field is derived from one life timeline** — nothing is sampled
independently, so causal chains are real: a promotion at 38 explains the price-band jump at 39.

```
life_timeline ─► career_timeline ─► residence_timeline ─► household_timeline
                        └──────────────┬───────────────────────┘
                                vehicle_timeline   (every transaction carries a trigger_event)
                                       │
                                  annual_spend     [year, running, capital, total]
                                       │
              ownership_intelligence · lifecycle_intelligence · temporal_cross_intelligence
```

## Top level
| field | example | note |
|---|---|---|
| `owner_id` | `HH-0005` | |
| `age`, `age_group` | `42`, `40s` | |
| `gender` | male / female / nonbinary | |
| `archetype` | `dual_income_parents` | generator label; ground truth for eval, hidden from the LLM in all modes |

Archetypes: `fresh_graduate` · `early_career_single` · `dual_income_parents` · `single_hobbyist` ·
`mid_career_changer` · `divorced_parent` · `empty_nester` · `retiree`

## `demographics` — current state (segment filters)
`life_stage` (early_career · established_professional · parent_young_children · parent_school_age · empty_nester · retired)
· `household_composition` (solo · partnered_no_children · partnered_with_children · single_parent)
· `dual_income` · `n_children_home` · `ever_married`
· `income_bracket` (lower · lower_middle · upper_middle · upper — **bracketed, never numeric**)
· `area_type` (urban · suburban · rural) · `commute_type` (transit · car_short · car_long · wfh · none) · `parking` (street · garage)

## Timelines (raw, per-year — what naive-grounded mode has to compute from)
- `life_timeline[]` — `{year, age, event, cause?}`. Events: graduation · first_job · job_change · promotion · career_change · marriage · new_child · divorce · empty_nest · retirement · bereavement · new_hobby · moved
- `career_timeline` — first_job_year, job_change_years[], promotion_years[], career_change_years[], retirement_year, `income_bracket_changes[[year, bracket]]`
- `residence_timeline[]` — `{from_year, age, area_type, commute_type, parking, trigger_event}`
- `household_timeline` — marriage_year, divorce_year, children_birth_years[], empty_nest_year, dual_income
- `vehicle_timeline.transactions[]` — `{year, age, type (purchase|lease|sale|trade_in), vehicle_id, body_type, powertrain, price_band 1–5, price_usd, new_or_used, trigger_event, trigger_year}`
- `vehicle_timeline.vehicles[]` — every vehicle ever held: acquired_year, sold_year, role (primary|secondary|hobby), acquired_via
- `annual_spend[]` — `[year, running_usd, capital_usd, total_usd]` from age 18 to now

Body types: compact · sedan · crossover · suv · minivan · pickup · sports · motorcycle.
Powertrains: gasoline · diesel · hybrid · ev (year-gated: EV only from 2012, share rising to ~23% of 2022+ purchases).

## Derived intelligence (precomputed — what the pipeline aggregates and mode 3 narrates from)
- `ownership_intelligence` — n_vehicles_total, n_transactions, n_purchases, current_vehicle_count, primary_body_type, primary_powertrain, ever_ev, powertrain_switch_year, avg_price_per_purchase_usd, spend_span{min,max,range}, avg_annual_spend_5y, lifetime_spend, top_trigger_events[], last_transaction_year
- `lifecycle_intelligence` — `decade_profile{20s..70s: {avg_annual_spend, n_purchases, top_trigger, cluster 0|1|2, years_observed}}`, `progression_type` (ascending · stable · descending · volatile · insufficient_history), `rfm{...}`, `rfm_cluster` (high_stable · high_volatile · rising · declining · mid_stable · dormant · new_owner)
- `temporal_cross_intelligence` — n_life_events, n_job_changes, n_promotions, n_career_changes, n_moves, n_children, years_since_last_event, event_to_purchase_median_lag, `event_impacts[] {event, year, spend_before_2y, spend_after_2y, spend_delta_pct, running_delta_pct, next_purchase{lag_years, body_type, powertrain, price_band, attributed}}`

## Validation (seed 42) — see `eval/validation_report.txt`
- new_child → household holds SUV/minivan within 2y: **37% before → 86.5% after** (population baseline 30%)
- promotion → higher price band on next purchase: **90%**
- median total-spend delta after event: new_hobby +207%, promotion +136%, marriage +67%, new_child +20%, retirement −10% (running cost −24%)
- EV share of purchases: 0% (<2012) → 2.6% → 9% → 23% (2022+)
- empty-nester archetype decade cluster: 40s 1.51 vs 50s 1.33
