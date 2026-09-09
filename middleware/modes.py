"""
Three ways of answering the same question with the same model.

  ungrounded        schema only, no records, no stats     -> "what most people ship"
  naive_grounded    raw records in context, LLM does math -> "what people think grounding means"
  pipeline_grounded aggregated stats only, LLM narrates   -> this architecture

The system prompts differ because the *task* differs (compute vs narrate). Model and
temperature are identical across modes. Nothing here nudges a mode toward failure: mode 1 is
told plainly what it does and doesn't have; mode 2 is told the sample size and is asked to show
its work; mode 3 is told to say "not in the data" when the pipeline says so.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

MODES = ("ungrounded", "naive_grounded", "pipeline_grounded")

# What each mode has in common
_TASK = ("You are an analyst answering questions about a synthetic vehicle-ownership survey of "
         "10,000 U.S. households, one record per household, with yearly histories from age 18 to 2026. "
         "Answer in under 250 words. Use specific numbers where you can; state the denominator for "
         "any percentage (households, purchases, or events).")

SCHEMA_SUMMARY = """Fields per household:
- age, age_group (20s..70s), gender
- demographics: life_stage (early_career | established_professional | parent_young_children | parent_school_age | empty_nester | retired),
  household_composition (solo | partnered_no_children | partnered_with_children | single_parent), dual_income, n_children_home,
  ever_married, income_bracket (lower | lower_middle | upper_middle | upper — categorical only), area_type, commute_type, parking
- life_timeline[]: {year, age, event} for graduation, first_job, job_change, promotion, career_change, marriage, new_child,
  divorce, empty_nest, retirement, bereavement, new_hobby, moved
- career_timeline: job_change_years[], promotion_years[], career_change_years[], retirement_year, income_bracket_changes[[year, bracket]]
- residence_timeline[]: {from_year, area_type, commute_type, parking, trigger_event}
- vehicle_timeline.transactions[]: {year, type (purchase|lease|sale|trade_in), body_type, powertrain (gasoline|diesel|hybrid|ev),
  price_band 1-5, price_usd, new_or_used, trigger_event}
- annual_spend[]: [year, running_usd, capital_usd, total_usd]"""


@dataclass
class ModeContext:
    mode: str
    system: str
    user: str
    context_chars: int
    context_meta: dict


# ── Mode 1: ungrounded ────────────────────────────────────────────────────────

def build_ungrounded(question: str) -> ModeContext:
    system = (_TASK + "\n\nYou have the survey's schema below but NOT its records or any computed statistics. "
              "Answer the question as helpfully as you can.")
    user = f"Schema:\n{SCHEMA_SUMMARY}\n\nQuestion: {question}"
    return ModeContext("ungrounded", system, user, len(system) + len(user), {"records": 0, "stats": False})


# ── Mode 2: naive grounded ────────────────────────────────────────────────────

RAW_KEYS = ("owner_id", "age", "age_group", "gender", "demographics", "life_timeline", "career_timeline",
            "residence_timeline", "household_timeline", "vehicle_timeline", "annual_spend")


def _slim(rec: dict) -> dict:
    """Raw record: timelines only. The three derived-intelligence blocks are dropped — that is the
    naive approach by definition: hand the model the data, let it compute."""
    out = {k: rec[k] for k in RAW_KEYS if k in rec}
    vt = out.get("vehicle_timeline", {})
    if vt:
        out["vehicle_timeline"] = {"transactions": [
            {k: t[k] for k in ("year", "type", "body_type", "powertrain", "price_band", "price_usd", "new_or_used", "trigger_event")}
            for t in vt["transactions"]]}
    return out


def build_naive_grounded(question: str, records_payload: dict, char_budget: int = 220_000) -> ModeContext:
    """records_payload is the pipeline /records response. Records are packed until char_budget."""
    packed, chars, n_used, ids = [], 0, 0, []
    for r in records_payload["records"]:
        s = json.dumps(_slim(r), separators=(",", ":"))
        if chars + len(s) > char_budget:
            break
        packed.append(s); chars += len(s); n_used += 1; ids.append(r["owner_id"])
    n_matched = records_payload["n_matched"]
    sample_note = (f"You are given {n_used} of {n_matched:,} households that match the question's segment "
                   f"(the segment filter was applied deterministically before sampling; this is a uniform random sample). "
                   if n_used < n_matched else
                   f"You are given all {n_used} households that match the question's segment. ")
    system = (_TASK + "\n\nYou are given raw household records as JSON. Answer STRICTLY from these records: "
              "compute every number yourself from the data below, show the denominator you used, and do not "
              "use outside knowledge. If the records genuinely cannot answer the question, say so.")
    user = (f"{sample_note}Total survey size is {records_payload['n_total']:,}.\n\n"
            f"Schema:\n{SCHEMA_SUMMARY}\n\nRecords (one JSON object per line):\n" + "\n".join(packed) +
            f"\n\nQuestion: {question}")
    return ModeContext("naive_grounded", system, user, len(system) + len(user),
                       {"records": n_used, "n_matched": n_matched, "truncated": n_used < n_matched, "stats": False,
                        "owner_ids": ids})


# ── Mode 3: pipeline grounded ─────────────────────────────────────────────────

PERSONAS = {
    "fleet_analyst": {
        "name": "Fleet Analyst", "temperature": 0.1,
        "style": ("Factual and statistical. Lead with segment size and the denominator. Present the key figures, "
                  "then two or three observations that each cite a specific number. No recommendations."),
    },
    "market_strategist": {
        "name": "Market Strategist", "temperature": 0.3,
        "style": ("Opportunity-focused. Open with segment size and why it matters, name two or three opportunities "
                  "each backed by a specific figure, and close with what data would strengthen the case. "
                  "Label every inference as an inference."),
    },
}

GROUNDING_RULES = """STRICT RULES — no exceptions:
1. Use ONLY numbers present in the statistics JSON. Do not compute new numbers except trivial differences between two given figures, and say so when you do.
2. Do not add external knowledge, benchmarks, industry context, or typical behaviour. Never write "typically", "generally", "research shows", or "on average consumers". Say "in this dataset".
3. Every percentage you state must name its denominator (households, purchases, or events) as given in the JSON.
4. If the JSON says answerable=false, say the question cannot be answered from this dataset, give the reason verbatim, and then report only what answerable_part contains.
5. If a figure needed for the question is absent from the JSON, say it is not available. Do not estimate."""


def build_pipeline_grounded(question: str, stats: dict, persona: str = "fleet_analyst") -> ModeContext:
    p = PERSONAS[persona]
    system = f"{_TASK}\n\nRole: {p['name']}. {p['style']}\n\n{GROUNDING_RULES}"
    slim_stats = {k: v for k, v in stats.items() if k not in ("processing_ms", "cached")}
    user = (f"Question: {question}\n\nStatistics JSON (the only source you may use):\n"
            f"{json.dumps(slim_stats, indent=1)}\n\nAnswer the question from the JSON above.")
    return ModeContext("pipeline_grounded", system, user, len(system) + len(user),
                       {"records": 0, "stats": True, "persona": persona})
