#!/usr/bin/env python3
"""
Build site/replay.json: everything the static page needs, no backend.
  python eval/build_replay.py [--analyst eval/runs/...scored.json] [--strategist eval/runs/...scored.json]
"""
import argparse, json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "middleware"))
from verifier import score_output
from modes import PERSONAS, GROUNDING_RULES, SCHEMA_SUMMARY, _TASK

FAMILY = {"suv", "minivan"}


def slim_output(o: dict) -> dict:
    cm = {k: v for k, v in o.get("context_meta", {}).items() if k != "owner_ids"}
    return {"text": o["text"], "model": o["model"], "input_tokens": o["input_tokens"], "output_tokens": o["output_tokens"],
            "latency_ms": o["latency_ms"], "stop_reason": o.get("stop_reason"), "context_chars": o.get("context_chars"),
            "context_meta": cm, "error": o.get("error")}


def slim_score(s: dict) -> dict:
    return {k: v for k, v in s.items() if k != "negated_mentions"} | {
        "claims": [{"text": c["text"], "kind": c["kind"], "start": c["start"], "end": c["end"], "status": c["status"],
                    "matched_path": c["matched_path"], "matched_value": c["matched_value"],
                    "denominator_stated": c["denominator_stated"]} for c in s["claims"]],
        "external_flags": [{"phrase": f["phrase"], "start": f["start"], "end": f["end"]} for f in s["external_flags"]]}


def persona_strip(rec: dict) -> dict:
    birth = 2026 - rec["age"]
    return {
        "owner_id": rec["owner_id"], "age": rec["age"], "archetype": rec["archetype"], "birth_year": birth,
        "life_stage": rec["demographics"]["life_stage"], "household": rec["demographics"]["household_composition"],
        "income_bracket": rec["demographics"]["income_bracket"], "area": rec["demographics"]["area_type"],
        "events": [{"year": e["year"], "age": e["age"], "event": e["event"], "cause": e.get("cause")} for e in rec["life_timeline"]],
        "transactions": [{"year": t["year"], "type": t["type"], "body": t["body_type"], "powertrain": t["powertrain"],
                          "price_usd": t["price_usd"], "band": t["price_band"], "trigger": t["trigger_event"]} for t in rec["vehicle_timeline"]["transactions"]],
        "spend": [[r[0], r[3]] for r in rec["annual_spend"]],
        "decades": {k: {"cluster": v["cluster"], "avg": v["avg_annual_spend"], "trigger": v["top_trigger"]} for k, v in rec["lifecycle_intelligence"]["decade_profile"].items()},
        "top_triggers": rec["ownership_intelligence"]["top_trigger_events"],
        "progression": rec["lifecycle_intelligence"]["progression_type"], "rfm": rec["lifecycle_intelligence"]["rfm_cluster"],
    }


def pick_personas(records):
    want = [("fresh_graduate", lambda r: r["ownership_intelligence"]["n_purchases"] >= 1),
            ("dual_income_parents", lambda r: r["temporal_cross_intelligence"]["n_children"] >= 2 and r["career_timeline"]["promotion_years"] and r["ownership_intelligence"]["n_purchases"] >= 4),
            ("single_hobbyist", lambda r: any(t["trigger_event"] == "new_hobby" for t in r["vehicle_timeline"]["transactions"])),
            ("retiree", lambda r: r["career_timeline"]["retirement_year"] and any(t["trigger_event"] == "retirement" for t in r["vehicle_timeline"]["transactions"]))]
    out = []
    for arch, pred in want:
        cands = [r for r in records if r["archetype"] == arch and pred(r)]
        cands.sort(key=lambda r: -len([t for t in r["vehicle_timeline"]["transactions"] if t["trigger_event"] not in ("replacement_cycle",)]))
        out.append(persona_strip(cands[len(cands) // 3]))     # not the most extreme — a typical strong example
    return out


def dataset_overview(records):
    n = len(records)
    ev = Counter(e["event"] for r in records for e in r["life_timeline"])
    per = defaultdict(lambda: [0, 0])
    for r in records:
        for t in r["vehicle_timeline"]["transactions"]:
            if t["type"] in ("purchase", "lease"):
                p = "before 2012" if t["year"] < 2012 else "2012–17" if t["year"] < 2018 else "2018–21" if t["year"] < 2022 else "2022–26"
                per[p][1] += 1; per[p][0] += t["powertrain"] == "ev"
    n_tx = sum(r["ownership_intelligence"]["n_transactions"] for r in records)
    # causal check for the page
    n_ev = before = after = 0
    for r in records:
        vs = r["vehicle_timeline"]["vehicles"]
        def held(y): return any(v["body_type"] in FAMILY and v["acquired_year"] <= y and (v["sold_year"] is None or v["sold_year"] > y) for v in vs)
        for e in r["life_timeline"]:
            if e["event"] == "new_child":
                n_ev += 1; before += held(e["year"] - 1); after += any(held(e["year"] + k) for k in (0, 1, 2))
    return {
        "n_households": n, "n_transactions": n_tx,
        "years": [min(r["annual_spend"][0][0] for r in records), 2026],
        "archetypes": dict(Counter(r["archetype"] for r in records).most_common()),
        "life_stage": dict(Counter(r["demographics"]["life_stage"] for r in records).most_common()),
        "body_type_now": dict(Counter(r["ownership_intelligence"]["primary_body_type"] for r in records).most_common()),
        "events": dict(ev.most_common()),
        "ev_share_by_period": {k: round(v[0] / v[1] * 100, 1) for k, v in per.items()},
        "causal": {"new_child_family_before_pct": round(before / n_ev * 100, 1), "new_child_family_after_pct": round(after / n_ev * 100, 1),
                   "population_family_pct": round(sum(1 for r in records if r["ownership_intelligence"]["primary_body_type"] in FAMILY) / n * 100, 1)},
        "trigger_share": dict(Counter(t["trigger_event"] for r in records for t in r["vehicle_timeline"]["transactions"] if t["type"] in ("purchase", "lease")).most_common()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyst", default=str(ROOT / "eval/runs/anthropic_sonnet5_fleet_analyst_v1.scored.json"))
    ap.add_argument("--strategist", default=str(ROOT / "eval/runs/anthropic_sonnet5_market_strategist_v1.scored.json"))
    ap.add_argument("--out", default=str(ROOT / "site/replay.json"))
    args = ap.parse_args()

    A = json.load(open(args.analyst))
    S = json.load(open(args.strategist)) if Path(args.strategist).exists() else None
    queries = json.load(open(ROOT / "eval/queries.json"))
    records = json.load(open(ROOT / "pipeline/data/households.json"))
    strat = {e["id"]: e for e in S["scored"]} if S else {}

    out_q = []
    for q in queries:
        e = next(x for x in A["scored"] if x["id"] == q["id"])
        item = {"id": q["id"], "question": e["question"], "exposes": q["exposes"], "spec": q["spec"],
                "stats": e["stats"], "sample_stats": e.get("sample_stats", {}).get("naive_grounded"),
                "modes": {m: {"output": slim_output(e["outputs"][m]), "score": slim_score(e["scores"][m])} for m in e["outputs"] if m in e["scores"]}}
        if q["id"] in strat and "pipeline_grounded" in strat[q["id"]]["outputs"]:
            se = strat[q["id"]]
            item["strategist"] = {"output": slim_output(se["outputs"]["pipeline_grounded"]), "score": slim_score(se["scores"]["pipeline_grounded"])}
        out_q.append(item)

    bundle = {
        "meta": {"model": A["model"], "created": A["created"], "settings": {"thinking": "adaptive (model default)", "max_tokens": 20000, "sampling": "none (rejected by Sonnet 5)"},
                 "naive_budget": "~200 pre-filtered households, ~285k tokens", "repo": "https://github.com/glee2dev/vehicle-intelligence"},
        "scoreboard": A["scoreboard"],
        "strategist_scoreboard": S["scoreboard"] if S else None,
        "queries": out_q,
        "personas": {k: {"name": v["name"], "style": v["style"]} for k, v in PERSONAS.items()},
        "prompts": {"task": _TASK, "grounding_rules": GROUNDING_RULES, "schema_summary": SCHEMA_SUMMARY},
        "dataset": dataset_overview(records),
        "persona_strips": pick_personas(records),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(bundle, open(args.out, "w"), separators=(",", ":"), ensure_ascii=False)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1024:.0f} KB) — {len(out_q)} queries, strategist={'yes' if S else 'no'}")


if __name__ == "__main__":
    main()
