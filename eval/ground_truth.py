#!/usr/bin/env python3
"""Run the curated queries through the engine and write eval/ground_truth.json."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from engine import run_query

ROOT = Path(__file__).parent.parent
records = json.load(open(ROOT / "pipeline/data/households.json"))
queries = json.load(open(ROOT / "eval/queries.json"))
out = {}
for q in queries:
    t0 = time.perf_counter()
    res = run_query(records, q["spec"])
    res["processing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out[q["id"]] = {"question": q["question"], "exposes": q["exposes"], "result": res}
    print(f"\n{'='*78}\n{q['id']}  {q['question']}\n{'-'*78}")
    print(f"matched {res['n_matched']:,}/{res['n_total']:,} ({res['match_pct']}%)  answerable={res['answerable']}  {res['processing_ms']}ms")
    if not res["answerable"]:
        print("  REASON:", res["unanswerable_reason"])
    for g, stats in res["groups"].items():
        s = json.dumps(stats, indent=None)
        print(f"  [{g}] {s[:600]}{'…' if len(s) > 600 else ''}")
    if "answerable_part" in res:
        ap = res["answerable_part"]
        print(f"  ALSO matched {ap['n_matched']:,}: " + json.dumps(ap["groups"])[:500])
json.dump(out, open(ROOT / "eval/ground_truth.json", "w"), indent=1)
print(f"\nwrote eval/ground_truth.json")
