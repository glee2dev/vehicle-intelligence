#!/usr/bin/env python3
"""
Run the curated queries through the three modes and save a run file.
In-process pipeline (no servers needed). Usage:
  LLM_BACKEND=mock python eval/run_modes.py
  LLM_BACKEND=anthropic ANTHROPIC_API_KEY=... python eval/run_modes.py --ids Q01 Q04 --persona fleet_analyst
"""
import argparse, asyncio, json, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "middleware"))
from llm_client import make_backend
from orchestrator import run_question
from pipeline_client import LocalPipeline
from modes import MODES


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    ap.add_argument("--persona", default="fleet_analyst")
    ap.add_argument("--queries", default=str(ROOT / "eval/queries.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()

    queries = json.load(open(args.queries))
    if args.ids:
        queries = [q for q in queries if q["id"] in args.ids]
    pipeline = LocalPipeline()
    backend = make_backend()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(q):
        async with sem:
            t0 = time.perf_counter()
            r = await run_question(q["question"], q["spec"], pipeline, backend, tuple(args.modes), args.persona)
            r["id"], r["exposes"] = q["id"], q.get("exposes")
            errs = [m for m, o in r["outputs"].items() if o.get("error")]
            print(f"{q['id']}  {time.perf_counter() - t0:5.1f}s  " +
                  "  ".join(f"{m[:6]}={o['input_tokens']:>6}in/{o['output_tokens']:>4}out" for m, o in r["outputs"].items()) +
                  (f"  ERRORS: {errs}" if errs else ""))
            return r

    runs = await asyncio.gather(*(one(q) for q in queries))
    out = Path(args.out) if args.out else ROOT / "eval/runs" / f"{backend.name}_{args.persona}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"backend": backend.name, "model": getattr(backend, "model", "?"), "persona": args.persona,
               "queries_file": args.queries, "created": datetime.now().isoformat(), "runs": runs}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
