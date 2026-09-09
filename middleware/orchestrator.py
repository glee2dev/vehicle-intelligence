"""
Run one question through up to three modes concurrently and return a trace.
The pipeline does the same deterministic work for every mode that gets data; the modes differ
only in what the LLM receives (see modes.py).
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict

from llm_client import make_backend
from modes import MODES, PERSONAS, build_naive_grounded, build_pipeline_grounded, build_ungrounded

MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "20000"))   # adaptive thinking counts against this; answers are capped by prompt at ~250 words
NAIVE_CHAR_BUDGET = 600_000       # ~150k tokens of records — as much as the context window comfortably allows; stated in the trace
NAIVE_MAX_RECORDS = 400


async def run_question(question: str, spec: dict, pipeline, backend=None, modes=MODES,
                       persona: str = "fleet_analyst", temperature: float | None = None, seed: int = 0) -> dict:
    backend = backend or make_backend()
    temp = PERSONAS[persona]["temperature"] if temperature is None else temperature
    t0 = time.perf_counter()

    # Deterministic work first — done once, shared by every mode that needs it.
    stats = await pipeline.query(spec)
    records_payload = None
    if "naive_grounded" in modes:
        base = await pipeline.records(spec.get("filters", {}), NAIVE_MAX_RECORDS, seed)
        also_f = (spec.get("also") or {}).get("filters")
        if spec.get("also") and also_f != spec.get("filters", {}):
            # two-cohort question: interleave records from both cohorts so the packer's budget
            # cut-off leaves mode 2 with roughly equal numbers from each
            other = await pipeline.records(also_f or {}, NAIVE_MAX_RECORDS, seed)
            merged = [r for pair in zip(base["records"], other["records"]) for r in pair]
            records_payload = {"n_matched": base["n_matched"] + other["n_matched"], "n_returned": len(merged),
                               "truncated": len(merged) < base["n_matched"] + other["n_matched"],
                               "n_total": base["n_total"], "records": merged,
                               "segments": [{"filters": spec.get("filters", {}), "n_matched": base["n_matched"]},
                                            {"filters": also_f or {}, "n_matched": other["n_matched"]}]}
        else:
            records_payload = base
    pipeline_ms = round((time.perf_counter() - t0) * 1000, 1)

    ctxs = {}
    if "ungrounded" in modes:
        ctxs["ungrounded"] = build_ungrounded(question)
    if "naive_grounded" in modes:
        ctxs["naive_grounded"] = build_naive_grounded(question, records_payload, NAIVE_CHAR_BUDGET)
    if "pipeline_grounded" in modes:
        ctxs["pipeline_grounded"] = build_pipeline_grounded(question, stats, persona)

    async def one(mode, ctx):
        r = await backend.generate(ctx.system, ctx.user, temperature=temp, max_tokens=MAX_TOKENS,
                                   mode=mode, stats=stats, question=question)
        d = asdict(r)
        d.update({"mode": mode, "context_chars": ctx.context_chars, "context_meta": ctx.context_meta,
                  "temperature": temp})
        return mode, d

    results = dict(await asyncio.gather(*(one(m, c) for m, c in ctxs.items())))
    return {
        "question": question, "spec": spec, "persona": persona, "backend": backend.name,
        "model": getattr(backend, "model", "?"), "temperature": temp,
        "pipeline_ms": pipeline_ms, "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "stats": {k: v for k, v in stats.items() if k not in ("cached",)},
        "records_meta": {k: v for k, v in (records_payload or {}).items() if k != "records"},
        "outputs": results,
    }
