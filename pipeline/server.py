#!/usr/bin/env python3
"""
Data pipeline server — port 8000.
Owns the dataset. Runs structured queries deterministically. Never calls an LLM.

Endpoints
  POST /query     structured QuerySpec -> aggregated StatsResult (what mode 3 narrates from)
  POST /records   filtered raw records, trimmed to a budget (what mode 2 has to compute from)
  GET  /schema    filterable fields, metrics, groupable dimensions
  GET  /queries   the curated query set
  GET  /health
"""
import hashlib
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from engine import (COUNT_FIELDS, DEMOGRAPHIC_FIELDS, GROUPABLE, METRICS, UNANSWERABLE,
                    apply_filters, run_query)

ROOT = Path(__file__).parent.parent
DATA_FILE = Path(os.getenv("DATA_FILE", ROOT / "pipeline" / "data" / "households.json"))
PORT = int(os.getenv("PIPELINE_PORT", "8000"))
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "http://localhost:3000")
MAX_CACHE = 500

_RECORDS: list[dict] = []
_CACHE: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _RECORDS
    if not DATA_FILE.exists():
        raise RuntimeError(f"Dataset not found at {DATA_FILE}. Run: python tools/generate_dataset.py")
    t0 = time.perf_counter()
    with open(DATA_FILE, encoding="utf-8") as f:
        _RECORDS = json.load(f)
    print(f"[pipeline] loaded {len(_RECORDS):,} households in {time.perf_counter() - t0:.1f}s")
    yield


app = FastAPI(title="Vehicle Intelligence Pipeline", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[ALLOWED_ORIGIN], allow_methods=["GET", "POST"], allow_headers=["*"])


class QuerySpec(BaseModel):
    filters: dict = Field(default_factory=dict)
    group_by: str | None = None
    metric: str = "segment_profile"
    params: dict = Field(default_factory=dict)
    also: dict | None = None


class RecordsRequest(BaseModel):
    filters: dict = Field(default_factory=dict)
    max_records: int = 200
    fields: list[str] | None = None       # top-level keys to keep; None = all except generator labels
    seed: int = 0


def _key(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@app.post("/query")
def query(spec: QuerySpec):
    body = spec.model_dump(exclude_none=True)
    k = _key(body)
    if k in _CACHE:
        return {**_CACHE[k], "cached": True}
    t0 = time.perf_counter()
    try:
        res = run_query(_RECORDS, body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    res["processing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    res["cached"] = False
    if len(_CACHE) >= MAX_CACHE:
        del _CACHE[next(iter(_CACHE))]
    _CACHE[k] = res
    return res


HIDDEN_KEYS = {"archetype"}   # generator ground-truth labels never leave the pipeline


@app.post("/records")
def records(req: RecordsRequest):
    """Deterministic filtering only. Returns raw records for the naive-grounded mode.
    The pipeline does the *filtering* for the LLM; the LLM still has to do the *math*."""
    try:
        matched = apply_filters(_RECORDS, req.filters)
    except ValueError as e:
        raise HTTPException(400, str(e))
    import random
    rng = random.Random(req.seed)
    sample = matched if len(matched) <= req.max_records else rng.sample(matched, req.max_records)
    keep = req.fields
    out = []
    for r in sample:
        rec = {k: v for k, v in r.items() if k not in HIDDEN_KEYS and (keep is None or k in keep)}
        out.append(rec)
    return {"n_matched": len(matched), "n_returned": len(out), "truncated": len(out) < len(matched),
            "n_total": len(_RECORDS), "records": out}


@app.get("/schema")
def schema():
    return {
        "n_households": len(_RECORDS),
        "filters": {
            "equality": sorted(DEMOGRAPHIC_FIELDS),
            "count": sorted(COUNT_FIELDS),
            "event": ["had_event {event, since_year?, before_year?, min_count?}", "never_had_event [events]", "has_full_decades [decades]"],
        },
        "group_by": sorted(GROUPABLE - {"archetype"}),
        "metrics": {k: (v.__doc__ or "").strip().split("\n")[0] for k, v in METRICS.items()},
        "unanswerable_reasons": UNANSWERABLE,
        "value_domains": {
            "age_group": ["20s", "30s", "40s", "50s", "60s", "70s"],
            "life_stage": ["early_career", "established_professional", "parent_young_children", "parent_school_age", "empty_nester", "retired"],
            "household_composition": ["solo", "partnered_no_children", "partnered_with_children", "single_parent"],
            "income_bracket": ["lower", "lower_middle", "upper_middle", "upper"],
            "area_type": ["urban", "suburban", "rural"],
            "commute_type": ["transit", "car_short", "car_long", "wfh", "none"],
            "body_type": ["compact", "sedan", "crossover", "suv", "minivan", "pickup", "sports", "motorcycle"],
            "powertrain": ["gasoline", "diesel", "hybrid", "ev"],
            "life_event": ["graduation", "first_job", "job_change", "promotion", "career_change", "marriage", "new_child", "divorce", "empty_nest", "retirement", "bereavement", "new_hobby", "moved"],
            "rfm_cluster": ["high_stable", "high_volatile", "rising", "declining", "mid_stable", "dormant", "new_owner"],
            "progression_type": ["ascending", "stable", "descending", "volatile", "insufficient_history"],
        },
    }


@app.get("/queries")
def queries():
    return json.load(open(ROOT / "eval" / "queries.json"))


@app.get("/health")
def health():
    return {"status": "ok", "records": len(_RECORDS), "cache_entries": len(_CACHE)}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=False)
