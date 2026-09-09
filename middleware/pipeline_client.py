"""Access to the pipeline — over HTTP (servers) or in-process (eval runs, tests). Same interface."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent


class HttpPipeline:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def query(self, spec: dict) -> dict:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(f"{self.base_url}/query", json=spec)
            r.raise_for_status()
            return r.json()

    async def records(self, filters: dict, max_records: int, seed: int = 0) -> dict:
        async with httpx.AsyncClient(timeout=120.0) as c:
            r = await c.post(f"{self.base_url}/records", json={"filters": filters, "max_records": max_records, "seed": seed})
            r.raise_for_status()
            return r.json()

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=5.0) as c:
            return (await c.get(f"{self.base_url}/health")).json()


class LocalPipeline:
    def __init__(self, data_file: Path | None = None):
        sys.path.insert(0, str(ROOT / "pipeline"))
        from engine import apply_filters, run_query  # noqa
        self._run, self._filter = run_query, apply_filters
        self.records_ = json.load(open(data_file or ROOT / "pipeline/data/households.json"))

    async def query(self, spec: dict) -> dict:
        return self._run(self.records_, spec)

    async def records(self, filters: dict, max_records: int, seed: int = 0) -> dict:
        m = self._filter(self.records_, filters)
        s = m if len(m) <= max_records else random.Random(seed).sample(m, max_records)
        return {"n_matched": len(m), "n_returned": len(s), "truncated": len(s) < len(m),
                "n_total": len(self.records_), "records": [{k: v for k, v in r.items() if k != "archetype"} for r in s]}

    async def health(self) -> dict:
        return {"status": "ok", "records": len(self.records_)}
