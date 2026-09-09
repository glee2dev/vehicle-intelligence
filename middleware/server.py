#!/usr/bin/env python3
"""
Middleware server — port 3000. Orchestrates the three modes; never touches raw data itself
(the pipeline filters, the pipeline aggregates). Serves the UI from ./static when present.
"""
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import make_backend
from modes import MODES, PERSONAS
from orchestrator import run_question
from pipeline_client import HttpPipeline

ROOT = Path(__file__).parent.parent
PIPELINE_URL = os.getenv("PIPELINE_URL", "http://localhost:8000")
PORT = int(os.getenv("MIDDLEWARE_PORT", "3000"))
API_KEY = os.getenv("MIDDLEWARE_API_KEY", "")

pipeline = HttpPipeline(PIPELINE_URL)
backend = None
QUERIES = {q["id"]: q for q in json.load(open(ROOT / "eval" / "queries.json"))}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global backend
    backend = make_backend()
    try:
        h = await pipeline.health()
        print(f"[middleware] pipeline ok: {h}")
    except Exception as e:
        print(f"[middleware] WARNING pipeline unreachable at {PIPELINE_URL}: {e}")
    print(f"[middleware] llm backend={backend.name} model={getattr(backend, 'model', '?')}")
    yield


app = FastAPI(title="Vehicle Intelligence Middleware", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def verify_key(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")


class RunRequest(BaseModel):
    query_id: str | None = None          # one of the curated queries
    question: str | None = None          # or a free question with an explicit spec
    spec: dict | None = None
    modes: list[str] = Field(default_factory=lambda: list(MODES))
    persona: str = "fleet_analyst"
    seed: int = 0


@app.post("/api/run")
async def run(req: RunRequest, _=Depends(verify_key)):
    if req.query_id:
        q = QUERIES.get(req.query_id)
        if not q:
            raise HTTPException(404, f"Unknown query_id. Valid: {sorted(QUERIES)}")
        question, spec = q["question"], q["spec"]
    elif req.question and req.spec:
        question, spec = req.question, req.spec
    else:
        raise HTTPException(400, "Provide query_id, or question + spec")
    bad = [m for m in req.modes if m not in MODES]
    if bad:
        raise HTTPException(400, f"Unknown modes {bad}. Valid: {MODES}")
    if req.persona not in PERSONAS:
        raise HTTPException(400, f"Unknown persona. Valid: {sorted(PERSONAS)}")
    try:
        return await run_question(question, spec, pipeline, backend, tuple(req.modes), req.persona, seed=req.seed)
    except Exception as e:  # pipeline down, bad spec, etc.
        raise HTTPException(502, f"{type(e).__name__}: {e}")


@app.get("/api/queries")
def queries():
    return [{"id": q["id"], "question": q["question"], "exposes": q["exposes"]} for q in QUERIES.values()]


@app.get("/api/modes")
def modes():
    return {"modes": list(MODES), "personas": {k: {"name": v["name"], "temperature": v["temperature"]} for k, v in PERSONAS.items()}}


@app.get("/health")
async def health():
    try:
        p = await pipeline.health()
    except Exception as e:
        p = {"status": "unreachable", "error": str(e)}
    return {"status": "ok", "backend": backend.name if backend else None, "pipeline": p}


static_dir = Path(__file__).parent / "static"
if static_dir.exists() and any(static_dir.iterdir()):
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=False)
