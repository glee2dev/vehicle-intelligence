"""
LLM client. Two backends:
  mock       — no network, no key. Mode 3 narrates from a template (numbers come from the stats,
               so the demo runs end-to-end and stays honest). Modes 1/2 return a labelled placeholder;
               there is no way to fake an LLM's mistakes without rigging the experiment.
  anthropic  — Messages API via httpx.AsyncClient. Never blocks the event loop.

Select with LLM_BACKEND=mock|anthropic. Model via LLM_MODEL (default claude-sonnet-5).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import re

import httpx
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResult:
    text: str
    model: str
    backend: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    stop_reason: str | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)


class MockBackend:
    name = "mock"
    model = "mock-template"

    async def generate(self, system: str, user: str, temperature: float, max_tokens: int,
                       mode: str = "", stats: dict | None = None, question: str = "") -> LLMResult:
        t0 = time.perf_counter()
        if mode == "pipeline_grounded" and stats is not None:
            text = _template_narrate(question, stats)
        elif mode == "naive_grounded":
            text = ("[mock backend] No model is configured, so nothing computed an answer from the "
                    "records. Set LLM_BACKEND=anthropic and ANTHROPIC_API_KEY to run this mode.")
        else:
            text = ("[mock backend] No model is configured. Set LLM_BACKEND=anthropic and "
                    "ANTHROPIC_API_KEY to run this mode.")
        return LLMResult(text=text, model=self.model, backend=self.name,
                         input_tokens=len(system + user) // 4, output_tokens=len(text) // 4,
                         latency_ms=round((time.perf_counter() - t0) * 1000, 1), stop_reason="end_turn")


_NO_SAMPLING = re.compile(r"claude-(?:sonnet-(?:[5-9]|\d{2})|opus-(?:4-(?:[7-9]|\d{2})|[5-9]|\d{2})|fable|mythos)")


def supports_sampling(model: str) -> bool:
    """Sonnet 5+ and Opus 4.7+ reject non-default temperature/top_p/top_k (400)."""
    return not _NO_SAMPLING.search(model)


class AnthropicBackend:
    name = "anthropic"

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "claude-sonnet-5")
        self.thinking = os.getenv("LLM_THINKING", "adaptive")          # adaptive (model default) | disabled
        self.effort = os.getenv("LLM_EFFORT", "")                       # optional: low|medium|high|xhigh|max
        self.url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set for LLM_BACKEND=anthropic")
        self._headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"}

    async def generate(self, system: str, user: str, temperature: float, max_tokens: int, **_) -> LLMResult:
        payload = {"model": self.model, "max_tokens": max_tokens,
                   "system": system, "messages": [{"role": "user", "content": user}]}
        sampling_applied = False
        if supports_sampling(self.model):
            payload["temperature"] = temperature; sampling_applied = True
        if self.thinking == "disabled":
            payload["thinking"] = {"type": "disabled"}
        if self.effort:
            payload["output_config"] = {"effort": self.effort}
        meta = {"sampling_applied": sampling_applied, "thinking": self.thinking, "effort": self.effort or "default"}
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                r = await client.post(self.url, headers=self._headers, json=payload)
                r.raise_for_status()
                d = r.json()
        except httpx.HTTPStatusError as e:
            return LLMResult(text="", model=self.model, backend=self.name, input_tokens=0, output_tokens=0,
                             latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                             error=f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        except httpx.HTTPError as e:
            return LLMResult(text="", model=self.model, backend=self.name, input_tokens=0, output_tokens=0,
                             latency_ms=round((time.perf_counter() - t0) * 1000, 1), error=str(e))
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        usage = d.get("usage", {})
        meta["thinking_blocks"] = sum(1 for b in d.get("content", []) if b.get("type") == "thinking")
        return LLMResult(text=text, model=d.get("model", self.model), backend=self.name,
                         input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                         latency_ms=round((time.perf_counter() - t0) * 1000, 1), stop_reason=d.get("stop_reason"),
                         meta=meta)


def make_backend(name: str | None = None):
    name = (name or os.getenv("LLM_BACKEND", "mock")).lower()
    if name == "mock":
        return MockBackend()
    if name == "anthropic":
        return AnthropicBackend()
    raise ValueError(f"Unknown LLM_BACKEND {name!r}; use mock or anthropic")


# ── Template narrator for the mock backend ─────────────────────────────────────

def _template_narrate(question: str, stats: dict) -> str:
    """Deterministic narration of a pipeline result. Only prints values present in stats."""
    lines = [f"Question: {question}", ""]
    if not stats.get("answerable", True):
        lines.append(f"This cannot be answered from the dataset. {stats.get('unanswerable_reason', '')}")
        ap = stats.get("answerable_part")
        if ap:
            lines.append("")
            lines.append(f"What the data does support (n={ap['n_matched']:,} of {ap['n_total']:,}):")
            for g, s in ap["groups"].items():
                lines.append(_flat(s, prefix=f"  [{g}] "))
        return "\n".join(lines)
    lines.append(f"Segment: {stats['n_matched']:,} of {stats['n_total']:,} households ({stats['match_pct']}%).")
    for g, s in stats["groups"].items():
        lines.append(_flat(s, prefix=f"[{g}] "))
    ap = stats.get("answerable_part")
    if ap:
        lines.append(f"Comparison segment: {ap['n_matched']:,} households.")
        for g, s in ap["groups"].items():
            lines.append(_flat(s, prefix=f"[{g}] "))
    lines.append("")
    lines.append("All figures above are taken directly from the pipeline output; nothing external was added.")
    return "\n".join(lines)


def _flat(s: dict, prefix: str = "", limit: int = 900) -> str:
    parts = []
    for k, v in s.items():
        if isinstance(v, (int, float, str)) or v is None:
            parts.append(f"{k}={v}")
        elif isinstance(v, dict) and all(isinstance(x, (int, float, str, type(None))) for x in v.values()):
            parts.append(f"{k}={json.dumps(v)}")
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            parts.append(f"{k}=" + "; ".join(json.dumps(x) for x in v[:4]))
    out = prefix + ", ".join(parts)
    return out if len(out) <= limit else out[:limit] + " …"
