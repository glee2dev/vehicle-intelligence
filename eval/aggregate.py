#!/usr/bin/env python3
"""
Aggregate every scored run under eval/runs/ into one table.
  python eval/aggregate.py                # reads eval/runs/*fleet_analyst*.scored.json
Writes eval/aggregate.json and eval/AGGREGATE.md. Runs are grouped by model label; a model with
repeats (Claude v1/v2/v3) gets a mean row with the per-run spread. The pooled row treats every
claim from every run as one sample, which is the number the site's hero shows.
"""
import argparse, glob, json, re, statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODES = ["ungrounded", "naive_grounded", "pipeline_grounded"]

LABELS = {  # run-file stem prefix -> display label; anything else falls back to the model id
    "anthropic_sonnet5": "Claude Sonnet 5 (adaptive thinking)",
    "openai_google-gemini-3-1-pro-preview": "Gemini 3.1 Pro",
    "openai_x-ai-grok-4-6": "Grok 4.6",
    "openai_openai-gpt-5-4-reasoning-medium": "GPT-5.4 (reasoning=medium)",
    "openai_openai-gpt-5-4": "GPT-5.4 (provider default: no reasoning)",
    "openai_qwen-qwen3-8-2-4t-a95b": "Qwen 3.8 2.4T-A95B",
}


def label_for(stem: str, model: str) -> str:
    stem = re.sub(r"_fleet_analyst_v\d+(\.scored)?$", "", stem)
    return LABELS.get(stem, model)


def per_run(s: dict, mode: str) -> dict:
    """Counts a scoreboard already carries, plus what the aggregate needs to pool later."""
    b = s["scoreboard"][mode]
    unans = [e["scores"][mode] for e in s["scored"] if mode in e["scores"] and e["scores"][mode]["unanswerable_expected"]]
    return {
        "claims": b["claims_total"], "grounded": b["verified"] + b["derived_or_raw"], "unverified": b["unverified"],
        "grounding_rate": b["grounding_rate_pooled"],
        "fully_grounded_queries": b["queries_fully_grounded"], "answered": b["answered"], "empty": b["empty_truncated"],
        "external_flags": b["external_flags"],
        "refused": sum(1 for u in unans if u["refusal_detected"]), "unanswerable": len(unans),
        "fabricated_on_unanswerable": b["fabricated_on_unanswerable"],
        "input_tokens_mean": b["input_tokens_mean"], "output_tokens_mean": b["output_tokens_mean"], "latency_ms_mean": b["latency_ms_mean"],
    }


def pool(rows: list[dict]) -> dict:
    claims = sum(r["claims"] for r in rows); grounded = sum(r["grounded"] for r in rows)
    return {
        "runs": len(rows), "claims": claims, "grounded": grounded, "unverified": sum(r["unverified"] for r in rows),
        "grounding_rate": round(grounded / max(1, claims), 3),
        "grounding_rate_min": min(r["grounding_rate"] for r in rows), "grounding_rate_max": max(r["grounding_rate"] for r in rows),
        "fully_grounded_queries": sum(r["fully_grounded_queries"] for r in rows), "queries": 10 * len(rows),
        "empty": sum(r["empty"] for r in rows),
        "external_flags": sum(r["external_flags"] for r in rows),
        "refused": sum(r["refused"] for r in rows), "unanswerable": sum(r["unanswerable"] for r in rows),
        "fabricated_on_unanswerable": sum(r["fabricated_on_unanswerable"] for r in rows),
        "input_tokens_mean": round(statistics.mean(r["input_tokens_mean"] for r in rows)),
        "output_tokens_mean": round(statistics.mean(r["output_tokens_mean"] for r in rows)),
        "latency_ms_mean": round(statistics.mean(r["latency_ms_mean"] for r in rows)),
    }


def pct(x): return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "eval/runs/*fleet_analyst*.scored.json"))
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    runs = []
    for f in files:
        s = json.load(open(f)); stem = Path(f).stem.replace(".scored", "")
        runs.append({"file": Path(f).name, "stem": stem, "label": label_for(stem, s["model"]), "model": s["model"],
                     "backend": s["backend"], "created": s["created"], "modes": {m: per_run(s, m) for m in MODES}})

    by_label = {}
    for r in runs: by_label.setdefault(r["label"], []).append(r)
    models = {lab: {m: pool([r["modes"][m] for r in rs]) for m in MODES} for lab, rs in by_label.items()}
    pooled = {m: pool([r["modes"][m] for r in runs]) for m in MODES}
    out = {"n_runs": len(runs), "n_models": len(models), "runs": runs, "models": models, "pooled": pooled}
    json.dump(out, open(ROOT / "eval/aggregate.json", "w"), indent=1)

    L = ["# Aggregate across models and repeats", "",
         f"{len(runs)} runs, {len(models)} model configurations, ten questions each, same three modes, same prompts. "
         "Grounding rate = (verified + derived) / all numeric claims, pooled over the run's claims. "
         "A model with repeats shows the pooled figure with the per-run range.", "",
         "## Pooled over everything", "", "| | ungrounded | naive-grounded | pipeline-grounded |", "|---|---:|---:|---:|"]
    P = pooled
    L += [f"| claims checked | {P['ungrounded']['claims']} | {P['naive_grounded']['claims']} | {P['pipeline_grounded']['claims']} |",
          "| numbers verified | " + " | ".join(pct(P[m]["grounding_rate"]) for m in MODES) + " |",
          "| questions fully grounded | " + " | ".join(f"{P[m]['fully_grounded_queries']}/{P[m]['queries']}" for m in MODES) + " |",
          "| empty or truncated answers | " + " | ".join(str(P[m]["empty"]) for m in MODES) + " |",
          "| external-knowledge phrases | " + " | ".join(str(P[m]["external_flags"]) for m in MODES) + " |",
          "| refused the unanswerable | " + " | ".join(f"{P[m]['refused']}/{P[m]['unanswerable']}" for m in MODES) + " |",
          "| fabricated on unanswerable | " + " | ".join(str(P[m]["fabricated_on_unanswerable"]) for m in MODES) + " |",
          "| input tokens per question | " + " | ".join(f"{P[m]['input_tokens_mean']:,}" for m in MODES) + " |", "",
          "## Per model", "", "| model | runs | ungrounded | naive-grounded | pipeline-grounded | naive empty | refused unanswerable (naive / pipeline) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for lab, M in models.items():
        def cell(m):
            c = M[m]
            return pct(c["grounding_rate"]) + (f" ({pct(c['grounding_rate_min'])}–{pct(c['grounding_rate_max'])})" if c["runs"] > 1 else "")
        L.append(f"| {lab} | {M['ungrounded']['runs']} | {cell('ungrounded')} | {cell('naive_grounded')} | {cell('pipeline_grounded')} | "
                 f"{M['naive_grounded']['empty']}/{M['naive_grounded']['queries']} | "
                 f"{M['naive_grounded']['refused']}/{M['naive_grounded']['unanswerable']} / {M['pipeline_grounded']['refused']}/{M['pipeline_grounded']['unanswerable']} |")
    L += ["", "Grounding rate counts claims in answers that were produced. A model whose naive-grounded answers came back empty "
          "(it spent the 20k-token budget thinking) has few claims to check there; read the naive column next to the empty column.", "",
          "## Every run", "", "| run | ungrounded | naive-grounded | pipeline-grounded |", "|---|---:|---:|---:|"]
    for r in runs:
        L.append(f"| `{r['file']}` | " + " | ".join(f"{r['modes'][m]['grounded']}/{r['modes'][m]['claims']}" for m in MODES) + " |")
    L += ["", "Rerun with `python eval/aggregate.py` after scoring a new run file.", ""]
    open(ROOT / "eval/AGGREGATE.md", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
