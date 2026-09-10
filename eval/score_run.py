#!/usr/bin/env python3
"""
Score a run file (from run_modes.py) and print the scoreboard.
  python eval/score_run.py eval/runs/anthropic_fleet_analyst_20260910_101500.json
Writes <run>.scored.json next to it. Mode 2 is scored against ground truth recomputed on the
exact records it received (owner_ids in the trace); modes 1 and 3 against population truth.
"""
import argparse, json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "pipeline"))
from check import score_output
from engine import run_query


def sample_truth(records_by_id: dict, owner_ids: list[str], spec: dict) -> dict:
    subset = [records_by_id[i] for i in owner_ids if i in records_by_id]
    return run_query(subset, spec)


def mean(xs, nd=3):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), nd) if xs else None


def scoreboard(scored: list[dict]) -> dict:
    modes = sorted({m for r in scored for m in r["scores"]})
    board = {}
    for m in modes:
        S = [r["scores"][m] for r in scored if m in r["scores"]]
        O = [r["outputs"][m] for r in scored if m in r["outputs"]]
        unans = [s for s in S if s["unanswerable_expected"]]
        board[m] = {
            "n_queries": len(S),
            "answered": sum(1 for s in S if s["answer_status"] == "answered"),
            "declined_no_numbers": sum(1 for s in S if s["answer_status"] == "declined"),
            "empty_truncated": sum(1 for s in S if s["answer_status"] == "empty"),
            "hit_max_tokens": sum(1 for o in O if o.get("stop_reason") == "max_tokens"),
            "claims_total": sum(s["n_claims"] for s in S),
            "verified": sum(s["n_verified"] for s in S),
            "derived_or_raw": sum(s["n_derived"] for s in S),
            "unverified": sum(s["n_unverified"] for s in S),
            "grounding_rate_pooled": round(sum(s["n_verified"] + s["n_derived"] for s in S) / max(1, sum(s["n_claims"] for s in S)), 3),
            "grounding_rate_mean": mean([s["grounding_rate"] for s in S]),
            "queries_fully_grounded": sum(1 for s in S if s["n_claims"] and s["n_unverified"] == 0),
            "external_flags": sum(s["n_external_flags"] for s in S),
            "queries_with_external": sum(1 for s in S if s["n_external_flags"]),
            "pct_with_denominator_mean": mean([s["pct_with_denominator"] for s in S]),
            "unanswerable_refused": f"{sum(1 for s in unans if s['refusal_detected'])}/{len(unans)}",
            "fabricated_on_unanswerable": sum(s["fabricated_on_unanswerable"] or 0 for s in unans),
            "input_tokens_mean": mean([o["input_tokens"] for o in O], 0),
            "output_tokens_mean": mean([o["output_tokens"] for o in O], 0),
            "latency_ms_mean": mean([o["latency_ms"] for o in O], 0),
            "errors": sum(1 for o in O if o.get("error")),
        }
    return board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_file")
    ap.add_argument("--data", default=str(ROOT / "pipeline/data/households.json"))
    args = ap.parse_args()

    run = json.load(open(args.run_file))
    records_by_id = None
    scored = []
    for r in run["runs"]:
        entry = {"id": r["id"], "question": r["question"], "outputs": r["outputs"], "stats": r["stats"], "scores": {}}
        for mode, out in r["outputs"].items():
            if out.get("error"):
                continue
            sample = None
            ids = out.get("context_meta", {}).get("owner_ids")
            if mode == "naive_grounded" and ids:
                if records_by_id is None:
                    records_by_id = {x["owner_id"]: x for x in json.load(open(args.data))}
                sample = sample_truth(records_by_id, ids, r["spec"])
                entry.setdefault("sample_stats", {})[mode] = sample
            given = None
            if mode == "naive_grounded":
                rm = r.get("records_meta", {})
                given = {"n_records_given": out["context_meta"].get("records"), "n_segment_matched": rm.get("n_matched"),
                         "n_survey_total": rm.get("n_total")}
                for i, seg in enumerate(rm.get("segments", [])):
                    given[f"segment_{i}_matched"] = seg.get("n_matched")
            raw = {i: records_by_id[i] for i in ids if i in records_by_id} if (mode == "naive_grounded" and ids) else None
            entry["scores"][mode] = score_output(out["text"], mode, r["stats"], sample, given, raw).__dict__
        scored.append(entry)

    board = scoreboard(scored)
    out_path = Path(args.run_file).with_suffix(".scored.json")
    json.dump({"backend": run["backend"], "model": run["model"], "persona": run["persona"], "created": run["created"],
               "scoreboard": board, "scored": scored}, open(out_path, "w"), indent=1)

    print(f"\nSCOREBOARD  backend={run['backend']} model={run['model']} persona={run['persona']}\n")
    keys = ["answered", "declined_no_numbers", "empty_truncated", "hit_max_tokens", "claims_total", "verified", "derived_or_raw", "unverified", "grounding_rate_pooled", "queries_fully_grounded",
            "external_flags", "queries_with_external", "pct_with_denominator_mean", "unanswerable_refused",
            "fabricated_on_unanswerable", "input_tokens_mean", "output_tokens_mean", "latency_ms_mean", "errors"]
    modes = list(board)
    print(f"{'':<28}" + "".join(f"{m:>20}" for m in modes))
    for k in keys:
        print(f"{k:<28}" + "".join(f"{str(board[m][k]):>20}" for m in modes))
    print(f"\nper query (unverified / claims):")
    for e in scored:
        def cell(m):
            sc = e["scores"][m]
            return f"{m[:6]}={'EMPTY' if sc['answer_status'] == 'empty' else 'decl.' if sc['answer_status'] == 'declined' else str(sc['n_unverified']) + '/' + str(sc['n_claims'])}"
        print(f"  {e['id']}  " + "  ".join(cell(m) for m in modes if m in e["scores"])
              + ("   [unanswerable]" if not e["stats"].get("answerable", True) else ""))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
