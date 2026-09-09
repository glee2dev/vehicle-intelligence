"""Mode context builders: fairness properties that must hold regardless of backend."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "middleware"))
from modes import build_ungrounded, build_naive_grounded, build_pipeline_grounded, RAW_KEYS


def rec(i):
    return {"owner_id": f"HH-{i}", "age": 40, "age_group": "40s", "gender": "female", "archetype": "SECRET",
            "demographics": {"life_stage": "x"}, "life_timeline": [], "career_timeline": {}, "residence_timeline": [],
            "household_timeline": {}, "vehicle_timeline": {"transactions": [{"year": 2020, "age": 34, "type": "purchase",
            "vehicle_id": "V1", "body_type": "suv", "powertrain": "ev", "price_band": 3, "price_usd": 30000,
            "new_or_used": "new", "trigger_event": "new_child", "trigger_year": 2020}], "vehicles": []},
            "annual_spend": [[2020, 1, 2, 3]], "ownership_intelligence": {"LEAK": 1},
            "lifecycle_intelligence": {"LEAK": 1}, "temporal_cross_intelligence": {"LEAK": 1}}


def test_ungrounded_has_no_data():
    c = build_ungrounded("q?")
    assert "HH-" not in c.user and "n_matched" not in c.user and c.context_meta["records"] == 0


def test_naive_gets_raw_timelines_but_no_derived_intelligence_or_labels():
    payload = {"n_matched": 3, "n_returned": 3, "truncated": False, "n_total": 10, "records": [rec(i) for i in range(3)]}
    c = build_naive_grounded("q?", payload)
    assert "LEAK" not in c.user and "SECRET" not in c.user
    assert "HH-0" in c.user and '"body_type":"suv"' in c.user
    assert "all 3 households" in c.user and c.context_meta["truncated"] is False


def test_naive_budget_truncates_and_says_so():
    payload = {"n_matched": 500, "n_returned": 50, "truncated": True, "n_total": 10000, "records": [rec(i) for i in range(50)]}
    c = build_naive_grounded("q?", payload, char_budget=2000)
    assert c.context_meta["records"] < 50 and c.context_meta["truncated"] is True
    assert f"{c.context_meta['records']} of 500" in c.user


def test_pipeline_grounded_gets_stats_only():
    stats = {"n_matched": 5, "n_total": 10, "match_pct": 50.0, "answerable": True, "groups": {"all": {"x": 1}}, "processing_ms": 3}
    c = build_pipeline_grounded("q?", stats, "fleet_analyst")
    assert '"x": 1' in c.user and "processing_ms" not in c.user and "HH-" not in c.user
    assert "STRICT RULES" in c.system and "Fleet Analyst" in c.system


def test_same_task_preamble_in_all_modes():
    a, b, c = build_ungrounded("q").system, build_naive_grounded("q", {"n_matched": 0, "n_returned": 0, "truncated": False, "n_total": 0, "records": []}).system, build_pipeline_grounded("q", {"groups": {}}).system
    first = lambda s: s.split("\n\n")[0]
    assert first(a) == first(b) == first(c)
