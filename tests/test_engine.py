"""Engine tests on tiny hand-built records — the denominators and windows must be exactly right."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
import pytest
from engine import apply_filters, run_query, _cmp, UNANSWERABLE


def hh(id_, age=40, stage="established_professional", comp="solo", income="upper_middle",
       dual=False, married=False, kids_home=0, events=(), purchases=(), spend=(), ever_ev=False,
       n_job_changes=0, n_moves=0, residence=(), decades=None):
    tx = [{"year": y, "age": y - (2026 - age), "type": "purchase", "vehicle_id": f"V{i}", "body_type": b,
           "powertrain": pt, "price_band": 2, "price_usd": pr, "new_or_used": "new",
           "trigger_event": "x", "trigger_year": y} for i, (y, b, pt, pr) in enumerate(purchases)]
    vehicles = [{"vehicle_id": f"V{i}", "body_type": b, "acquired_year": y, "sold_year": None, "role": "primary"}
                for i, (y, b, pt, pr) in enumerate(purchases)]
    return {
        "owner_id": id_, "age": age, "age_group": f"{age // 10 * 10}s", "gender": "female", "archetype": "t",
        "demographics": {"life_stage": stage, "household_composition": comp, "income_bracket": income,
                         "dual_income": dual, "ever_married": married, "n_children_home": kids_home,
                         "area_type": "suburban", "commute_type": "car_short", "parking": "garage"},
        "life_timeline": [{"year": y, "age": y - (2026 - age), "event": e} for y, e in events],
        "residence_timeline": [{"from_year": y, "age": 0, "area_type": "suburban", "commute_type": c, "parking": "garage", "trigger_event": t}
                               for y, c, t in residence] or [{"from_year": 2000, "age": 18, "area_type": "suburban", "commute_type": "car_short", "parking": "garage", "trigger_event": "initial"}],
        "vehicle_timeline": {"transactions": tx, "vehicles": vehicles},
        "annual_spend": [[y, s, 0, s] for y, s in spend],
        "ownership_intelligence": {"primary_body_type": purchases[-1][1] if purchases else "none", "primary_powertrain": "gasoline",
                                   "ever_ev": ever_ev, "n_purchases": len(tx), "current_vehicle_count": len(tx),
                                   "avg_price_per_purchase_usd": sum(p[3] for p in purchases) / len(purchases) if purchases else 0,
                                   "spend_span": {"min": min((s for _, s in spend), default=0), "max": max((s for _, s in spend), default=0),
                                                  "range": (max((s for _, s in spend), default=0) - min((s for _, s in spend), default=0))},
                                   "avg_annual_spend_5y": 0, "lifetime_spend": sum(s for _, s in spend), "top_trigger_events": [],
                                   "last_transaction_year": None},
        "lifecycle_intelligence": {"decade_profile": decades or {}, "progression_type": "stable", "rfm_cluster": "mid_stable"},
        "temporal_cross_intelligence": {"n_job_changes": n_job_changes, "n_moves": n_moves, "n_children": kids_home,
                                        "n_life_events": len(events), "n_promotions": 0, "n_career_changes": 0},
    }


def test_cmp_operators():
    assert _cmp(3, {"gte": 2}) and not _cmp(1, {"gte": 2})
    assert _cmp(0, {"eq": 0}) and _cmp("a", ["a", "b"]) and _cmp("a", "a")
    assert _cmp(5, {"gte": 2, "lt": 6}) and not _cmp(6, {"gte": 2, "lt": 6})


def test_filters_by_count_and_event():
    rs = [hh("a", n_job_changes=2, events=[(2024, "new_child")]),
          hh("b", n_job_changes=0, events=[(2019, "new_child")])]
    assert [r["owner_id"] for r in apply_filters(rs, {"n_job_changes": {"gte": 2}})] == ["a"]
    assert [r["owner_id"] for r in apply_filters(rs, {"had_event": {"event": "new_child", "since_year": 2024}})] == ["a"]
    assert len(apply_filters(rs, {"had_event": {"event": "new_child"}})) == 2
    with pytest.raises(ValueError):
        apply_filters(rs, {"bogus": 1})


def test_spend_around_event_aligns_per_household():
    # A: event 2020, before (2018-19) avg 100, after (2020-21) avg 200 -> +100%
    # B: event 2015, before (2013-14) avg 400, after (2015-16) avg 200 -> -50%
    a = hh("a", events=[(2020, "new_child")], spend=[(2018, 100), (2019, 100), (2020, 200), (2021, 200)])
    b = hh("b", events=[(2015, "new_child")], spend=[(2013, 400), (2014, 400), (2015, 200), (2016, 200)])
    g = run_query([a, b], {"metric": "spend_around_event", "params": {"event": "new_child"}})["groups"]["all"]
    assert g["n_events_scored"] == 2
    assert g["mean_annual_spend_before_usd"] == 250 and g["mean_annual_spend_after_usd"] == 200
    assert g["delta_pct_of_means"] == -20.0            # (200-250)/250
    assert g["median_per_household_delta_pct"] == 25.0  # median of (+100, -50)
    assert g["share_of_events_with_increase_pct"] == 50.0
    # a calendar-year computation (2020 vs 2018) would give a different answer — that's the trap


def test_price_per_purchase_denominators():
    a = hh("a", purchases=[(2020, "sedan", "gasoline", 10000), (2022, "sedan", "gasoline", 10000), (2024, "sedan", "gasoline", 10000)])
    b = hh("b", purchases=[(2021, "suv", "gasoline", 40000)])
    g = run_query([a, b], {"metric": "price_per_purchase"})["groups"]["all"]
    assert g["n_purchases"] == 4 and g["n_households"] == 2
    assert g["mean_price_per_purchase_usd"] == 17500      # 70000 / 4
    assert g["mean_of_household_means_usd"] == 25000      # (10000 + 40000) / 2


def test_share_with_two_cohorts_via_also():
    rs = [hh("a", n_job_changes=2, ever_ev=True), hh("b", n_job_changes=3, ever_ev=False),
          hh("c", n_job_changes=0, ever_ev=False), hh("d", n_job_changes=0, ever_ev=False)]
    res = run_query(rs, {"filters": {"n_job_changes": {"gte": 2}}, "metric": "share", "params": {"of": {"ever_ev": True}},
                         "also": {"filters": {"n_job_changes": {"eq": 0}}, "metric": "share", "params": {"of": {"ever_ev": True}}}})
    assert res["groups"]["all"]["share_pct"] == 50.0
    assert res["answerable_part"]["groups"]["all"]["share_pct"] == 0.0


def test_sequence_requires_change_after_move():
    a = hh("a", events=[(2010, "moved"), (2018, "moved")], n_moves=2,
           residence=[(2000, "transit", "initial"), (2010, "transit", "moved"), (2018, "car_long", "moved")])
    b = hh("b", events=[(2010, "moved"), (2018, "moved")], n_moves=2,
           residence=[(2000, "transit", "initial"), (2010, "transit", "moved"), (2018, "transit", "moved")])
    c = hh("c", events=[(2010, "moved")], n_moves=1, residence=[(2000, "transit", "initial"), (2010, "car_long", "moved")])
    g = run_query([a, b, c], {"metric": "sequence", "params": {"event": "moved", "min_count": 2, "attribute": "commute_type"}})["groups"]["all"]
    assert g["n_households_with_min_events"] == 2 and g["n_changed_after"] == 1
    assert g["share_of_cohort_pct"] == 50.0 and g["share_of_total_pct"] == 33.3


def test_event_to_body_change_lift():
    a = hh("a", events=[(2015, "new_child")], purchases=[(2010, "sedan", "gasoline", 1), (2016, "suv", "gasoline", 1)])
    g = run_query([a], {"metric": "event_to_body_change", "params": {"target_bodies": ["suv", "minivan"], "window": 2}})["groups"]["all"]
    row = g["by_event"][0]
    assert row["event"] == "new_child" and row["held_before_pct"] == 0.0 and row["held_within_window_pct"] == 100.0


def test_group_by_and_spend_span():
    a = hh("a", stage="retired", spend=[(2020, 100), (2021, 900)])
    b = hh("b", stage="early_career", spend=[(2020, 100), (2021, 200)])
    res = run_query([a, b], {"group_by": "life_stage", "metric": "spend_span"})
    assert res["groups"]["retired"]["mean_range_usd"] == 800 and res["groups"]["early_career"]["mean_range_usd"] == 100
    with pytest.raises(ValueError):
        run_query([a], {"group_by": "owner_id", "metric": "spend_span"})


def test_unanswerable_with_partial_answer():
    a = hh("a", stage="retired")
    res = run_query([a], {"filters": {"life_stage": "retired"}, "metric": "unanswerable", "params": {"reason": "numeric_income"},
                          "also": {"filters": {"life_stage": "retired"}, "metric": "segment_profile"}})
    assert res["answerable"] is False and res["unanswerable_key"] == "numeric_income"
    assert res["unanswerable_reason"] == UNANSWERABLE["numeric_income"]
    assert res["answerable_part"]["groups"]["all"]["n_households"] == 1


def test_decade_explain_excludes_partial():
    full = hh("a", stage="empty_nester", age=58, decades={"40s": {"avg_annual_spend": 9000, "n_purchases": 2, "top_trigger": "promotion", "cluster": 2, "partial": False, "years_observed": 10},
                                                           "50s": {"avg_annual_spend": 5000, "n_purchases": 1, "top_trigger": None, "cluster": 1, "partial": True, "years_observed": 8}})
    res = run_query([full], {"metric": "decade_explain", "params": {"decades": ["40s", "50s"]}})["groups"]["all"]
    assert res["n_excluded_partial_history"] == 1 and res["n_with_full_decades"] == 0


DATA = Path(__file__).parent.parent / "pipeline" / "data" / "households.json"

@pytest.mark.skipif(not DATA.exists(), reason="dataset not generated")
def test_curated_queries_run_on_real_data():
    import json
    records = json.load(open(DATA))
    queries = json.load(open(Path(__file__).parent.parent / "eval" / "queries.json"))
    assert len(queries) == 10
    for q in queries:
        res = run_query(records, q["spec"])
        assert res["n_matched"] > 0, q["id"]
        if q["id"] in ("Q09", "Q10"):
            assert res["answerable"] is False and "answerable_part" in res
        else:
            assert res["answerable"] and res["groups"]
