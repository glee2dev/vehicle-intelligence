"""
Deterministic query engine over the vehicle-ownership dataset.

    QuerySpec  ->  apply_filters()  ->  [group_by]  ->  metric  ->  StatsResult

No LLM anywhere in this file. Every number the narrator is allowed to say comes from here.
Every result carries explicit denominators (n_households / n_events / n_purchases) because
the denominator is exactly what LLMs get wrong when they do the math themselves.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any, Callable

CURRENT_YEAR = 2026
FAMILY_BODIES = {"suv", "minivan"}

# ── Filterable fields ─────────────────────────────────────────────────────────
DEMOGRAPHIC_FIELDS = {
    "age_group": lambda r: r["age_group"],
    "gender": lambda r: r["gender"],
    "life_stage": lambda r: r["demographics"]["life_stage"],
    "household_composition": lambda r: r["demographics"]["household_composition"],
    "income_bracket": lambda r: r["demographics"]["income_bracket"],
    "area_type": lambda r: r["demographics"]["area_type"],
    "commute_type": lambda r: r["demographics"]["commute_type"],
    "parking": lambda r: r["demographics"]["parking"],
    "dual_income": lambda r: r["demographics"]["dual_income"],
    "ever_married": lambda r: r["demographics"]["ever_married"],
    "primary_body_type": lambda r: r["ownership_intelligence"]["primary_body_type"],
    "primary_powertrain": lambda r: r["ownership_intelligence"]["primary_powertrain"],
    "ever_ev": lambda r: r["ownership_intelligence"]["ever_ev"],
    "rfm_cluster": lambda r: r["lifecycle_intelligence"]["rfm_cluster"],
    "progression_type": lambda r: r["lifecycle_intelligence"]["progression_type"],
}
COUNT_FIELDS = {
    "n_job_changes": lambda r: r["temporal_cross_intelligence"]["n_job_changes"],
    "n_moves": lambda r: r["temporal_cross_intelligence"]["n_moves"],
    "n_children": lambda r: r["temporal_cross_intelligence"]["n_children"],
    "n_children_home": lambda r: r["demographics"]["n_children_home"],
    "n_purchases": lambda r: r["ownership_intelligence"]["n_purchases"],
    "n_life_events": lambda r: r["temporal_cross_intelligence"]["n_life_events"],
    "current_vehicle_count": lambda r: r["ownership_intelligence"]["current_vehicle_count"],
    "age": lambda r: r["age"],
}
GROUPABLE = set(DEMOGRAPHIC_FIELDS) | {"archetype"}

# Things people ask for that this dataset cannot answer. Kept as data so the pipeline
# can say "no" precisely rather than letting a narrator improvise.
UNANSWERABLE = {
    "numeric_income": "Income is recorded only as a bracket (lower / lower_middle / upper_middle / upper). "
                      "No dollar income exists in the dataset, so an average income cannot be computed.",
    "external_benchmark": "The dataset contains no external or national benchmark. Only within-dataset "
                          "figures can be reported; any comparison to a national rate would be invented.",
    "brand_or_model": "Vehicles are recorded by body type and powertrain only. No make, brand, or model exists.",
    "future": "The panel ends in 2026. Nothing in the data describes future behaviour.",
}


class UnanswerableError(Exception):
    def __init__(self, key: str):
        super().__init__(UNANSWERABLE[key])
        self.key = key


# ═════════════════════════════════════════════════════════════════════════════
# Filters
# ═════════════════════════════════════════════════════════════════════════════

def _cmp(value, cond) -> bool:
    if isinstance(cond, dict):
        ops = {"eq": lambda a, b: a == b, "gte": lambda a, b: a >= b, "lte": lambda a, b: a <= b,
               "gt": lambda a, b: a > b, "lt": lambda a, b: a < b, "in": lambda a, b: a in b}
        return all(ops[op](value, v) for op, v in cond.items())
    if isinstance(cond, list):
        return value in cond
    return value == cond


def _had_event(r: dict, spec: dict) -> bool:
    """spec: {"event": "new_child", "since_year": 2024, "before_year": ..., "min_count": 1}"""
    ev = [e for e in r["life_timeline"] if e["event"] == spec["event"]]
    if "since_year" in spec:
        ev = [e for e in ev if e["year"] >= spec["since_year"]]
    if "before_year" in spec:
        ev = [e for e in ev if e["year"] < spec["before_year"]]
    if "min_age" in spec:
        ev = [e for e in ev if e["age"] >= spec["min_age"]]
    return len(ev) >= spec.get("min_count", 1)


def _has_full_decades(r: dict, decades: list[str]) -> bool:
    dp = r["lifecycle_intelligence"]["decade_profile"]
    return all(d in dp and not dp[d]["partial"] for d in decades)


def apply_filters(records: list[dict], filters: dict) -> list[dict]:
    preds: list[Callable[[dict], bool]] = []
    for key, cond in (filters or {}).items():
        if key in DEMOGRAPHIC_FIELDS:
            f = DEMOGRAPHIC_FIELDS[key]
            preds.append(lambda r, f=f, c=cond: _cmp(f(r), c))
        elif key in COUNT_FIELDS:
            f = COUNT_FIELDS[key]
            preds.append(lambda r, f=f, c=cond: _cmp(f(r), c))
        elif key == "had_event":
            specs = cond if isinstance(cond, list) else [cond]
            for s in specs:
                preds.append(lambda r, s=s: _had_event(r, s))
        elif key == "never_had_event":
            evs = cond if isinstance(cond, list) else [cond]
            preds.append(lambda r, evs=evs: not any(e["event"] in evs for e in r["life_timeline"]))
        elif key == "has_full_decades":
            preds.append(lambda r, d=cond: _has_full_decades(r, d))
        elif key == "archetype":
            preds.append(lambda r, c=cond: _cmp(r["archetype"], c))
        else:
            raise ValueError(f"Unknown filter: {key}")
    return [r for r in records if all(p(r) for p in preds)]


def group_records(records: list[dict], group_by: str | None) -> dict[str, list[dict]]:
    if not group_by:
        return {"all": records}
    if group_by not in GROUPABLE:
        raise ValueError(f"Cannot group by {group_by}")
    f = DEMOGRAPHIC_FIELDS.get(group_by, lambda r: r["archetype"])
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        out[str(f(r))].append(r)
    return dict(sorted(out.items()))


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _mean(xs, nd=1):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), nd) if xs else None


def _median(xs, nd=1):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), nd) if xs else None


def _pct(a, b, nd=1):
    return round(a / b * 100, nd) if b else None


def _dist(xs, top=None):
    c = Counter(xs)
    items = c.most_common(top) if top else sorted(c.items())
    return dict(items)


def _purchases(r):
    return [t for t in r["vehicle_timeline"]["transactions"] if t["type"] in ("purchase", "lease")]


def _spend_window(r, year, lo, hi, col=3):
    vals = [row[col] for row in r["annual_spend"] if year + lo <= row[0] <= year + hi]
    return statistics.mean(vals) if vals else None


def _held_body(r, year, bodies):
    return any(v["body_type"] in bodies and v["acquired_year"] <= year and
               (v["sold_year"] is None or v["sold_year"] > year) for v in r["vehicle_timeline"]["vehicles"])


# ═════════════════════════════════════════════════════════════════════════════
# Metrics — each takes (records, params) and returns a dict with explicit denominators
# ═════════════════════════════════════════════════════════════════════════════

def m_segment_profile(rs: list[dict], p: dict) -> dict:
    n = len(rs)
    if not n:
        return {"n_households": 0}
    o = [r["ownership_intelligence"] for r in rs]
    return {
        "n_households": n,
        "avg_age": _mean([r["age"] for r in rs]),
        "age_group": _dist(r["age_group"] for r in rs),
        "life_stage": _dist(r["demographics"]["life_stage"] for r in rs),
        "household_composition": _dist(r["demographics"]["household_composition"] for r in rs),
        "income_bracket": _dist(r["demographics"]["income_bracket"] for r in rs),
        "area_type": _dist(r["demographics"]["area_type"] for r in rs),
        "commute_type": _dist(r["demographics"]["commute_type"] for r in rs),
        "primary_body_type": _dist(x["primary_body_type"] for x in o),
        "primary_powertrain": _dist(x["primary_powertrain"] for x in o),
        "ever_ev_pct": _pct(sum(1 for x in o if x["ever_ev"]), n),
        "avg_current_vehicles": _mean([x["current_vehicle_count"] for x in o], 2),
        "avg_purchases_lifetime": _mean([x["n_purchases"] for x in o], 2),
        "avg_price_per_purchase_usd": _mean([x["avg_price_per_purchase_usd"] for x in o if x["n_purchases"]], 0),
        "avg_annual_spend_5y_usd": _mean([x["avg_annual_spend_5y"] for x in o], 0),
        "rfm_cluster": _dist(r["lifecycle_intelligence"]["rfm_cluster"] for r in rs),
        "progression_type": _dist(r["lifecycle_intelligence"]["progression_type"] for r in rs),
        "top_trigger_events": _dist((t for x in o for t in x["top_trigger_events"]), top=5),
        "avg_job_changes": _mean([r["temporal_cross_intelligence"]["n_job_changes"] for r in rs], 2),
        "avg_moves": _mean([r["temporal_cross_intelligence"]["n_moves"] for r in rs], 2),
    }


def m_spend_around_event(rs: list[dict], p: dict) -> dict:
    """Per household: mean annual spend in [before] years before vs [after] years after the event.
    Alignment is per household-event, then averaged. This is the operation naive LLMs botch."""
    event, before, after = p["event"], p.get("before", 2), p.get("after", 2)
    since = p.get("since_year")
    col = 1 if p.get("cost") == "running" else 3
    b_vals, a_vals, deltas, n_events, n_hh = [], [], [], 0, 0
    for r in rs:
        evs = [e for e in r["life_timeline"] if e["event"] == event and (since is None or e["year"] >= since)]
        if not evs:
            continue
        n_hh += 1
        for e in evs:
            b = _spend_window(r, e["year"], -before, -1, col)
            a = _spend_window(r, e["year"], 0, after - 1, col)
            if b is None or a is None or b == 0:
                continue
            n_events += 1
            b_vals.append(b); a_vals.append(a); deltas.append((a - b) / b * 100)
    mb, ma = _mean(b_vals, 0), _mean(a_vals, 0)
    return {
        "event": event, "window_before_years": before, "window_after_years": after,
        "cost_basis": "running" if col == 1 else "total (running + capital)",
        "n_households_with_event": n_hh, "n_events_scored": n_events,
        "mean_annual_spend_before_usd": mb, "mean_annual_spend_after_usd": ma,
        "delta_pct_of_means": _pct(ma - mb, mb) if mb else None,
        "median_per_household_delta_pct": _median(deltas),
        "share_of_events_with_increase_pct": _pct(sum(1 for d in deltas if d > 0), n_events),
    }


def m_event_to_body_change(rs: list[dict], p: dict) -> dict:
    """For each life event type: share of occurrences where the household held a target body type
    within [window] years after, vs the year before. Returns a table sorted by lift."""
    bodies = set(p.get("target_bodies", FAMILY_BODIES))
    window = p.get("window", 2)
    events = p.get("events")  # None = all
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # n, before, after
    for r in rs:
        for e in r["life_timeline"]:
            if e["event"] in ("first_job", "graduation") or (events and e["event"] not in events):
                continue
            a = agg[e["event"]]
            a[0] += 1
            if _held_body(r, e["year"] - 1, bodies):
                a[1] += 1
            if any(_held_body(r, e["year"] + k, bodies) for k in range(0, window + 1)):
                a[2] += 1
    rows = []
    for ev, (n, b, a) in agg.items():
        rows.append({"event": ev, "n_events": n,
                     "held_before_pct": _pct(b, n), "held_within_window_pct": _pct(a, n),
                     "lift_pts": round(_pct(a, n) - _pct(b, n), 1) if n else None})
    rows.sort(key=lambda x: -(x["lift_pts"] or -999))
    return {"target_bodies": sorted(bodies), "window_years": window,
            "population_holding_now_pct": _pct(sum(1 for r in rs if r["ownership_intelligence"]["primary_body_type"] in bodies), len(rs)),
            "by_event": rows}


def m_share(rs: list[dict], p: dict) -> dict:
    """Share of households satisfying a predicate (field == value, or a filter dict)."""
    sub = apply_filters(rs, p["of"])
    return {"n_households": len(rs), "n_matching": len(sub), "share_pct": _pct(len(sub), len(rs)),
            "of": p["of"]}


def m_price_per_purchase(rs: list[dict], p: dict) -> dict:
    """Reports BOTH the per-purchase mean (denominator = purchases) and the per-household mean
    (denominator = households). They differ; the question decides which is right."""
    since = p.get("since_year")
    prices = [t["price_usd"] for r in rs for t in _purchases(r) if since is None or t["year"] >= since]
    hh_means = [statistics.mean([t["price_usd"] for t in _purchases(r) if since is None or t["year"] >= since] or [0])
                for r in rs if any(since is None or t["year"] >= since for t in _purchases(r))]
    return {
        "n_households": len(rs), "n_purchases": len(prices),
        "mean_price_per_purchase_usd": _mean(prices, 0), "median_price_per_purchase_usd": _median(prices, 0),
        "mean_of_household_means_usd": _mean(hh_means, 0),
        "avg_purchases_per_household": _mean([len(_purchases(r)) for r in rs], 2),
        "price_band_distribution": _dist(t["price_band"] for r in rs for t in _purchases(r)),
        "new_share_pct": _pct(sum(1 for r in rs for t in _purchases(r) if t["new_or_used"] == "new"), len(prices)),
    }


def m_spend_span(rs: list[dict], p: dict) -> dict:
    spans = [r["ownership_intelligence"]["spend_span"] for r in rs]
    return {"n_households": len(rs),
            "mean_range_usd": _mean([s["range"] for s in spans], 0),
            "median_range_usd": _median([s["range"] for s in spans], 0),
            "mean_min_usd": _mean([s["min"] for s in spans], 0),
            "mean_max_usd": _mean([s["max"] for s in spans], 0),
            "mean_lifetime_spend_usd": _mean([r["ownership_intelligence"]["lifetime_spend"] for r in rs], 0)}


def m_sequence(rs: list[dict], p: dict) -> dict:
    """Households with >= [min_count] of [event] where an attribute changed after (any) such event.
    attribute in {commute_type, area_type, primary_powertrain, primary_body_type}."""
    event, min_count, attr = p["event"], p.get("min_count", 2), p["attribute"]
    n_base = n_changed = 0
    for r in rs:
        evs = [e for e in r["life_timeline"] if e["event"] == event]
        if len(evs) < min_count:
            continue
        n_base += 1
        changed = False
        if attr in ("commute_type", "area_type"):
            periods = r["residence_timeline"]
            for e in evs:
                before = [x for x in periods if x["from_year"] < e["year"]]
                after = [x for x in periods if x["from_year"] >= e["year"]]
                if before and after and after[0][attr] != before[-1][attr]:
                    changed = True; break
        else:
            key = "powertrain" if attr == "primary_powertrain" else "body_type"
            pur = _purchases(r)
            for e in evs:
                before = [t for t in pur if t["year"] < e["year"]]
                after = [t for t in pur if t["year"] >= e["year"]]
                if before and after and after[0][key] != before[-1][key]:
                    changed = True; break
        n_changed += changed
    return {"event": event, "min_count": min_count, "attribute": attr,
            "n_households_total": len(rs), "n_households_with_min_events": n_base,
            "n_changed_after": n_changed, "share_of_cohort_pct": _pct(n_changed, n_base),
            "share_of_total_pct": _pct(n_changed, len(rs))}


def m_purchases_after_event(rs: list[dict], p: dict) -> dict:
    """What gets bought within [years] after [event]. Denominator = purchases in window."""
    event, years = p["event"], p.get("years", 3)
    tx = []
    n_hh = 0
    for r in rs:
        evs = [e for e in r["life_timeline"] if e["event"] == event]
        if not evs:
            continue
        n_hh += 1
        for e in evs:
            tx.extend(t for t in _purchases(r) if 0 <= t["year"] - e["year"] <= years)
    return {"event": event, "window_years": years, "n_households_with_event": n_hh,
            "n_purchases_in_window": len(tx), "purchases_per_household": _mean([len(tx) / n_hh] if n_hh else [], 2),
            "body_type": _dist(t["body_type"] for t in tx), "powertrain": _dist(t["powertrain"] for t in tx),
            "new_or_used": _dist(t["new_or_used"] for t in tx), "price_band": _dist(t["price_band"] for t in tx),
            "mean_price_usd": _mean([t["price_usd"] for t in tx], 0)}


def m_decade_explain(rs: list[dict], p: dict) -> dict:
    """The 'why' metric: decade clusters + what happened in each decade. Uses precomputed
    cross-intelligence; excludes partial decades and reports the exclusion."""
    decades = p.get("decades", ["40s", "50s"])
    full = [r for r in rs if _has_full_decades(r, decades)]
    out = {"decades": decades, "n_households_total": len(rs), "n_with_full_decades": len(full),
           "n_excluded_partial_history": len(rs) - len(full), "per_decade": {}}
    for d in decades:
        lo = int(d[:-1])
        profs = [r["lifecycle_intelligence"]["decade_profile"][d] for r in full]
        ev_c = Counter(e["event"] for r in full for e in r["life_timeline"] if lo <= e["age"] <= lo + 9
                       and e["event"] not in ("first_job", "graduation"))
        share_ev = {ev: _pct(sum(1 for r in full if any(x["event"] == ev and lo <= x["age"] <= lo + 9 for x in r["life_timeline"])), len(full))
                    for ev in ("promotion", "new_child", "empty_nest", "retirement", "job_change", "new_hobby", "moved")}
        out["per_decade"][d] = {
            "mean_cluster_0_to_2": _mean([x["cluster"] for x in profs], 2),
            "cluster_distribution": _dist(x["cluster"] for x in profs),
            "mean_annual_spend_usd": _mean([x["avg_annual_spend"] for x in profs], 0),
            "mean_purchases": _mean([x["n_purchases"] for x in profs], 2),
            "top_purchase_trigger": _dist((x["top_trigger"] for x in profs if x["top_trigger"]), top=3),
            "share_of_households_with_event_in_decade_pct": share_ev,
            "event_counts": dict(ev_c.most_common(6)),
        }
    return out


def m_unanswerable(rs: list[dict], p: dict) -> dict:
    raise UnanswerableError(p["reason"])


METRICS: dict[str, Callable[[list[dict], dict], dict]] = {
    "segment_profile": m_segment_profile,
    "spend_around_event": m_spend_around_event,
    "event_to_body_change": m_event_to_body_change,
    "share": m_share,
    "price_per_purchase": m_price_per_purchase,
    "spend_span": m_spend_span,
    "sequence": m_sequence,
    "purchases_after_event": m_purchases_after_event,
    "decade_explain": m_decade_explain,
    "unanswerable": m_unanswerable,
}


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def run_query(records: list[dict], spec: dict) -> dict:
    """
    spec = {
      "filters": {...},                  # see apply_filters
      "group_by": "life_stage" | None,
      "metric": "spend_around_event",
      "params": {...},
      "also": {"answerable_part": {...}} # optional: for partially answerable questions
    }
    """
    metric = spec.get("metric", "segment_profile")
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric}. Valid: {sorted(METRICS)}")
    matched = apply_filters(records, spec.get("filters", {}))
    result: dict[str, Any] = {
        "metric": metric, "filters": spec.get("filters", {}), "group_by": spec.get("group_by"),
        "params": spec.get("params", {}),
        "n_total": len(records), "n_matched": len(matched), "match_pct": _pct(len(matched), len(records)),
        "answerable": True, "unanswerable_reason": None, "groups": {},
    }
    try:
        for name, rs in group_records(matched, spec.get("group_by")).items():
            result["groups"][name] = METRICS[metric](rs, spec.get("params", {}))
    except UnanswerableError as e:
        result["answerable"] = False
        result["unanswerable_reason"] = str(e)
        result["unanswerable_key"] = e.key
    if "also" in spec:   # partial answer: run the part that IS answerable
        result["answerable_part"] = run_query(records, spec["also"])
    return result
