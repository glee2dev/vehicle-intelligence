#!/usr/bin/env python3
r"""
Timeline-first synthetic vehicle-ownership dataset generator.

Every field in a record is derived from ONE life timeline:

    life_timeline  ->  career_timeline  ->  residence_timeline  ->  household_timeline
                                    \                 |                  /
                                     -->   vehicle_timeline (transactions w/ trigger_event)
                                                      |
                                               annual_spend (per year)
                                                      |
                              derived intelligence (decade clusters, RFM, temporal cross-intelligence)

Output: pipeline/data/households.json  (10,000 records, one owner/household each)
Run:    python tools/generate_dataset.py [--n 10000] [--seed 42]
"""
import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CURRENT_YEAR = 2026
BRACKETS = ["lower", "lower_middle", "upper_middle", "upper"]
AREAS = ["urban", "suburban", "rural"]
BODY_TYPES = ["compact", "sedan", "crossover", "suv", "minivan", "pickup", "sports", "motorcycle"]
FAMILY_BODIES = {"suv", "minivan"}
HOBBY_BODIES = {"sports", "motorcycle", "pickup"}

# ── Archetypes: (age range, weight). They shape the *sequence* of life events.
ARCHETYPES = {
    "fresh_graduate":       ((24, 29), 0.12),
    "early_career_single":  ((27, 35), 0.13),
    "dual_income_parents":  ((31, 47), 0.22),
    "single_hobbyist":      ((38, 52), 0.11),
    "mid_career_changer":   ((36, 50), 0.10),
    "divorced_parent":      ((40, 56), 0.08),
    "empty_nester":         ((50, 63), 0.13),
    "retiree":              ((63, 75), 0.11),
}

PRICE_BAND_USD = {1: 9000, 2: 18000, 3: 30000, 4: 48000, 5: 75000}


def choice_w(pairs):
    items, weights = zip(*pairs)
    return random.choices(items, weights=weights, k=1)[0]


def spread_ages(n, lo, hi, min_gap=2):
    """Pick n distinct ages in [lo, hi] with a minimum gap; returns sorted list."""
    if n <= 0 or hi < lo:
        return []
    picks = []
    tries = 0
    while len(picks) < n and tries < 200:
        a = random.randint(lo, hi)
        if all(abs(a - p) >= min_gap for p in picks):
            picks.append(a)
        tries += 1
    return sorted(picks)


# ═══════════════════════════════════════════════════════════════════════════
# 1. LIFE TIMELINE
# ═══════════════════════════════════════════════════════════════════════════

def build_life_timeline(arch: str, age: int, birth_year: int) -> list[dict]:
    ev: list[dict] = []

    def add(a, event, **attrs):
        if 18 <= a <= age:
            ev.append({"age": a, "year": birth_year + a, "event": event, **attrs})

    # education / first job
    if random.random() < 0.85:
        grad = min(random.choice([21, 22, 22, 23, 24]), age - 1)
        add(grad, "graduation")
        add(min(grad + random.choice([0, 0, 1, 2]), age), "first_job")
    else:
        add(random.choice([18, 19, 20]), "first_job")
    first_job_age = next(e["age"] for e in ev if e["event"] == "first_job")

    # career events -------------------------------------------------------
    jc_range = {
        "fresh_graduate": (0, 1), "early_career_single": (1, 2), "dual_income_parents": (1, 3),
        "single_hobbyist": (2, 4), "mid_career_changer": (2, 3), "divorced_parent": (1, 3),
        "empty_nester": (1, 3), "retiree": (1, 3),
    }[arch]
    career_end = min(age, 58)
    for a in spread_ages(random.randint(*jc_range), first_job_age + 2, career_end, 3):
        add(a, "job_change")

    promo_range = {
        "fresh_graduate": (0, 1), "early_career_single": (0, 2), "dual_income_parents": (1, 3),
        "single_hobbyist": (1, 3), "mid_career_changer": (1, 2), "divorced_parent": (1, 2),
        "empty_nester": (1, 3), "retiree": (1, 3),
    }[arch]
    for a in spread_ages(random.randint(*promo_range), first_job_age + 3, career_end, 3):
        add(a, "promotion")

    if arch == "mid_career_changer":
        add(random.randint(36, min(45, age)), "career_change")
    elif random.random() < 0.05 and age > 32:
        add(random.randint(30, min(45, age)), "career_change")

    # relationship / children --------------------------------------------
    marriage_p = {
        "fresh_graduate": 0.05, "early_career_single": 0.15, "dual_income_parents": 0.95,
        "single_hobbyist": 0.0, "mid_career_changer": 0.6, "divorced_parent": 1.0,
        "empty_nester": 0.95, "retiree": 0.9,
    }[arch]
    marriage_age = None
    if random.random() < marriage_p:
        marriage_age = random.randint(24, 32)
        add(marriage_age, "marriage")

    child_n = 0
    if marriage_age is not None:
        child_n = {
            "dual_income_parents": random.choice([1, 2, 2, 3]),
            "divorced_parent": random.choice([1, 2, 2]),
            "empty_nester": random.choice([1, 2, 2, 3]),
            "retiree": random.choice([1, 2, 2, 3]),
            "mid_career_changer": random.choice([0, 1, 2]),
            "early_career_single": random.choice([0, 0, 1]),
        }.get(arch, 0)
    child_ages = []
    next_a = (marriage_age or 28) + random.randint(1, 3)
    for _ in range(child_n):
        if next_a <= age:
            add(next_a, "new_child", child_index=len(child_ages) + 1)
            child_ages.append(next_a)
            next_a += random.randint(2, 4)

    if arch == "divorced_parent":
        add(random.randint(38, min(50, age)), "divorce")
    elif marriage_age and random.random() < 0.06 and age > 40:
        add(random.randint(36, min(50, age)), "divorce")

    if child_ages:
        empty_a = child_ages[-1] + random.randint(19, 22)
        if arch in ("empty_nester", "retiree") or random.random() < 0.7:
            add(empty_a, "empty_nest")

    if arch == "retiree":
        add(random.randint(60, min(67, age)), "retirement")
    if age >= 60 and random.random() < (0.3 if arch == "retiree" else 0.1):
        add(random.randint(58, age), "bereavement")

    # hobbies --------------------------------------------------------------
    if arch == "single_hobbyist":
        for a in spread_ages(random.randint(1, 2), 32, min(46, age), 4):
            add(a, "new_hobby")
    elif random.random() < 0.10 and age > 30:
        add(random.randint(28, min(50, age)), "new_hobby")

    # moves ------------------------------------------------------------------
    # Some moves are *caused* by other events (child, divorce, retirement, career change).
    triggered_moves = []
    for e in list(ev):
        if e["event"] == "new_child" and e.get("child_index") == 1 and random.random() < 0.7:
            triggered_moves.append((e["age"] + random.choice([0, 1, 1]), "new_child"))
        if e["event"] == "divorce":
            triggered_moves.append((e["age"] + random.choice([0, 0, 1]), "divorce"))
        if e["event"] == "retirement" and random.random() < 0.5:
            triggered_moves.append((e["age"] + random.choice([0, 1, 2]), "retirement"))
        if e["event"] == "career_change" and random.random() < 0.6:
            triggered_moves.append((e["age"] + random.choice([0, 0, 1]), "career_change"))
        if e["event"] == "job_change" and random.random() < 0.3:
            triggered_moves.append((e["age"], "job_change"))
    free_moves = {
        "fresh_graduate": (1, 2), "early_career_single": (1, 2), "dual_income_parents": (0, 1),
        "single_hobbyist": (1, 2), "mid_career_changer": (0, 1), "divorced_parent": (0, 1),
        "empty_nester": (0, 1), "retiree": (0, 1),
    }[arch]
    used = set()
    for a, cause in triggered_moves:
        if a <= age and a not in used:
            add(a, "moved", cause=cause)
            used.add(a)
    for a in spread_ages(random.randint(*free_moves), 18, age, 3):
        if a not in used:
            add(a, "moved", cause="unspecified")
            used.add(a)

    ev.sort(key=lambda e: (e["age"], e["event"]))
    return ev


# ═══════════════════════════════════════════════════════════════════════════
# 2. CAREER TIMELINE  (income bracket per year)
# ═══════════════════════════════════════════════════════════════════════════

def build_career_timeline(life: list[dict], age: int, birth_year: int) -> dict:
    first_job = next(e for e in life if e["event"] == "first_job")
    idx = 0
    income_by_age: dict[int, int] = {}
    pending_recover = None
    for a in range(18, age + 1):
        for e in life:
            if e["age"] != a:
                continue
            if e["event"] == "promotion":
                idx = min(3, idx + 1)
            elif e["event"] == "career_change":
                idx = max(0, idx - 1)
                pending_recover = a + 2
            elif e["event"] == "retirement":
                idx = max(0, idx - 1)
        if pending_recover == a:
            idx = min(3, idx + 2)
            pending_recover = None
        # slow drift: tenure pushes people out of "lower" by mid-30s
        if a == 34 and idx == 0 and random.random() < 0.7:
            idx = 1
        income_by_age[a] = idx if a >= first_job["age"] else 0

    changes = []
    last = None
    for a, i in income_by_age.items():
        if i != last:
            changes.append([birth_year + a, BRACKETS[i]])
            last = i
    return {
        "first_job_year": first_job["year"],
        "job_change_years": [e["year"] for e in life if e["event"] == "job_change"],
        "promotion_years": [e["year"] for e in life if e["event"] == "promotion"],
        "career_change_years": [e["year"] for e in life if e["event"] == "career_change"],
        "retirement_year": next((e["year"] for e in life if e["event"] == "retirement"), None),
        "income_bracket_changes": changes,       # [[year, bracket], ...] change points
        "_income_by_age": income_by_age,         # stripped before output
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. RESIDENCE TIMELINE (area / commute / parking per period)
# ═══════════════════════════════════════════════════════════════════════════

def pick_area(cause: str, prev: str, age: int) -> str:
    if cause == "new_child":
        return choice_w([("suburban", 0.75), ("rural", 0.10), ("urban", 0.15)])
    if cause == "initial":
        pass
    if cause == "retirement":
        return choice_w([("suburban", 0.45), ("rural", 0.40), ("urban", 0.15)])
    if cause in ("job_change", "career_change"):
        return choice_w([("urban", 0.5), ("suburban", 0.4), ("rural", 0.1)])
    if cause == "divorce":
        return choice_w([("urban", 0.45), ("suburban", 0.45), ("rural", 0.1)])
    if age < 26:
        return choice_w([("urban", 0.55), ("suburban", 0.35), ("rural", 0.10)])
    return choice_w([("urban", 0.35), ("suburban", 0.50), ("rural", 0.15)])


def pick_commute(area: str, year: int, retired: bool) -> str:
    if retired:
        return "none"
    wfh = 0.25 if year >= 2020 else 0.05
    if area == "urban":
        return choice_w([("transit", 0.45), ("car_short", 0.35), ("wfh", wfh), ("car_long", 0.05)])
    if area == "suburban":
        return choice_w([("car_short", 0.45), ("car_long", 0.35), ("wfh", wfh), ("transit", 0.08)])
    return choice_w([("car_long", 0.70), ("car_short", 0.20), ("wfh", wfh)])


def pick_parking(area: str) -> str:
    return {"urban": choice_w([("street", 0.6), ("garage", 0.4)]),
            "suburban": choice_w([("garage", 0.8), ("street", 0.2)]),
            "rural": choice_w([("garage", 0.92), ("street", 0.08)])}[area]


def build_residence_timeline(life: list[dict], age: int, birth_year: int) -> list[dict]:
    retire_age = next((e["age"] for e in life if e["event"] == "retirement"), 999)
    area = pick_area("none", None, 18)
    periods = [{"from_year": birth_year + 18, "age": 18, "area_type": area,
                "commute_type": pick_commute(area, birth_year + 18, False),
                "parking": pick_parking(area), "trigger_event": "initial"}]
    for e in life:
        if e["event"] == "moved":
            area = pick_area(e["cause"], area, e["age"])
            periods.append({"from_year": e["year"], "age": e["age"], "area_type": area,
                            "commute_type": pick_commute(area, e["year"], e["age"] >= retire_age),
                            "parking": pick_parking(area), "trigger_event": e["cause"]})
        elif e["event"] == "retirement":
            periods.append({"from_year": e["year"], "age": e["age"], "area_type": area,
                            "commute_type": "none", "parking": periods[-1]["parking"],
                            "trigger_event": "retirement"})
        elif e["event"] in ("job_change", "career_change") and random.random() < 0.5:
            periods.append({"from_year": e["year"], "age": e["age"], "area_type": area,
                            "commute_type": pick_commute(area, e["year"], False),
                            "parking": periods[-1]["parking"], "trigger_event": e["event"]})
    return periods


def residence_at(periods: list[dict], age: int) -> dict:
    cur = periods[0]
    for p in periods:
        if p["age"] <= age:
            cur = p
    return cur


# ═══════════════════════════════════════════════════════════════════════════
# 4. HOUSEHOLD TIMELINE
# ═══════════════════════════════════════════════════════════════════════════

def build_household_timeline(life: list[dict], age: int, birth_year: int) -> dict:
    m = next((e for e in life if e["event"] == "marriage"), None)
    d = next((e for e in life if e["event"] == "divorce"), None)
    kids = [e["year"] for e in life if e["event"] == "new_child"]
    en = next((e for e in life if e["event"] == "empty_nest"), None)
    partnered = m is not None and d is None
    return {
        "marriage_year": m["year"] if m else None,
        "divorce_year": d["year"] if d else None,
        "children_birth_years": kids,
        "empty_nest_year": en["year"] if en else None,
        "dual_income": bool(partnered and random.random() < 0.75),
        "_partnered": partnered,
    }


def household_state(hh: dict, life: list[dict], age: int, birth_year: int) -> dict:
    """Snapshot of the household at a given age."""
    year = birth_year + age
    partnered = hh["marriage_year"] is not None and hh["marriage_year"] <= year and \
                not (hh["divorce_year"] and hh["divorce_year"] <= year)
    kids_home = [y for y in hh["children_birth_years"] if y <= year and year - y < 19]
    return {"partnered": partnered, "n_children_home": len(kids_home),
            "youngest_child_age": (year - max(kids_home)) if kids_home else None}


# ═══════════════════════════════════════════════════════════════════════════
# 5. VEHICLE TIMELINE  (transactions emitted by events)
# ═══════════════════════════════════════════════════════════════════════════

def powertrain_for(year: int, commute: str, parking: str, income_idx: int, trigger: str) -> str:
    if year < 2005:
        return choice_w([("gasoline", 0.9), ("diesel", 0.1)])
    if year < 2012:
        return choice_w([("gasoline", 0.82), ("diesel", 0.08), ("hybrid", 0.10)])
    if year < 2018:
        ev, hy = 0.03, 0.15
    elif year < 2022:
        ev, hy = 0.10, 0.20
    else:
        ev, hy = 0.25, 0.25
    if commute == "car_long" and trigger in ("job_change", "career_change") and year >= 2016:
        ev *= 2.2
    if parking == "street":
        ev *= 0.4
    if income_idx == 3:
        ev *= 1.5
    ev = min(ev, 0.6)
    return choice_w([("ev", ev), ("hybrid", hy), ("gasoline", 1 - ev - hy)])


def body_for(trigger: str, state: dict, income_idx: int, first_car: bool) -> str:
    if trigger == "new_child":
        if state["n_children_home"] >= 2:
            return choice_w([("minivan", 0.45), ("suv", 0.45), ("crossover", 0.10)])
        return choice_w([("suv", 0.6), ("minivan", 0.2), ("crossover", 0.2)])
    if trigger == "new_hobby":
        return choice_w([("sports", 0.5), ("motorcycle", 0.3), ("pickup", 0.2)])
    if trigger in ("empty_nest", "retirement"):
        return choice_w([("sedan", 0.45), ("crossover", 0.40), ("compact", 0.15)])
    if first_car:
        return choice_w([("compact", 0.6), ("sedan", 0.3), ("crossover", 0.1)])
    if state["n_children_home"] >= 1:
        return choice_w([("suv", 0.45), ("crossover", 0.3), ("minivan", 0.15), ("sedan", 0.1)])
    return {
        0: choice_w([("compact", 0.55), ("sedan", 0.35), ("crossover", 0.10)]),
        1: choice_w([("sedan", 0.45), ("crossover", 0.35), ("compact", 0.20)]),
        2: choice_w([("crossover", 0.4), ("suv", 0.3), ("sedan", 0.3)]),
        3: choice_w([("suv", 0.4), ("sedan", 0.3), ("sports", 0.15), ("crossover", 0.15)]),
    }[income_idx]


def price_band_for(income_idx: int, body: str, promo_recent: bool, used: bool, trigger: str = "") -> int:
    band = income_idx + 1 + (1 if promo_recent else 0) + (1 if body in ("sports", "suv") else 0)
    if trigger == "new_hobby":
        band += 1
    if used:
        band -= 1
    if body == "motorcycle":
        band = min(band, 2)
    return max(1, min(5, band))


def build_vehicle_timeline(life, career, residence, hh, age, birth_year) -> tuple[list[dict], list[dict]]:
    """Simulate year by year. Returns (transactions, vehicles_ever)."""
    transactions: list[dict] = []
    vehicles: list[dict] = []          # every vehicle ever owned
    active: list[dict] = []            # currently held
    vid = 0
    scheduled: dict[int, list[tuple[str, str]]] = defaultdict(list)  # age -> [(action, trigger)]
    first_job_age = next(e["age"] for e in life if e["event"] == "first_job")
    promo_ages = {e["age"] for e in life if e["event"] == "promotion"}

    # -- schedule event-driven actions -------------------------------------
    for e in life:
        a, t = e["age"], e["event"]
        if t == "first_job":
            scheduled[a + random.choice([0, 0, 1])].append(("acquire", "first_job"))
        elif t == "new_child":
            if e.get("child_index", 1) == 1 or random.random() < 0.65:
                scheduled[a + random.choice([0, 0, 1])].append(("replace_family", "new_child"))
        elif t == "promotion":
            if random.random() < 0.55:
                scheduled[a + random.choice([1, 1, 2])].append(("upgrade", "promotion"))
        elif t == "job_change":
            if random.random() < 0.35:
                scheduled[a + random.choice([0, 1])].append(("replace", "job_change"))
        elif t == "career_change":
            if random.random() < 0.45:
                scheduled[a + random.choice([0, 1])].append(("replace", "career_change"))
        elif t == "marriage":
            if random.random() < 0.55:
                scheduled[a + random.choice([0, 1])].append(("add_second", "marriage"))
        elif t == "new_hobby":
            if random.random() < 0.8:
                scheduled[a + random.choice([0, 0, 1])].append(("add_hobby", "new_hobby"))
        elif t == "moved":
            scheduled[a].append(("move_check", e["cause"]))
        elif t == "empty_nest":
            if random.random() < 0.6:
                scheduled[a + random.choice([0, 1, 2])].append(("downsize", "empty_nest"))
        elif t == "retirement":
            scheduled[a + random.choice([0, 1])].append(("downsize", "retirement"))
            if random.random() < 0.5:
                scheduled[a + random.choice([0, 1])].append(("sell_second", "retirement"))
        elif t == "divorce":
            scheduled[a + random.choice([0, 0, 1])].append(("sell_second", "divorce"))
        elif t == "bereavement":
            if random.random() < 0.5:
                scheduled[a + random.choice([1, 2])].append(("sell_second", "bereavement"))

    def add_tx(a, ttype, v, trigger, trigger_year):
        held = birth_year + a - v["acquired_year"]
        amount = v["price_usd"] if ttype in ("purchase", "lease") else int(v["price_usd"] * max(0.15, 0.6 - 0.05 * held))
        transactions.append({
            "year": birth_year + a, "age": a, "type": ttype, "vehicle_id": v["vehicle_id"],
            "body_type": v["body_type"], "powertrain": v["powertrain"],
            "price_band": v["price_band"], "price_usd": amount,
            "new_or_used": v["new_or_used"], "trigger_event": trigger, "trigger_year": trigger_year,
        })

    def acquire(a, trigger, body=None, used=None, role="primary"):
        nonlocal vid
        vid += 1
        res = residence_at(residence, a)
        inc = career["_income_by_age"].get(a, 0)
        state = household_state(hh, life, a, birth_year)
        promo_recent = any(0 <= a - p <= 2 for p in promo_ages)
        body = body or body_for(trigger, state, inc, first_car=(vid == 1))
        if used is None:
            used = random.random() < {0: 0.8, 1: 0.55, 2: 0.3, 3: 0.15}[inc]
        band = price_band_for(inc, body, promo_recent, used, trigger)
        price = int(PRICE_BAND_USD[band] * random.uniform(0.8, 1.2))
        v = {"vehicle_id": f"V{vid}", "body_type": body,
             "powertrain": powertrain_for(birth_year + a, res["commute_type"], res["parking"], inc, trigger),
             "price_band": band, "price_usd": price, "new_or_used": "used" if used else "new",
             "acquired_year": birth_year + a, "acquired_via": "lease" if (not used and random.random() < 0.25) else "purchase",
             "trigger_event": trigger, "role": role, "sold_year": None}
        vehicles.append(v)
        active.append(v)
        add_tx(a, v["acquired_via"], v, trigger, birth_year + a)
        return v

    def dispose(a, v, trigger, how=None):
        if v not in active:
            return
        active.remove(v)
        v["sold_year"] = birth_year + a
        add_tx(a, how or ("trade_in" if random.random() < 0.6 else "sale"), v, trigger, birth_year + a)

    def primary():
        return next((v for v in active if v["role"] == "primary"), active[0] if active else None)

    last_action_age = -10
    for a in range(18, age + 1):
        acts = scheduled.get(a, [])
        res = residence_at(residence, a)
        for action, trig in acts:
            if action == "acquire":
                if not active and not (res["area_type"] == "urban" and res["commute_type"] == "transit" and random.random() < 0.35):
                    acquire(a, trig); last_action_age = a
            elif action == "replace_family":
                p = primary()
                if p is None:
                    acquire(a, trig); last_action_age = a
                elif p["body_type"] not in FAMILY_BODIES:
                    dispose(a, p, trig); acquire(a, trig); last_action_age = a
            elif action == "upgrade":
                p = primary()
                if p and (birth_year + a - p["acquired_year"]) >= 3:
                    dispose(a, p, trig); acquire(a, trig, used=False); last_action_age = a
                elif p is None:
                    acquire(a, trig, used=False); last_action_age = a
            elif action == "replace":
                p = primary()
                if p and (birth_year + a - p["acquired_year"]) >= 2:
                    dispose(a, p, trig); acquire(a, trig); last_action_age = a
                elif p is None:
                    acquire(a, trig); last_action_age = a
            elif action == "add_second":
                st = household_state(hh, life, a, birth_year)
                if st["partnered"] and len(active) < 2 and res["area_type"] != "urban":
                    acquire(a, trig, role="secondary"); last_action_age = a
            elif action == "add_hobby":
                if sum(1 for v in active if v["role"] == "hobby") == 0:
                    acquire(a, trig, used=random.random() < 0.3, role="hobby"); last_action_age = a
            elif action == "move_check":
                if res["area_type"] == "urban" and res["commute_type"] == "transit" and active and random.random() < 0.35:
                    dispose(a, primary(), "moved", how="sale"); last_action_age = a
                elif res["area_type"] != "urban" and not active:
                    acquire(a, "moved"); last_action_age = a
            elif action == "downsize":
                p = primary()
                if p and p["body_type"] in FAMILY_BODIES | {"pickup"}:
                    dispose(a, p, trig); acquire(a, trig); last_action_age = a
            elif action == "sell_second":
                extras = [v for v in active if v["role"] != "primary"]
                if extras:
                    dispose(a, extras[-1], trig, how="sale"); last_action_age = a
        # natural replacement cycle for primary
        p = primary()
        if p and a - last_action_age >= 2:
            held = birth_year + a - p["acquired_year"]
            limit = 3 if p["acquired_via"] == "lease" else random.randint(7, 10)
            if held >= limit:
                dispose(a, p, "replacement_cycle"); acquire(a, "replacement_cycle"); last_action_age = a
        # secondary cars age out too
        for v in [v for v in active if v["role"] == "secondary"]:
            if birth_year + a - v["acquired_year"] >= 11 and random.random() < 0.5:
                dispose(a, v, "replacement_cycle", how="sale")

    return transactions, vehicles


# ═══════════════════════════════════════════════════════════════════════════
# 6. ANNUAL SPEND
# ═══════════════════════════════════════════════════════════════════════════

RUNNING_BASE = {"compact": 1400, "sedan": 1700, "crossover": 2000, "suv": 2500,
                "minivan": 2300, "pickup": 2600, "sports": 3200, "motorcycle": 800}
FUEL = {"gasoline": 1.0, "diesel": 0.9, "hybrid": 0.65, "ev": 0.35}
COMMUTE_MULT = {"none": 0.5, "wfh": 0.6, "transit": 0.7, "car_short": 1.0, "car_long": 1.6}


def build_annual_spend(vehicles, transactions, residence, age, birth_year) -> list[list]:
    rows = []
    tx_by_year = defaultdict(list)
    for t in transactions:
        tx_by_year[t["year"]].append(t)
    for a in range(18, age + 1):
        y = birth_year + a
        res = residence_at(residence, a)
        running = 0.0
        for v in vehicles:
            if v["acquired_year"] <= y and (v["sold_year"] is None or v["sold_year"] > y):
                vehicle_age = y - v["acquired_year"] + (3 if v["new_or_used"] == "used" else 0)
                maint = RUNNING_BASE[v["body_type"]] * (0.4 + 0.08 * vehicle_age)
                insurance = 450 + 220 * v["price_band"]
                fuel = 1500 * FUEL[v["powertrain"]] * COMMUTE_MULT[res["commute_type"]] * (0.5 if v["role"] == "hobby" else 1)
                running += maint + insurance + fuel
        capital = sum(t["price_usd"] for t in tx_by_year[y] if t["type"] in ("purchase", "lease"))
        capital -= sum(t["price_usd"] for t in tx_by_year[y] if t["type"] in ("sale", "trade_in"))
        accessories = sum(random.randint(300, 1200) for t in tx_by_year[y] if t["type"] in ("purchase", "lease"))
        running = int(running * random.uniform(0.9, 1.1)) + accessories
        rows.append([y, running, max(0, int(capital)), running + max(0, int(capital))])
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 7. DERIVED INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════

def spend_window(spend_rows, year, lo, hi, col=3):
    vals = [r[col] for r in spend_rows if year + lo <= r[0] <= year + hi]
    return round(statistics.mean(vals)) if vals else None


def derive(rec, life, career, residence, hh, transactions, vehicles, spend, age, birth_year):
    year_now = CURRENT_YEAR
    spend_by_year = {r[0]: r for r in spend}
    purchases = [t for t in transactions if t["type"] in ("purchase", "lease")]
    active = [v for v in vehicles if v["sold_year"] is None]

    # ownership intelligence
    trig = Counter(t["trigger_event"] for t in purchases)
    powertrains = [t["powertrain"] for t in purchases]
    switch_year = None
    for t in purchases:
        if t["powertrain"] == "ev":
            switch_year = t["year"]; break
    totals = [r[3] for r in spend]
    last5 = [r[3] for r in spend if r[0] > year_now - 5]
    primary_v = next((v for v in active if v["role"] == "primary"), active[0] if active else None)
    ownership = {
        "n_vehicles_total": len(vehicles),
        "n_transactions": len(transactions),
        "n_purchases": len(purchases),
        "current_vehicle_count": len(active),
        "primary_body_type": primary_v["body_type"] if primary_v else "none",
        "primary_powertrain": primary_v["powertrain"] if primary_v else "none",
        "ever_ev": "ev" in powertrains,
        "powertrain_switch_year": switch_year,
        "avg_price_per_purchase_usd": round(statistics.mean(t["price_usd"] for t in purchases)) if purchases else 0,
        "spend_span": {"min": min(totals), "max": max(totals), "range": max(totals) - min(totals)},
        "avg_annual_spend_5y": round(statistics.mean(last5)) if last5 else 0,
        "lifetime_spend": sum(totals),
        "top_trigger_events": [k for k, _ in trig.most_common(3) if k != "replacement_cycle"],
        "last_transaction_year": transactions[-1]["year"] if transactions else None,
    }

    # decade profile
    decades = {}
    for d in (20, 30, 40, 50, 60, 70):
        if age < d:
            continue
        yrs = [birth_year + a for a in range(d, min(d + 9, age) + 1)]
        vals = [spend_by_year[y][3] for y in yrs if y in spend_by_year]
        if not vals:
            continue
        dp = [t for t in purchases if birth_year + d <= t["year"] <= birth_year + d + 9]
        dtrig = Counter(t["trigger_event"] for t in dp)
        decades[f"{d}s"] = {
            "avg_annual_spend": round(statistics.mean(vals)),
            "n_purchases": len(dp),
            "top_trigger": dtrig.most_common(1)[0][0] if dtrig else None,
            "cluster": None,   # filled in after global terciles are known
            "partial": len(vals) < 10,
            "years_observed": len(vals),
        }

    # temporal cross-intelligence: what each life event did to spend and the next purchase
    impacts = []
    for e in life:
        if e["event"] in ("first_job", "graduation"):
            continue
        nxt = next((t for t in purchases if 0 <= t["year"] - e["year"] <= 3), None)
        before = spend_window(spend, e["year"], -2, -1)
        after = spend_window(spend, e["year"], 0, 2)
        rb = spend_window(spend, e["year"], -2, -1, col=1)
        ra = spend_window(spend, e["year"], 0, 2, col=1)
        impacts.append({
            "event": e["event"], "year": e["year"], "age": e["age"],
            "spend_before_2y": before, "spend_after_2y": after,
            "spend_delta_pct": round((after - before) / before * 100, 1) if before and after else None,
            "running_delta_pct": round((ra - rb) / rb * 100, 1) if rb and ra else None,
            "next_purchase": {"lag_years": nxt["year"] - e["year"], "body_type": nxt["body_type"],
                              "powertrain": nxt["powertrain"], "price_band": nxt["price_band"],
                              "attributed": nxt["trigger_event"] == e["event"]} if nxt else None,
        })
    lags = [i["next_purchase"]["lag_years"] for i in impacts if i["next_purchase"] and i["next_purchase"]["attributed"]]
    last_event_year = max((e["year"] for e in life), default=birth_year + 18)
    temporal = {
        "n_life_events": len([e for e in life if e["event"] not in ("first_job", "graduation")]),
        "n_job_changes": len(career["job_change_years"]),
        "n_promotions": len(career["promotion_years"]),
        "n_career_changes": len(career["career_change_years"]),
        "n_moves": len([e for e in life if e["event"] == "moved"]),
        "n_children": len(hh["children_birth_years"]),
        "years_since_last_event": year_now - last_event_year,
        "event_to_purchase_median_lag": statistics.median(lags) if lags else None,
        "event_impacts": impacts,
    }

    # rfm (raw; scored globally later)
    rfm_raw = {
        "recency_years": (year_now - ownership["last_transaction_year"]) if ownership["last_transaction_year"] else 99,
        "frequency_10y": len([t for t in transactions if t["year"] > year_now - 10]),
        "monetary_5y": sum(last5),
    }
    return ownership, decades, temporal, rfm_raw


def current_demographics(life, career, residence, hh, age, birth_year):
    st = household_state(hh, life, age, birth_year)
    res = residence_at(residence, age)
    retired = career["retirement_year"] is not None
    had_kids = bool(hh["children_birth_years"])
    if retired:
        stage = "retired"
    elif st["n_children_home"] and st["youngest_child_age"] is not None and st["youngest_child_age"] < 6:
        stage = "parent_young_children"
    elif st["n_children_home"]:
        stage = "parent_school_age"
    elif had_kids:
        stage = "empty_nester"
    elif age < 32:
        stage = "early_career"
    else:
        stage = "established_professional"
    if st["partnered"]:
        comp = "partnered_with_children" if st["n_children_home"] else "partnered_no_children"
    else:
        comp = "single_parent" if st["n_children_home"] else "solo"
    inc_idx = career["_income_by_age"][age]
    return {
        "life_stage": stage,
        "household_composition": comp,
        "dual_income": hh["dual_income"] and st["partnered"],
        "n_children_home": st["n_children_home"],
        "ever_married": hh["marriage_year"] is not None,
        "income_bracket": BRACKETS[inc_idx],
        "area_type": res["area_type"],
        "commute_type": res["commute_type"],
        "parking": res["parking"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. GENERATE
# ═══════════════════════════════════════════════════════════════════════════

def generate_household(i: int) -> dict:
    arch = choice_w([(k, w) for k, ((lo, hi), w) in ARCHETYPES.items()])
    lo, hi = ARCHETYPES[arch][0]
    age = random.randint(lo, hi)
    birth_year = CURRENT_YEAR - age

    life = build_life_timeline(arch, age, birth_year)
    career = build_career_timeline(life, age, birth_year)
    residence = build_residence_timeline(life, age, birth_year)
    hh = build_household_timeline(life, age, birth_year)
    transactions, vehicles = build_vehicle_timeline(life, career, residence, hh, age, birth_year)
    spend = build_annual_spend(vehicles, transactions, residence, age, birth_year)
    ownership, decades, temporal, rfm_raw = derive(None, life, career, residence, hh, transactions, vehicles, spend, age, birth_year)
    demo = current_demographics(life, career, residence, hh, age, birth_year)

    career_out = {k: v for k, v in career.items() if not k.startswith("_")}
    hh_out = {k: v for k, v in hh.items() if not k.startswith("_")}

    return {
        "owner_id": f"HH-{i:04d}",
        "age": age,
        "age_group": f"{(age // 10) * 10}s",
        "gender": random.choice(["male", "female", "nonbinary"] if random.random() < 0.04 else ["male", "female"]),
        "archetype": arch,
        "demographics": demo,
        "life_timeline": life,
        "career_timeline": career_out,
        "residence_timeline": residence,
        "household_timeline": hh_out,
        "vehicle_timeline": {
            "transactions": transactions,
            "vehicles": vehicles,
        },
        "annual_spend": spend,                 # [year, running, capital, total]
        "ownership_intelligence": ownership,
        "lifecycle_intelligence": {"decade_profile": decades, "progression_type": None, "rfm": None, "rfm_cluster": None},
        "temporal_cross_intelligence": temporal,
        "_rfm_raw": rfm_raw,
    }


def tercile_edges(values):
    s = sorted(values)
    return s[len(s) // 3], s[2 * len(s) // 3]


def decile_score(values):
    s = sorted(values)
    edges = [s[int(len(s) * k / 10)] for k in range(1, 10)]
    def score(v):
        return 1 + sum(1 for e in edges if v > e)
    return score


def finalize(records: list[dict]):
    # decade clusters from global terciles of decade avg spend
    all_dec = [d["avg_annual_spend"] for r in records for d in r["lifecycle_intelligence"]["decade_profile"].values()]
    t1, t2 = tercile_edges(all_dec)
    for r in records:
        seq = []
        for name, d in r["lifecycle_intelligence"]["decade_profile"].items():
            d["cluster"] = 0 if d["avg_annual_spend"] <= t1 else (1 if d["avg_annual_spend"] <= t2 else 2)
            if d["years_observed"] >= 5:
                seq.append(d["cluster"])
        if len(seq) < 2:
            prog = "insufficient_history"
        else:
            ups = sum(1 for a, b in zip(seq, seq[1:]) if b > a)
            downs = sum(1 for a, b in zip(seq, seq[1:]) if b < a)
            if ups == 0 and downs == 0: prog = "stable"
            elif downs == 0: prog = "ascending"
            elif ups == 0: prog = "descending"
            else: prog = "volatile"
        r["lifecycle_intelligence"]["progression_type"] = prog

    # RFM scores as deciles
    rs = decile_score([-r["_rfm_raw"]["recency_years"] for r in records])   # more recent = higher
    fs = decile_score([r["_rfm_raw"]["frequency_10y"] for r in records])
    ms = decile_score([r["_rfm_raw"]["monetary_5y"] for r in records])
    for r in records:
        raw = r.pop("_rfm_raw")
        R, F, M = rs(-raw["recency_years"]), fs(raw["frequency_10y"]), ms(raw["monetary_5y"])
        prog = r["lifecycle_intelligence"]["progression_type"]
        first_year = r["vehicle_timeline"]["vehicles"][0]["acquired_year"] if r["vehicle_timeline"]["vehicles"] else None
        if raw["recency_years"] >= 9 or r["ownership_intelligence"]["current_vehicle_count"] == 0:
            cluster = "dormant"
        elif first_year and CURRENT_YEAR - first_year <= 3 and r["ownership_intelligence"]["n_purchases"] <= 2:
            cluster = "new_owner"
        elif M >= 7 and prog == "volatile":
            cluster = "high_volatile"
        elif M >= 7:
            cluster = "high_stable"
        elif prog == "ascending" and M >= 4:
            cluster = "rising"
        elif prog == "descending":
            cluster = "declining"
        else:
            cluster = "mid_stable"
        r["lifecycle_intelligence"]["rfm"] = {"recency_score": R, "frequency_score": F, "monetary_score": M, **raw}
        r["lifecycle_intelligence"]["rfm_cluster"] = cluster


# ═══════════════════════════════════════════════════════════════════════════
# 9. VALIDATION REPORT — do the causal chains actually hold?
# ═══════════════════════════════════════════════════════════════════════════

def pct(a, b):
    return f"{(a / b * 100):.1f}%" if b else "n/a"


def validate(records: list[dict]):
    print("\n" + "=" * 72)
    print("VALIDATION REPORT")
    print("=" * 72)

    print("\n[1] Archetype distribution")
    for k, n in Counter(r["archetype"] for r in records).most_common():
        print(f"    {k:<24}{n:>5}  {pct(n, len(records))}")

    print("\n[2] Per-archetype means (job changes / moves / life events / purchases / avg price)")
    by = defaultdict(list)
    for r in records:
        by[r["archetype"]].append(r)
    for k, rs in sorted(by.items()):
        t = [r["temporal_cross_intelligence"] for r in rs]
        o = [r["ownership_intelligence"] for r in rs]
        print(f"    {k:<24} jc={statistics.mean(x['n_job_changes'] for x in t):.2f}  "
              f"mv={statistics.mean(x['n_moves'] for x in t):.2f}  "
              f"ev={statistics.mean(x['n_life_events'] for x in t):.2f}  "
              f"buy={statistics.mean(x['n_purchases'] for x in o):.2f}  "
              f"$/buy={statistics.mean(x['avg_price_per_purchase_usd'] for x in o):>7.0f}")

    print("\n[3] Causal check: new_child -> household holds a family body (suv/minivan) within 2 years")
    n_events = n_family = n_before = 0
    for r in records:
        vs = r["vehicle_timeline"]["vehicles"]
        def holds_family(y):
            return any(v["body_type"] in FAMILY_BODIES and v["acquired_year"] <= y and (v["sold_year"] is None or v["sold_year"] > y) for v in vs)
        for e in r["life_timeline"]:
            if e["event"] == "new_child":
                n_events += 1
                if holds_family(e["year"] - 1): n_before += 1
                if any(holds_family(e["year"] + k) for k in (0, 1, 2)): n_family += 1
    print(f"    held family body the year before: {pct(n_before, n_events)}   within 2y after: {pct(n_family, n_events)}   (n={n_events})")
    print("    (control) all households currently holding a family body: " + pct(sum(1 for r in records if r["ownership_intelligence"]["primary_body_type"] in FAMILY_BODIES), len(records)))

    print("\n[4] Causal check: promotion -> price band on next purchase vs previous purchase")
    ups = total = 0
    for r in records:
        tx = [t for t in r["vehicle_timeline"]["transactions"] if t["type"] in ("purchase", "lease")]
        for py in r["career_timeline"]["promotion_years"]:
            prev = [t for t in tx if t["year"] < py]
            nxt = [t for t in tx if py <= t["year"] <= py + 3]
            if prev and nxt:
                total += 1
                if nxt[0]["price_band"] > prev[-1]["price_band"]:
                    ups += 1
    print(f"    band increased in {ups}/{total} = {pct(ups, total)} cases with a purchase within 3y")

    print("\n[5] Causal check: spend delta after life events (mean %, 2y after vs 2y before)")
    deltas = defaultdict(list); rdeltas = defaultdict(list)
    for r in records:
        for imp in r["temporal_cross_intelligence"]["event_impacts"]:
            if imp["spend_delta_pct"] is not None:
                deltas[imp["event"]].append(imp["spend_delta_pct"])
            if imp["running_delta_pct"] is not None:
                rdeltas[imp["event"]].append(imp["running_delta_pct"])
    print(f"    {'event':<16} {'n':>6}  {'total median':>13}  {'running median':>15}")
    for ev, vals in sorted(deltas.items(), key=lambda kv: -statistics.median(kv[1])):
        print(f"    {ev:<16} {len(vals):>6}  {statistics.median(vals):>+12.1f}%  {statistics.median(rdeltas[ev]):>+14.1f}%")

    print("\n[6] Decade cluster: empty_nester archetype, 40s vs 50s (mean cluster 0..2)")
    c40 = [r["lifecycle_intelligence"]["decade_profile"]["40s"]["cluster"] for r in by["empty_nester"] if "40s" in r["lifecycle_intelligence"]["decade_profile"]]
    c50 = [r["lifecycle_intelligence"]["decade_profile"]["50s"]["cluster"] for r in by["empty_nester"] if "50s" in r["lifecycle_intelligence"]["decade_profile"]]
    print(f"    40s={statistics.mean(c40):.2f}   50s={statistics.mean(c50):.2f}")

    print("\n[7] EV share of purchases by period")
    per = defaultdict(lambda: [0, 0])
    for r in records:
        for t in r["vehicle_timeline"]["transactions"]:
            if t["type"] in ("purchase", "lease"):
                p = "<2012" if t["year"] < 2012 else "2012-17" if t["year"] < 2018 else "2018-21" if t["year"] < 2022 else "2022+"
                per[p][1] += 1
                if t["powertrain"] == "ev":
                    per[p][0] += 1
    for p in ["<2012", "2012-17", "2018-21", "2022+"]:
        print(f"    {p:<8} {pct(per[p][0], per[p][1])}  ({per[p][1]} purchases)")

    print("\n[8] Progression / RFM cluster distribution")
    print("    " + ", ".join(f"{k}={n}" for k, n in Counter(r["lifecycle_intelligence"]["progression_type"] for r in records).most_common()))
    print("    " + ", ".join(f"{k}={n}" for k, n in Counter(r["lifecycle_intelligence"]["rfm_cluster"] for r in records).most_common()))

    print("\n[9] Query 4 preview — per-transaction price: single 40s no kids vs 30s dual-income parents")
    def seg(pred):
        prices = [t["price_usd"] for r in records if pred(r) for t in r["vehicle_timeline"]["transactions"] if t["type"] in ("purchase", "lease")]
        people = [r for r in records if pred(r)]
        return statistics.mean(prices) if prices else 0, len(prices), len(people)
    a = seg(lambda r: r["age_group"] == "40s" and r["demographics"]["household_composition"] == "solo" and not r["demographics"]["ever_married"])
    b = seg(lambda r: r["age_group"] == "30s" and r["demographics"]["dual_income"] and r["demographics"]["n_children_home"] > 0)
    print(f"    single 40s never-married : ${a[0]:,.0f} per purchase  ({a[1]} purchases, {a[2]} people)")
    print(f"    30s dual-income parents  : ${b[0]:,.0f} per purchase  ({b[1]} purchases, {b[2]} people)")

    print("\n[10] Sample record sanity — one dual_income_parents household")
    r = next(r for r in by["dual_income_parents"] if r["temporal_cross_intelligence"]["n_children"] >= 2 and r["career_timeline"]["promotion_years"])
    print(f"    {r['owner_id']} age {r['age']} {r['demographics']['life_stage']} income={r['demographics']['income_bracket']}")
    for e in r["life_timeline"]:
        print(f"      {e['year']} (age {e['age']:>2}) {e['event']}" + (f" [{e.get('cause')}]" if e.get("cause") else ""))
    print("    transactions:")
    for t in r["vehicle_timeline"]["transactions"]:
        print(f"      {t['year']} {t['type']:<9} {t['body_type']:<10} {t['powertrain']:<8} band={t['price_band']} ${t['price_usd']:>6,}  <- {t['trigger_event']}")
    print(f"    decade profile: { {k: (v['cluster'], v['avg_annual_spend']) for k, v in r['lifecycle_intelligence']['decade_profile'].items()} }")
    print(f"    progression={r['lifecycle_intelligence']['progression_type']}  rfm={r['lifecycle_intelligence']['rfm_cluster']}  "
          f"top_triggers={r['ownership_intelligence']['top_trigger_events']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    random.seed(args.seed)

    records = [generate_household(i + 1) for i in range(args.n)]
    finalize(records)

    out = Path(args.out) if args.out else Path(__file__).parent.parent / "pipeline" / "data" / "households.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
    n_tx = sum(r["ownership_intelligence"]["n_transactions"] for r in records)
    print(f"Generated {len(records):,} households, {n_tx:,} transactions -> {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    validate(records)


if __name__ == "__main__":
    main()
