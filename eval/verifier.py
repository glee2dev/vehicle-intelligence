"""
Verifier: turns "we told the model not to hallucinate" into "we measured whether it did".

    extract_claims(text)          -> every number the answer asserts, with its span and kind
    flatten_truth(stats)          -> every number the pipeline actually produced, with its path
    match_claims(claims, truth)   -> verified / derived / unverified per claim
    detect_external(text)         -> phrases that import knowledge from outside the dataset
    detect_refusal(text)          -> did the answer say the question can't be answered from the data
    score_output(...)             -> one dict per (query, mode) for the scoreboard

Matching tolerance: a claim matches a truth value if |claim - truth| <= max(half a unit at the
claim's stated precision, 2.5% of the truth). "29%" against 29.1 counts; "30%" does not — rounding must be at the stated precision.
"derived" = equals the difference or ratio of two truth values of the same kind (the grounding
rules permit trivial arithmetic on given figures).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

# ── Claim extraction ──────────────────────────────────────────────────────────

# order matters: dollar and percent forms first so plain integers don't steal them
_NUM = r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
CLAIM_PATTERNS = [
    ("usd", re.compile(r"(?:\$|USD\s?)\s?" + _NUM + r"\s?(?P<suffix>[kKmM])?(?!\s?%)")),
    ("usd", re.compile(_NUM + r"\s?(?P<suffix>[kKmM])?\s?(?:USD|dollars)\b")),
    ("pct", re.compile(_NUM + r"\s?(?:%|percent(?:age)?\b(?!\s+points))")),
    ("pts", re.compile(_NUM + r"\s?(?:percentage\s+points?|pts?\b|points?\b)")),
    ("count", re.compile(_NUM + r"(?=\s+(?:households?|purchases?|events?|owners?|records?|transactions?|vehicles?|people|respondents?)\b)", re.I)),
    ("plain", re.compile(r"(?<![\d.,$%])" + _NUM + r"(?![\d.,%]|\s?%|\s?(?:k|m)\b)", re.I)),
]
YEAR_RANGE = (1980, 2035)
SMALL_PLAIN_MAX = 12          # a bare "3" is not a claim; a bare "577" is


@dataclass
class Claim:
    text: str
    value: float
    kind: str            # usd | pct | pts | count | plain
    start: int
    end: int
    context: str
    status: str = "unverified"          # verified | derived | unverified | ignored
    matched_path: str | None = None
    matched_value: float | None = None
    denominator_stated: bool | None = None


def _to_float(num: str, suffix: str | None) -> float:
    v = float(num.replace(",", ""))
    if suffix and suffix.lower() == "k":
        v *= 1_000
    elif suffix and suffix.lower() == "m":
        v *= 1_000_000
    return v


def extract_claims(text: str) -> list[Claim]:
    taken: list[tuple[int, int]] = []
    claims: list[Claim] = []

    def overlaps(a, b):
        return any(not (b <= s or a >= e) for s, e in taken)

    for kind, pat in CLAIM_PATTERNS:
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            if overlaps(s, e):
                continue
            v = _to_float(m.group("num"), m.groupdict().get("suffix"))
            if kind == "plain":
                if YEAR_RANGE[0] <= v <= YEAR_RANGE[1] and float(v).is_integer():
                    continue                       # a year, not a claim
                if v <= SMALL_PLAIN_MAX and float(v).is_integer():
                    continue                       # "2 years", "3 opportunities"
                # ordinal / list markers like "1." or "(2)"
                after = text[e:e + 1]
                if after in (".", ")") and v < 20:
                    continue
            taken.append((s, e))
            ctx = text[max(0, s - 60): min(len(text), e + 60)].replace("\n", " ")
            raw = m.group(0)
            e = s + len(raw.rstrip())
            c = Claim(text=raw.strip(), value=v, kind=kind, start=s, end=e, context=ctx)
            if kind == "pct":
                # denominator must be stated in the same sentence as the percentage
                sb = max((text.rfind(ch, 0, s) for ch in ".!?\n"), default=-1) + 1
                se_candidates = [i for i in (text.find(ch, e) for ch in ".!?\n") if i != -1]
                se = min(se_candidates) if se_candidates else len(text)
                window = text[sb:se].lower()
                c.denominator_stated = bool(re.search(
                    r"\b(of|among|out of)\b.{0,40}\b(households?|purchases?|events?|owners?|records?|transactions?|segment|cohort|sample)\b|"
                    r"\b(households?|purchases?|events?|owners?|records?|transactions?)\b.{0,15}\b(had|have|are|were|hold|switched|bought)\b|"
                    r"n\s?=\s?\d", window))
            claims.append(c)
    claims.sort(key=lambda c: c.start)
    return claims


# ── Truth flattening ──────────────────────────────────────────────────────────

_PCT_HINT = re.compile(r"(pct|percent|share|rate)", re.I)
_USD_HINT = re.compile(r"(usd|price|spend|amount|range|min|max)", re.I)
_SKIP_KEYS = {"processing_ms", "cached", "window_before_years", "window_after_years", "window_years",
              "min_count", "years_observed", "year", "age", "temperature", "match_pct_of_sample"}


@dataclass
class TruthValue:
    path: str
    value: float
    kind: str    # pct | usd | count


def flatten_truth(stats: dict, prefix: str = "") -> list[TruthValue]:
    out: list[TruthValue] = []

    def walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _SKIP_KEYS or k in ("filters", "params", "of", "spec"):
                    continue
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
        elif isinstance(obj, bool):
            return
        elif isinstance(obj, (int, float)) and obj is not None:
            tail = path.rsplit(".", 1)[-1]
            parent = path.rsplit(".", 2)[-2] if path.count(".") >= 1 else ""
            kind = "pct" if (_PCT_HINT.search(tail) or _PCT_HINT.search(parent) or tail.endswith("_pct")) else \
                   "usd" if (_USD_HINT.search(tail) or _USD_HINT.search(parent)) else "count"
            out.append(TruthValue(path, float(obj), kind))

    walk(stats, prefix)
    return out


# ── Matching ──────────────────────────────────────────────────────────────────

def _decimals(text: str) -> int:
    m = re.search(r"\.(\d+)", text.replace(",", ""))
    return len(m.group(1)) if m else 0


def _close(claim_val: float, claim_text: str, truth: float, rel: float = 0.025) -> bool:
    tol = max(0.5 * 10 ** (-_decimals(claim_text)), rel * abs(truth))
    if re.search(r"[kKmM]\b", claim_text):          # "$26k" — coarser
        tol = max(tol, 0.05 * abs(truth))
    return abs(claim_val - truth) <= tol


def _compatible(claim_kind: str, truth_kind: str) -> bool:
    if claim_kind == "pct":
        return truth_kind == "pct"
    if claim_kind == "pts":
        return truth_kind == "pct"
    if claim_kind == "usd":
        return truth_kind == "usd"
    return True                                     # bare numbers may be counts, dollars, or unlabelled percentages


def match_claims(claims: list[Claim], truth: list[TruthValue]) -> list[Claim]:
    scalars = [t for t in truth if not re.search(r"\[\d+\]$", t.path) or True]
    for c in claims:
        if c.status == "ignored":
            continue
        best = None
        for t in truth:
            if _compatible(c.kind, t.kind) and _close(c.value, c.text, t.value):
                if best is None or abs(c.value - t.value) < abs(c.value - best.value):
                    best = t
        if best:
            c.status, c.matched_path, c.matched_value = "verified", best.path, best.value
            continue
        # derived: difference or ratio of two same-kind truth values
        same = [t for t in scalars if _compatible(c.kind, t.kind)]
        found = None
        for i, a in enumerate(same):
            for b in same[i + 1:]:
                d = abs(a.value - b.value)
                if d > 0 and _close(c.value, c.text, d):
                    found = (f"|{a.path} - {b.path}|", d); break
                if c.kind in ("plain", "count") and b.value and _close(c.value, c.text, a.value / b.value):
                    found = (f"{a.path} / {b.path}", a.value / b.value); break
                if c.kind == "pct" and a.kind != "pct" and b.value and _close(c.value, c.text, a.value / b.value * 100):
                    found = (f"{a.path} / {b.path} * 100", a.value / b.value * 100); break
            if found:
                break
        if found:
            c.status, c.matched_path, c.matched_value = "derived", found[0], round(found[1], 2)
        else:
            c.status = "unverified"
    return claims


# ── Language checks ───────────────────────────────────────────────────────────

EXTERNAL_PATTERNS = [
    r"\btypically\b", r"\bgenerally\b", r"\bresearch (?:shows|suggests|indicates)\b", r"\bstudies (?:show|suggest)\b",
    r"\bindustry (?:average|benchmark|data|norms?|trends?)\b", r"\bnational(?:ly)? (?:average|rate|figure|data|level)\b",
    r"\bin the (?:u\.?s\.?|united states|us market)\b", r"\baccording to\b(?! the (?:data|dataset|records|statistics|json|pipeline))",
    r"\bbenchmark\b", r"\bmarket data\b", r"\bconsumers? (?:tend|usually|often|generally)\b",
    r"\b(?:it is|it's) (?:well[- ])?known\b", r"\bcommon(?:ly)? (?:known|observed|seen)\b", r"\bon average,? (?:people|consumers|americans|households) \b",
    r"\bcensus\b", r"\bbls\b", r"\bfederal\b", r"\bkelley blue book\b", r"\bedmunds\b", r"\bj\.?d\.? power\b",
]
_EXT = [re.compile(p, re.I) for p in EXTERNAL_PATTERNS]

REFUSAL_PATTERNS = [
    r"cannot be (?:answered|computed|determined|calculated)", r"can(?:no|')t be (?:answered|computed|determined|calculated)",
    r"not (?:possible|able) to (?:answer|compute|determine|calculate)", r"(?:does|do) not (?:contain|include|have|record)",
    r"(?:is|are) not (?:available|recorded|present|in the data|in the dataset|included)", r"no (?:such|external|national|numeric|dollar) ",
    r"(?:isn't|is not|aren't|are not) (?:in|part of) (?:the|this) (?:data|dataset|records)", r"unanswerable",
    r"(?:only|solely) (?:as|in) (?:a )?(?:bracket|categor)", r"(?:would|could) (?:be|require) (?:invent|guess|speculat|assum|external)",
]
_REF = [re.compile(p, re.I) for p in REFUSAL_PATTERNS]


_NEGATED = re.compile(r"\b(no|not|cannot|can't|without|lacks?|absent|invented?|unavailable|does not|doesn't|isn't|aren't|"
                      r"would be (?:invent|guess)|is not (?:in|part of|available))\b|question:", re.I)


def _sentence(text: str, pos: int) -> str:
    sb = max((text.rfind(ch, 0, pos) for ch in ".!?\n"), default=-1) + 1
    ends = [i for i in (text.find(ch, pos) for ch in ".!?\n") if i != -1]
    return text[sb:(min(ends) if ends else len(text))]


def detect_external(text: str, include_negated: bool = False) -> list[dict]:
    """External-knowledge phrases. A phrase inside a sentence that negates it ("contains no national
    benchmark") or restates the question is a *refusal*, not an import — those are returned only when
    include_negated=True, tagged negated=True, and never counted."""
    hits = []
    for p in _EXT:
        for m in p.finditer(text):
            sent = _sentence(text, m.start())
            neg = bool(_NEGATED.search(sent))
            if neg and not include_negated:
                continue
            hits.append({"phrase": m.group(0), "start": m.start(), "end": m.end(), "negated": neg,
                         "context": text[max(0, m.start() - 60): m.end() + 60].replace("\n", " ")})
    hits.sort(key=lambda h: h["start"])
    return hits


def detect_refusal(text: str) -> bool:
    return any(p.search(text) for p in _REF)


# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class Score:
    mode: str
    n_claims: int
    n_verified: int
    n_derived: int
    n_unverified: int
    grounding_rate: float | None          # (verified + derived) / n_claims
    unverified_rate: float | None
    n_external_flags: int
    pct_claims: int
    pct_with_denominator: float | None
    unanswerable_expected: bool
    refusal_detected: bool | None         # only meaningful when unanswerable_expected
    fabricated_on_unanswerable: int | None
    truth_basis: str                      # population | sample
    claims: list[dict] = field(default_factory=list)
    external_flags: list[dict] = field(default_factory=list)
    negated_mentions: list[dict] = field(default_factory=list)


def score_output(text: str, mode: str, stats: dict, sample_stats: dict | None = None) -> Score:
    """stats = population ground truth (pipeline result). sample_stats = the same spec recomputed on
    the exact records mode 2 received; when given, claims are matched against BOTH and the sample
    is the primary basis (a correct computation on its sample is a correct answer)."""
    basis = "sample" if sample_stats is not None else "population"
    truth = flatten_truth(stats)
    if sample_stats is not None:
        truth = flatten_truth(sample_stats, "sample") + truth
    claims = match_claims(extract_claims(text), truth)
    ext_all = detect_external(text, include_negated=True)
    ext = [h for h in ext_all if not h["negated"]]

    unans = not stats.get("answerable", True)
    refusal = detect_refusal(text) if unans else None
    fabricated = None
    if unans:
        # any unverified $ or % claim on an unanswerable question is a fabricated figure
        fabricated = sum(1 for c in claims if c.status == "unverified" and c.kind in ("usd", "pct"))

    n = len([c for c in claims if c.status != "ignored"])
    nv = sum(1 for c in claims if c.status == "verified")
    nd = sum(1 for c in claims if c.status == "derived")
    nu = sum(1 for c in claims if c.status == "unverified")
    pcts = [c for c in claims if c.kind == "pct"]
    return Score(
        mode=mode, n_claims=n, n_verified=nv, n_derived=nd, n_unverified=nu,
        grounding_rate=round((nv + nd) / n, 3) if n else None,
        unverified_rate=round(nu / n, 3) if n else None,
        n_external_flags=len(ext), pct_claims=len(pcts),
        pct_with_denominator=round(sum(1 for c in pcts if c.denominator_stated) / len(pcts), 3) if pcts else None,
        unanswerable_expected=unans, refusal_detected=refusal, fabricated_on_unanswerable=fabricated,
        truth_basis=basis, claims=[asdict(c) for c in claims], external_flags=ext,
        negated_mentions=[h for h in ext_all if h["negated"]],
    )
