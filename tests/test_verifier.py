import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from verifier import extract_claims, flatten_truth, match_claims, detect_external, detect_refusal, score_output

STATS = {"n_total": 10000, "n_matched": 6251, "match_pct": 62.5, "answerable": True,
         "groups": {"all": {"n_households": 6251, "n_matching": 1817, "share_pct": 29.1}},
         "answerable_part": {"n_matched": 714, "groups": {"all": {"n_households": 714, "n_matching": 120, "share_pct": 16.8}}}}


def test_extraction_kinds_and_skips():
    cs = extract_claims("In 2024, 29.1% of 6,251 households (n=1,817) paid about $26k; 3 opportunities; item 2. done. 12.3 percentage points.")
    kinds = [(c.text, c.kind) for c in cs]
    assert ("29.1%", "pct") in kinds and ("6,251", "count") in kinds and ("$26k", "usd") in kinds
    assert ("12.3 percentage points", "pts") in kinds
    assert not any(c.value == 2024 for c in cs)          # year skipped
    assert not any(c.value == 3 for c in cs)             # small bare int skipped
    assert any(c.value == 1817 for c in cs)


def test_matching_tolerance_and_kinds():
    truth = flatten_truth(STATS)
    cs = match_claims(extract_claims("29.1% verified; 29% rounds fine; 30% is a misstatement; 1,817 households; $1,817 is wrong kind"), truth)
    by = {c.text: c.status for c in cs}
    assert by["29.1%"] == "verified" and by["29%"] == "verified" and by["30%"] == "unverified"
    assert by["1,817"] == "verified" and by["$1,817"] == "unverified"


def test_derived_difference():
    truth = flatten_truth(STATS)
    cs = match_claims(extract_claims("a gap of 12.3 percentage points; 1,697 more households"), truth)
    assert cs[0].status == "derived" and cs[1].status == "derived"   # 29.1-16.8 ; 1817-120


def test_external_and_refusal():
    assert len(detect_external("Research shows consumers typically buy SUVs; according to the dataset, 29%.")) == 2
    assert detect_external("according to the data, 29%") == []
    assert detect_external("The dataset contains no national benchmark, so no comparison is possible.") == []
    assert len(detect_external("The dataset contains no national benchmark.", include_negated=True)) == 1


def test_unlabelled_percentage_matches():
    cs = match_claims(extract_claims("share_pct came to 29.1 in this segment"), flatten_truth(STATS))
    assert cs[0].status == "verified"
    assert detect_refusal("This cannot be answered from the dataset because income is recorded only as a bracket.")
    assert not detect_refusal("29.1% of households switched.")


def test_score_unanswerable_fabrication():
    unans = {"answerable": False, "unanswerable_reason": "no dollar income", "n_matched": 1041, "n_total": 10000, "match_pct": 10.4,
             "groups": {}, "answerable_part": {"groups": {"all": {"income_bracket": {"lower": 169, "upper_middle": 488}}}}}
    good = score_output("This cannot be computed: income is recorded only as a bracket. 488 households are upper_middle.", "m", unans)
    bad = score_output("Retired owners earn about $48,000 on average, roughly 15% below the national average.", "m", unans)
    assert good.refusal_detected and good.fabricated_on_unanswerable == 0 and good.n_unverified == 0
    assert not bad.refusal_detected and bad.fabricated_on_unanswerable == 2 and bad.n_external_flags >= 1


def test_denominator_detection():
    s = score_output("29.1% of 6,251 households switched. Separately, 16.8% did.", "m", STATS)
    pcts = [c for c in s.claims if c["kind"] == "pct"]
    assert pcts[0]["denominator_stated"] is True and pcts[1]["denominator_stated"] is False


def test_sample_basis_is_primary():
    sample = {"answerable": True, "n_matched": 200, "n_total": 200, "match_pct": 100.0, "groups": {"all": {"share_pct": 31.5, "n_households": 200}}}
    s = score_output("31.5% of the 200 households in the sample switched.", "naive_grounded", STATS, sample)
    assert s.truth_basis == "sample" and s.n_unverified == 0 and s.claims[0]["matched_path"].startswith("sample")
