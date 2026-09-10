"""Verifier gaps found in the hand-check across eight runs. Each was a correct number scored unverified."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from check import extract_claims, score_output, _sentence

STATS = {"n_total": 10000, "n_matched": 1041, "match_pct": 10.4, "answerable": True,
         "groups": {"retired": {"n_households": 1041, "mean_range_usd": 57705.0, "mean_max_usd": 57756.0},
                    "empty_nester": {"n_households": 1371, "mean_range_usd": 54433.0}}}


def texts(claims):
    return [c.text for c in claims if c.status != "ignored"]


def test_household_id_is_not_a_claim():
    assert "0959" not in texts(extract_claims("HH-0959 spent $7,677; HH‑1201 and HH–9344 likewise."))


def test_dollar_followed_by_word_starting_with_m_is_not_millions():
    c = [x for x in extract_claims("Retired: $57,705 mean spending span; $1,620 more per purchase.") if x.kind == "usd"]
    assert [x.value for x in c] == [57705.0, 1620.0]


def test_bare_m_suffix_still_means_millions():
    assert extract_claims("about $1.2m in total")[0].value == 1_200_000


def test_abbreviation_does_not_end_sentence():
    t = "HH-9220 spent $23,746 in the 40s vs. $83,999 in the 50s."
    assert "HH-9220" in _sentence(t, t.index("$83,999"))


def test_repeated_number_counts_once():
    s = score_output("Compact 87/150 (58%), sedan 50/150 (33%), crossover 13/150 (9%).", "naive_grounded", STATS)
    assert sum(1 for c in s.claims if c["text"] == "150" and c["status"] != "ignored") == 1


def test_zero_bucket_share_is_derived():
    s = score_output("upper: 0 households (0%) out of 1,041 households", "pipeline_grounded", STATS)
    assert [c["status"] for c in s.claims if c["text"] == "0%"] == ["derived"]


def test_years_after_preposition_are_not_counts():
    assert "2025" not in texts(extract_claims("events in 2025 and purchases since 2024"))


def test_pipeline_values_verify():
    s = score_output("Retired (1,041 households) span $57,705, $3,272 wider than empty nesters ($54,433).", "pipeline_grounded", STATS)
    assert s.n_unverified == 0 and s.n_claims == 4
